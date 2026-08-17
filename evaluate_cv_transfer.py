# =============================================================================
# evaluate_cv_transfer.py  —  Zero-Shot CV Transfer Evaluation (Day 16)
# =============================================================================
# PURPOSE:
#   Loads the EfficientNet-B0 checkpoint trained in
#   notebooks/train_cv_classifier_colab.ipynb (on FaceForensics++, gated at
#   >=90% held-out video-level accuracy there before it was ever saved) and
#   evaluates it ZERO-SHOT — no fine-tuning, no exposure to this data during
#   training — against this project's own 32 CV-eligible Nigerian video
#   clips. This is the CV counterpart to the NLP engine's Day 3 finding that
#   DistilGPT-2 perplexity "runs backwards" on Nigerian-register text
#   (docs/PROJECT_CONTEXT.md §4) — both are direct tests of §2.3.4's claim
#   that Western-trained AI-detection tooling degrades on Global South media.
#
# THE RESULT IS THE GAP, NOT THE RAW NUMBER:
#   A low accuracy here, on its own, proves nothing — it could mean the
#   model is bad, or the Nigerian corpus is hard, or both. What's actually
#   interpretable is the DROP relative to the notebook's own FF++ held-out
#   number (loaded from config.CV_MODEL_FF_METRICS_PATH, if present): the
#   same architecture, the same preprocessing, the same aggregation method,
#   evaluated in-domain vs. out-of-domain. A model that clears FF++'s gate
#   and then collapses here is not a failed evaluation — that collapse IS
#   the finding this script exists to measure.
#
# WHY matched_subset IS PRIMARY, NOT THE FULL 32-CLIP CORPUS:
#   Same reasoning as evaluate_cv_heuristics.py, restated precisely per
#   compute_strata.py's actual definition: the full corpus is confounded on
#   platform/orientation (most authentic samples are landscape YouTube, most
#   synthetic samples are portrait Instagram — see compute_strata.py's
#   MATCHED-SUBSET FLAG note and docs/PROJECT_CONTEXT.md §4). `matched_subset`
#   (Instagram + portrait, 4 authentic / 9 synthetic) is the one cell where
#   both labels have real representation, so it's the only subset where a
#   score difference is more likely to reflect synthesis artifacts than
#   "which platform/aspect-ratio is this". Full-corpus numbers are reported
#   only as secondary, confounded context — never the primary result.
#
# PREPROCESSING PARITY WITH THE COLAB NOTEBOOK:
#   The notebook can't cleanly import this project's package (see its own
#   markdown on this), so it manually duplicates letterbox()/detect_face_bbox().
#   This script has no such constraint — it imports the REAL
#   watchdog.cv_engine.detect_face_bbox() / letterbox() / _padded_crop()
#   directly, so any future change to the real detection logic is inherited
#   here automatically rather than needing a second manual edit. Both places
#   use the same frame-sampling count (config.CV_TRANSFER_FRAMES_PER_CLIP)
#   and the same clip-level aggregation (mean P(synthetic) across valid face
#   crops, threshold 0.5) specifically so the measured gap reflects domain
#   transfer, not a methodology difference between the two evaluations.
#
# USAGE:
#   python evaluate_cv_transfer.py
#   (requires models/cv_watchdog_v1.pt downloaded from the Colab notebook's
#   Drive output first — exits with a clear message if it's missing)
#
# OUTPUT:
#   - Per-clip table printed to the terminal, matched_subset first
#   - results/cv_transfer_eval.csv          — per-clip detail
#   - results/cv_transfer_eval_summary.csv  — matched_subset + full-corpus metrics
#   - The FF++-vs-Nigerian-corpus gap, printed prominently, if
#     config.CV_MODEL_FF_METRICS_PATH is present
# =============================================================================

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _sample_frame_indices(n_total_frames: int, n_samples: int) -> list:
    """Evenly-spaced frame indices — identical logic to the Colab notebook's
    sample_frame_indices(), so both places sample frames the same way."""
    if n_total_frames <= 0:
        return []
    n_samples = min(n_samples, n_total_frames)
    return sorted(set(int(i) for i in np.linspace(0, n_total_frames - 1, n_samples)))


def _extract_face_crops(video_path: str, n_samples: int, crop_size: tuple) -> list:
    """
    Real watchdog.cv_engine detection, reused (not duplicated — see module
    docstring). Each sampled frame is independent (no prev_box continuity
    tracking across samples — same simplification the notebook makes, for
    the same reason: these are independent training/eval examples, not a
    continuous video track).
    """
    from watchdog.cv_engine import detect_face_bbox, letterbox, _padded_crop, _CROP_MARGIN_FRAC

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crops = []
    for idx in _sample_frame_indices(n_total, n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        box = detect_face_bbox(frame, prev_box=None)
        if box is None:
            continue
        padded = _padded_crop(frame, box, _CROP_MARGIN_FRAC)
        if padded is None:
            continue
        crop, _origin = padded
        squared, _, _ = letterbox(crop, crop_size)
        crops.append(squared)
    cap.release()
    return crops


def _crops_to_batch(crops: list) -> "torch.Tensor":
    import torch

    batch = np.stack(crops).astype(np.float32) / 255.0        # (N, H, W, 3) BGR, [0,1]
    batch = batch[..., ::-1]                                   # BGR -> RGB
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    batch = np.transpose(batch, (0, 3, 1, 2)).copy()            # NHWC -> NCHW
    return torch.from_numpy(batch).float()


def load_cv_classifier():
    """
    Loads config.CV_MODEL_PATH (a checkpoint dict shaped like
    train_nlp_classifier.py's: model_state_dict/base_model/label_map — see
    notebooks/train_cv_classifier_colab.ipynb's save cell). CPU-only per
    CLAUDE.md's hard constraint — this script never assumes local CUDA.
    """
    import torch
    from torchvision.models import efficientnet_b0
    from config import CV_MODEL_PATH

    if not CV_MODEL_PATH.exists():
        logger.error(
            f"No CV classifier checkpoint at {CV_MODEL_PATH}. Run "
            f"notebooks/train_cv_classifier_colab.ipynb in Colab first, then download "
            f"cv_watchdog_v1.pt (and ideally cv_watchdog_v1_ff_metrics.json) into models/."
        )
        sys.exit(1)

    checkpoint = torch.load(CV_MODEL_PATH, map_location="cpu", weights_only=False)
    assert checkpoint["base_model"] == "efficientnet_b0", (
        f"Unexpected base_model in checkpoint: {checkpoint['base_model']!r}"
    )

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    crop_size = tuple(checkpoint.get("input_size", (224, 224)))
    return model, crop_size


def score_clip(model, video_path: str, crop_size: tuple) -> dict:
    """
    Returns {"p_synthetic": float or None, "n_crops": int, "abstain_reason": str or None}.
    Abstains — same conservative posture as every other CV signal in this
    project (NFR-01) — when fewer than config.CV_TRANSFER_MIN_FACE_CROPS
    valid face crops were found, rather than average 1-2 crops into a
    confident-looking clip-level score.
    """
    import torch
    from config import CV_TRANSFER_FRAMES_PER_CLIP, CV_TRANSFER_MIN_FACE_CROPS

    crops = _extract_face_crops(video_path, CV_TRANSFER_FRAMES_PER_CLIP, crop_size)
    if len(crops) < CV_TRANSFER_MIN_FACE_CROPS:
        return {
            "p_synthetic": None, "n_crops": len(crops),
            "abstain_reason": (f"only {len(crops)} valid face crop(s) out of "
                                f"{CV_TRANSFER_FRAMES_PER_CLIP} sampled frames, below the "
                                f"{CV_TRANSFER_MIN_FACE_CROPS}-crop floor"),
        }

    batch = _crops_to_batch(crops)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[:, 1]
    return {"p_synthetic": float(probs.mean().item()), "n_crops": len(crops), "abstain_reason": None}


# =============================================================================
# Metrics (mirrors evaluate_cv_heuristics.py's / train_nlp_classifier.py's shape)
# =============================================================================

def _compute_metrics(df, subset_name: str) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    scored = df.dropna(subset=["p_synthetic"])
    if scored.empty:
        return {"subset": subset_name, "n": len(df), "n_scored": 0, "n_abstained": len(df)}

    y_true = (scored["true_label"] == "SYNTHETIC").astype(int)
    y_pred = scored["predicted_label_int"]

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    n_authentic = tn + fp

    return {
        "subset": subset_name, "n": len(df), "n_scored": len(scored), "n_abstained": len(df) - len(scored),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr_authentic": (fp / n_authentic) if n_authentic else float("nan"),
        "n_authentic": n_authentic, "n_fp_authentic": fp,
    }


def _print_ff_comparison(nigerian_metrics: dict):
    from config import CV_MODEL_FF_METRICS_PATH, CV_TRANSFER_FF_ACCURACY_GATE

    if not CV_MODEL_FF_METRICS_PATH.exists():
        print(f"\n(No FF++ metrics file at {CV_MODEL_FF_METRICS_PATH} — run/re-download from "
              f"the Colab notebook to see the transfer gap here. Nigerian-corpus numbers above "
              f"stand on their own but can't be compared to an in-domain baseline yet.)")
        return

    with open(CV_MODEL_FF_METRICS_PATH) as f:
        ff_metrics = json.load(f)
    ff_acc = ff_metrics["video_level"]["accuracy"]
    nigerian_acc = nigerian_metrics.get("accuracy")

    print("\n" + "#" * 90)
    print("  THE RESULT — FF++ (in-domain) vs. Nigerian corpus (zero-shot transfer)")
    print("#" * 90)
    print(f"  FF++ held-out video-level accuracy (gate was {CV_TRANSFER_FF_ACCURACY_GATE:.2f}): {ff_acc:.4f}")
    if nigerian_acc is None:
        print(f"  Nigerian matched_subset: no clips could be scored (all abstained) — no gap computable.")
        return
    gap = ff_acc - nigerian_acc
    print(f"  Nigerian matched_subset accuracy:                                    {nigerian_acc:.4f}")
    print(f"  GAP (FF++ minus Nigerian):                                           {gap:+.4f}")
    if gap > 0.15:
        print("\n  A drop this large is consistent with §2.3.4's claim that Western-trained "
              "detection tooling transfers poorly to Global South media — the CV counterpart "
              "to the NLP engine's Day 3 perplexity finding. Report this gap directly in "
              "Chapter 4, not just the Nigerian-corpus number in isolation.")
    print("#" * 90)


def run_evaluation():
    import pandas as pd
    from config import MANIFEST_PATH, DATA_DIR
    from watchdog.dataset_manager import LABEL_AUTHENTIC, LABEL_SYNTHETIC, require_manifest_exists

    try:
        require_manifest_exists(MANIFEST_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    strata_path = DATA_DIR / "strata.csv"
    if not strata_path.exists():
        logger.error(f"No strata file found at {strata_path}. Run compute_strata.py first.")
        sys.exit(1)

    model, crop_size = load_cv_classifier()
    logger.info(f"Loaded CV classifier checkpoint. Face-crop size: {crop_size}")

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    strata = pd.read_csv(strata_path, dtype=str, keep_default_na=False)

    labelled = manifest[manifest["label"].isin([str(LABEL_AUTHENTIC), str(LABEL_SYNTHETIC)])]
    merged = strata.merge(
        labelled[["sample_id", "local_path", "label", "notes"]], on="sample_id", how="inner",
    )
    excluded_mask = merged["notes"].str.contains("EXCLUDED", case=False, na=False)
    eligible = merged[~excluded_mask].copy()

    logger.info(f"{len(eligible)} CV-eligible clips to evaluate zero-shot (excluded {int(excluded_mask.sum())}).")
    if len(eligible) != 32:
        logger.warning(
            f"Expected 32 CV-eligible clips per docs/PROJECT_CONTEXT.md §4, found {len(eligible)}. "
            f"Proceeding anyway — verify the manifest/strata state first if this is unexpected."
        )

    results = []
    for i, row in eligible.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            logger.warning(f"File missing on disk, skipping: {local_path}")
            continue

        label_name = "AUTHENTIC" if int(row["label"]) == LABEL_AUTHENTIC else "SYNTHETIC"
        matched = row["matched_subset"] == "True"
        logger.info(f"[{len(results) + 1}/{len(eligible)}] {row['sample_id'][:8]} "
                    f"({label_name}, matched_subset={matched})...")

        scored = score_clip(model, str(local_path), crop_size)
        pred_int = int(scored["p_synthetic"] >= 0.5) if scored["p_synthetic"] is not None else None
        results.append({
            "sample_id": row["sample_id"][:8], "true_label": label_name, "matched_subset": matched,
            "genre": row.get("genre", ""), "p_synthetic": scored["p_synthetic"],
            "predicted_label_int": pred_int, "n_crops": scored["n_crops"],
            "abstain_reason": scored["abstain_reason"],
        })
        print(f"    p_synthetic={scored['p_synthetic']!r}  n_crops={scored['n_crops']}"
              + (f"  ABSTAINED: {scored['abstain_reason']}" if scored["abstain_reason"] else ""))

    if not results:
        logger.warning("No clips were actually evaluated (all files missing or unopenable?).")
        return

    results_df = pd.DataFrame(results)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    detail_path = output_dir / "cv_transfer_eval.csv"
    results_df.to_csv(detail_path, index=False)

    matched_df = results_df[results_df["matched_subset"]]
    full_df = results_df

    print("\n" + "#" * 90)
    print(f"  PRIMARY RESULT — matched_subset == True (n={len(matched_df)}), controls for the "
          f"platform/orientation confound (compute_strata.py)")
    print("#" * 90)
    matched_metrics = _compute_metrics(matched_df, "matched_subset")
    print(matched_metrics)

    print("\n" + "#" * 90)
    print(f"  SECONDARY — full corpus (n={len(full_df)}), confounded by platform/orientation")
    print("#" * 90)
    full_metrics = _compute_metrics(full_df, "full_corpus")
    print(full_metrics)

    summary_df = pd.DataFrame([matched_metrics, full_metrics])
    summary_path = output_dir / "cv_transfer_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    _print_ff_comparison(matched_metrics)

    print(f"\nSaved per-clip detail to {detail_path}")
    print(f"Saved summary to {summary_path}")
    print(f"\n{len(results_df)}/{len(eligible)} eligible clips evaluated "
          f"({int(results_df['p_synthetic'].isna().sum())} abstained).\n")


if __name__ == "__main__":
    run_evaluation()
