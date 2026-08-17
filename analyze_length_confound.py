# =============================================================================
# analyze_length_confound.py  —  Is the NLP classifier partly a word counter?
# =============================================================================
# PURPOSE:
#   train_nlp_classifier.py's own CAVEATS section flagged, but did not test,
#   the length confound: authentic TEXTUAL_CORPUS samples average 865 words,
#   synthetic average 303 (docs/PROJECT_CONTEXT.md §4). A classifier that
#   partly learned "longer -> authentic" would report a misleadingly good
#   NFR-03 FPR (0.088 in the 5-fold CV) without having learned genuine
#   authorship signal — and that number would not survive contact with a
#   long synthetic article or a short authentic wire brief in production.
#
#   Two direct tests against results/nlp_classifier_cv_eval.csv's real
#   out-of-fold predictions (not a new model, not retraining):
#
#   1. PER-LENGTH-BAND BREAKDOWN. Bin all 65 samples by word count. If the
#      classifier is using length as a shortcut, accuracy should be high in
#      bands where one class dominates (length alone nearly separates them)
#      and collapse in bands where both classes are well represented (length
#      gives no shortcut there, so only genuine signal is left).
#
#   2. LENGTH-MATCHED SUBSET. Restrict to the word-count range where both
#      classes actually overlap substantially, where length CANNOT be used
#      as a shortcut by construction, and recompute precision/recall/F1/FPR
#      on that subset alone. If those numbers are close to the full-corpus
#      aggregate, length wasn't doing much work. If they're notably worse,
#      it was.
#
# WHAT THIS DOES NOT DO:
#   Retrain anything, or change any threshold/config value. Per CLAUDE.md
#   rule 7 and the explicit instruction this script was written under:
#   measure first. If this finds the length signal is doing real work, the
#   fix is a length-matched retraining set — a separate, deliberate next
#   step, not something this script does on its own.
#
# USAGE:
#   python analyze_length_confound.py
#
# OUTPUT:
#   - Tables printed to the terminal
#   - results/nlp_length_band_breakdown.csv
#   - results/nlp_length_matched_subset_eval.csv
# =============================================================================

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LENGTH_BANDS = [(0, 200), (200, 400), (400, 800), (800, float("inf"))]

# "Roughly 200-850 words" per the instruction — verified against the real
# distribution below (printed at runtime) rather than assumed correct.
LENGTH_MATCHED_MIN = 200
LENGTH_MATCHED_MAX = 850


def _band_label(lo, hi):
    return f"{lo}-{int(hi)}" if hi != float("inf") else f"{lo}+"


def _load_word_counts():
    import pandas as pd
    from config import MANIFEST_PATH
    from watchdog.dataset_manager import CATEGORY_TEXTUAL_CORPUS

    df = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    text_rows = df[df["category"] == CATEGORY_TEXTUAL_CORPUS].copy()
    text_rows["sample_id_prefix"] = text_rows["sample_id"].str[:8]

    word_counts = {}
    for _, row in text_rows.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            continue
        wc = len(local_path.read_text(encoding="utf-8", errors="replace").split())
        word_counts[row["sample_id_prefix"]] = wc

    return word_counts


def _metrics(df):
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = [1 if l == "SYNTHETIC" else 0 for l in df["true_label"]]
    y_pred = [1 if l == "SYNTHETIC" else 0 for l in df["predicted_label"]]

    auth = df[df["true_label"] == "AUTHENTIC"]
    n_authentic = len(auth)
    n_fp = int((auth["predicted_label"] == "SYNTHETIC").sum())
    fpr = n_fp / n_authentic if n_authentic else float("nan")

    return {
        "n": len(df),
        "n_authentic": n_authentic,
        "n_synthetic": len(df) - n_authentic,
        "accuracy": (df["true_label"] == df["predicted_label"]).mean() if len(df) else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0) if len(df) else float("nan"),
        "recall": recall_score(y_true, y_pred, zero_division=0) if len(df) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0) if len(df) else float("nan"),
        "fpr_authentic": fpr,
        "n_fp_authentic": n_fp,
        "mean_prob_synthetic_authentic": auth["prob_synthetic"].mean() if n_authentic else float("nan"),
        "mean_prob_synthetic_synthetic": (
            df[df["true_label"] == "SYNTHETIC"]["prob_synthetic"].mean() if len(df) - n_authentic else float("nan")
        ),
    }


def run_analysis():
    import pandas as pd
    from config import MANIFEST_PATH

    cv_eval_path = Path("results") / "nlp_classifier_cv_eval.csv"
    if not cv_eval_path.exists():
        logger.error(f"No {cv_eval_path} found — run train_nlp_classifier.py first.")
        sys.exit(1)

    cv_df = pd.read_csv(cv_eval_path, dtype={"sample_id": str})
    word_counts = _load_word_counts()

    cv_df["word_count"] = cv_df["sample_id"].map(word_counts)
    missing = cv_df["word_count"].isna().sum()
    if missing:
        logger.error(
            f"{missing} sample(s) in {cv_eval_path} couldn't be matched to a word count via "
            f"{MANIFEST_PATH} — aborting rather than silently dropping/mismatching rows."
        )
        sys.exit(1)
    cv_df["word_count"] = cv_df["word_count"].astype(int)

    print(f"Loaded {len(cv_df)} out-of-fold predictions with word counts.")
    print(cv_df.groupby("true_label")["word_count"].describe().to_string())
    print()

    # =========================================================================
    # 1. Per-length-band breakdown
    # =========================================================================
    print("=" * 100)
    print("  1. PER-LENGTH-BAND BREAKDOWN")
    print("  If the classifier is using length as a shortcut: accuracy stays high where one class")
    print("  dominates a band, and collapses where both classes are well represented in the same band.")
    print("=" * 100)

    band_rows = []
    for lo, hi in LENGTH_BANDS:
        band_df = cv_df[(cv_df["word_count"] >= lo) & (cv_df["word_count"] < hi)]
        m = _metrics(band_df)
        m["band"] = _band_label(lo, hi)
        band_rows.append(m)

    band_df_out = pd.DataFrame(band_rows)[
        ["band", "n", "n_authentic", "n_synthetic", "accuracy", "fpr_authentic",
         "mean_prob_synthetic_authentic", "mean_prob_synthetic_synthetic"]
    ]
    print(band_df_out.to_string(index=False))
    print()
    for _, r in band_df_out.iterrows():
        both_present = r["n_authentic"] >= 2 and r["n_synthetic"] >= 2
        flavour = "BOTH classes well represented" if both_present else "one class dominates or band is sparse"
        acc_str = f"{r['accuracy']:.3f}" if pd.notna(r["accuracy"]) else "n/a"
        print(f"  [{r['band']:>8} words]  n={r['n']:>2} (auth={r['n_authentic']}, synth={r['n_synthetic']})  "
              f"accuracy={acc_str}  <- {flavour}")

    band_path = Path("results") / "nlp_length_band_breakdown.csv"
    band_df_out.to_csv(band_path, index=False)

    # =========================================================================
    # 2. Length-matched subset
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"  2. LENGTH-MATCHED SUBSET ({LENGTH_MATCHED_MIN}-{LENGTH_MATCHED_MAX} words)")
    print("  Restricted to the range where both classes actually overlap — length cannot be a")
    print("  shortcut here by construction. Compare these metrics to the full-corpus aggregate")
    print("  (precision=0.897, recall=0.839, f1=0.867, FPR=0.088) from train_nlp_classifier.py.")
    print("=" * 100)

    matched_df = cv_df[
        (cv_df["word_count"] >= LENGTH_MATCHED_MIN) & (cv_df["word_count"] <= LENGTH_MATCHED_MAX)
    ]
    matched_metrics = _metrics(matched_df)
    matched_metrics["band"] = f"{LENGTH_MATCHED_MIN}-{LENGTH_MATCHED_MAX} (length-matched)"

    print(f"\n  n={matched_metrics['n']} (authentic={matched_metrics['n_authentic']}, "
          f"synthetic={matched_metrics['n_synthetic']})")
    if matched_metrics["n"] < 20:
        print(f"  *** n={matched_metrics['n']} is under ~20 — TREAT AS INDICATIVE ONLY, not a reliable estimate. ***")
    print(f"  accuracy={matched_metrics['accuracy']:.3f}  precision={matched_metrics['precision']:.3f}  "
          f"recall={matched_metrics['recall']:.3f}  f1={matched_metrics['f1']:.3f}")
    print(f"  FPR on authentic: {matched_metrics['fpr_authentic']:.3f} "
          f"({matched_metrics['n_fp_authentic']}/{matched_metrics['n_authentic']})")
    print(f"  mean P(synthetic) | true=AUTHENTIC: {matched_metrics['mean_prob_synthetic_authentic']:.3f}")
    print(f"  mean P(synthetic) | true=SYNTHETIC: {matched_metrics['mean_prob_synthetic_synthetic']:.3f}")

    delta_fpr = matched_metrics["fpr_authentic"] - 0.088
    delta_f1 = matched_metrics["f1"] - 0.867
    print(f"\n  vs full-corpus aggregate: FPR delta = {delta_fpr:+.3f}, F1 delta = {delta_f1:+.3f}")

    matched_out = pd.DataFrame([matched_metrics])
    matched_path = Path("results") / "nlp_length_matched_subset_eval.csv"
    matched_out.to_csv(matched_path, index=False)

    print(f"\nSaved per-band breakdown to {band_path}")
    print(f"Saved length-matched subset evaluation to {matched_path}")

    # =========================================================================
    # 3. Length-only baseline ceiling
    # =========================================================================
    # The most direct version of the question: what's the BEST accuracy any
    # rule using word_count ALONE could achieve on this corpus? Best-threshold
    # search over "predict synthetic if word_count <= t", evaluated by
    # RESUBSTITUTION (fit and scored on the same 65 samples — an optimistic,
    # in-sample upper bound, NOT a fair held-out estimate the way the
    # classifier's out-of-fold numbers are). If the real classifier's honest
    # out-of-fold accuracy doesn't clear even this generous ceiling, length
    # alone could explain its performance. If it clears the ceiling, the
    # classifier is doing more than word-counting could achieve even under
    # best-case conditions.
    print("\n" + "=" * 100)
    print("  3. LENGTH-ONLY BASELINE CEILING")
    print("  Best possible accuracy from word_count ALONE (optimistic in-sample threshold search)")
    print("  vs. the real classifier's honest out-of-fold accuracy on the same 65 samples.")
    print("=" * 100)

    y_true = (cv_df["true_label"] == "SYNTHETIC").astype(int)
    best_acc, best_t, best_fpr = 0.0, None, float("nan")
    for t in sorted(cv_df["word_count"].unique()):
        pred = (cv_df["word_count"] <= t).astype(int)
        acc = (pred == y_true).mean()
        if acc > best_acc:
            auth_mask = y_true == 0
            fp = int((pred[auth_mask] == 1).sum())
            best_acc, best_t, best_fpr = acc, int(t), fp / auth_mask.sum()

    real_accuracy = (cv_df["true_label"] == cv_df["predicted_label"]).mean()
    real_fpr = float(
        (cv_df[cv_df["true_label"] == "AUTHENTIC"]["predicted_label"] == "SYNTHETIC").mean()
    )

    print(f"\n  Length-only best rule: predict SYNTHETIC if word_count <= {best_t}")
    print(f"    resubstitution accuracy (optimistic, in-sample): {best_acc:.3f}")
    print(f"    resubstitution FPR on authentic:                 {best_fpr:.3f}")
    print(f"\n  Real classifier (honest, out-of-fold, same 65 samples):")
    print(f"    accuracy: {real_accuracy:.3f}")
    print(f"    FPR on authentic: {real_fpr:.3f}")
    verdict = (
        "beats the optimistic length-only ceiling despite using a fair held-out estimate — "
        "length alone cannot explain this performance"
        if real_accuracy > best_acc else
        "does NOT clear the optimistic length-only ceiling — length could be doing significant work"
    )
    print(f"\n  VERDICT: real out-of-fold accuracy ({real_accuracy:.3f}) {verdict}.")

    ceiling_df = pd.DataFrame([{
        "best_length_threshold": best_t,
        "length_only_resubstitution_accuracy": best_acc,
        "length_only_resubstitution_fpr": best_fpr,
        "classifier_out_of_fold_accuracy": real_accuracy,
        "classifier_out_of_fold_fpr": real_fpr,
    }])
    ceiling_path = Path("results") / "nlp_length_only_baseline_ceiling.csv"
    ceiling_df.to_csv(ceiling_path, index=False)
    print(f"\nSaved length-only baseline ceiling to {ceiling_path}\n")


if __name__ == "__main__":
    run_analysis()
