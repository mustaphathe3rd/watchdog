# =============================================================================
# evaluate_cv_heuristics.py  —  Full-Corpus CV Heuristic Validation
# =============================================================================
# PURPOSE:
#   CV equivalent of evaluate_nlp_heuristics.py. Runs all four Stage 4
#   heuristics (watchdog/cv_engine.py: blink rate, emotional rigidity,
#   viseme-phoneme mismatch, blending artifacts) against every CV-eligible
#   clip in the real, labelled manifest — not the 9-clip calibration sample
#   used to set the thresholds (see docs/PROJECT_CONTEXT.md §4), the full
#   32-clip set.
#
# WHY matched_subset IS THE PRIMARY RESULT, NOT THE FULL CORPUS:
#   The full CV-eligible corpus is confounded by genre and edit pace —
#   authentic samples skew toward longer-form interview/panel footage,
#   synthetic samples skew toward fast-cut ai_avatar_ad content, so a
#   full-corpus score difference could just be measuring "which genre is
#   this" rather than "is this synthetic". data/strata.csv's matched_subset
#   column marks the genre/duration-matched pairs that control for this
#   (docs/PROJECT_CONTEXT.md §4: 4 authentic / 9 synthetic). This script
#   reports matched_subset numbers as the primary result and full-corpus
#   numbers only as confounded secondary context — never the other way
#   round.
#
# WHY BLINK-RATE ABSTENTION IS TREATED AS MISSING DATA, NOT SIGNAL:
#   config.MIN_FACE_RUN_FOR_BLINK_S (10s) structurally excludes every
#   ai_avatar_ad synthetic clip in the corpus — none of them ever holds a
#   contiguous tracked face longer than 6.5s (Stage 4 face-coverage
#   diagnostic, docs/PROJECT_CONTEXT.md §4). A 100% synthetic-class
#   abstention rate on blink rate is therefore a corpus-coverage artefact of
#   fast-cut ad editing, not evidence the heuristic distinguishes classes —
#   reported here as an abstention rate specifically so it CANNOT be misread
#   as a score.
#
# EXPECTATIONS GOING IN (set deliberately, per the calibration run):
#   Three of the four heuristics showed no class separation on 9-clip
#   calibration; the fourth (blink rate) abstains on every synthetic clip.
#   The likely outcome here is that these heuristics discriminate as poorly
#   on 32 clips as the NLP heuristics did on 14 text samples (Day 3). That
#   would be a legitimate, publishable negative result and the argument for
#   the Day 8-9 trained classifier — expected, not a surprise if it happens.
#
# USAGE:
#   python evaluate_cv_heuristics.py
#
# OUTPUT:
#   - Per-clip table printed to the terminal
#   - Score-distribution and abstention-rate tables, matched_subset first
#   - results/cv_heuristic_eval.csv        — per-clip detail, for Chapter 4
#   - results/cv_heuristic_eval_summary.csv — grouped stats, for Chapter 4
#
# NOTE ON RUNTIME:
#   Real MTCNN + MediaPipe Face Mesh inference on 32 real clips, no mocking
#   (this project's "test against real behaviour" convention — see
#   docs/PROJECT_CONTEXT.md §6). Expect several minutes, not seconds.
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

HEURISTICS = [
    ("blink_rate_score", "blink_rate_abstained", "Blink Rate"),
    ("rigidity_score", "rigidity_abstained", "Emotional Rigidity"),
    ("viseme_score", "viseme_abstained", "Viseme-Phoneme Mismatch"),
    ("blending_score", "blending_abstained", "Blending Artifacts"),
]


def _print_distributions(df, title):
    print(f"\n{'=' * 90}\n  {title}\n{'=' * 90}")
    for score_col, _abstain_col, name in HEURISTICS:
        print(f"\n  -- {name} --")
        sub = df[["true_label", score_col]].dropna(subset=[score_col])
        if sub.empty:
            print("     (no non-abstained scores in this subset)")
            continue
        summary = sub.groupby("true_label")[score_col].agg(["count", "mean", "std", "min", "max"])
        print(summary.to_string())


def _abstention_rates(df, subset_name):
    rows = []
    for score_col, abstain_col, name in HEURISTICS:
        for label_name, group in df.groupby("true_label"):
            n = len(group)
            n_abstained = int(group[abstain_col].sum())
            rows.append({
                "subset": subset_name, "heuristic": name, "class": label_name,
                "n": n, "n_abstained": n_abstained,
                "abstention_rate": n_abstained / n if n else float("nan"),
            })
    return rows


def _distribution_rows(df, subset_name):
    rows = []
    for score_col, _abstain_col, name in HEURISTICS:
        for label_name, group in df.groupby("true_label"):
            sub = group[score_col].dropna()
            rows.append({
                "subset": subset_name, "heuristic": name, "class": label_name,
                "n_scored": len(sub),
                "mean_score": sub.mean() if len(sub) else float("nan"),
                "std_score": sub.std() if len(sub) else float("nan"),
                "min_score": sub.min() if len(sub) else float("nan"),
                "max_score": sub.max() if len(sub) else float("nan"),
            })
    return rows


def run_evaluation():
    import pandas as pd
    from config import MANIFEST_PATH, DATA_DIR
    from watchdog.cv_engine import run_heuristics
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

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    strata = pd.read_csv(strata_path, dtype=str, keep_default_na=False)

    labelled = manifest[manifest["label"].isin([str(LABEL_AUTHENTIC), str(LABEL_SYNTHETIC)])]
    merged = strata.merge(
        labelled[["sample_id", "local_path", "label", "notes"]], on="sample_id", how="inner",
    )

    excluded_mask = merged["notes"].str.contains("EXCLUDED", case=False, na=False)
    n_excluded = int(excluded_mask.sum())
    eligible = merged[~excluded_mask].copy()

    logger.info(
        f"{len(merged)} labelled video rows matched to strata; {n_excluded} excluded "
        f"(see manifest notes); {len(eligible)} CV-eligible clips to evaluate."
    )
    if len(eligible) != 32:
        logger.warning(
            f"Expected 32 CV-eligible clips per docs/PROJECT_CONTEXT.md §4, found "
            f"{len(eligible)}. Proceeding anyway — verify the manifest/strata state first "
            f"if this is unexpected."
        )
    print()

    results = []
    for i, row in eligible.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            logger.warning(f"File missing on disk, skipping: {local_path}")
            continue

        label = int(row["label"])
        label_name = "AUTHENTIC" if label == LABEL_AUTHENTIC else "SYNTHETIC"
        matched = row["matched_subset"] == "True"

        logger.info(
            f"[{len(results) + 1}/{len(eligible)}] {row['sample_id'][:8]} "
            f"({label_name}, matched_subset={matched}, genre={row.get('genre', '')})..."
        )
        heur = run_heuristics(local_path)
        if heur is None:
            logger.warning(f"Could not open video, skipping: {local_path}")
            continue

        results.append({
            "sample_id": row["sample_id"][:8],
            "true_label": label_name,
            "matched_subset": matched,
            "genre": row.get("genre", ""),
            "duration_s": row.get("duration_s", ""),
            "blink_rate_score": heur.blink_rate.score,
            "blink_rate_abstained": heur.blink_rate.score is None,
            "rigidity_score": heur.emotional_rigidity.score,
            "rigidity_abstained": heur.emotional_rigidity.score is None,
            "viseme_score": heur.viseme_mismatch.score,
            "viseme_abstained": heur.viseme_mismatch.score is None,
            "blending_score": heur.blending_artifacts.score,
            "blending_abstained": heur.blending_artifacts.score is None,
            "total_time_s": round(heur.total_time_s, 2),
        })
        print(
            f"    blink={heur.blink_rate.score!r:>20}  rigidity={heur.emotional_rigidity.score!r:>20}  "
            f"viseme={heur.viseme_mismatch.score!r:>20}  blending={heur.blending_artifacts.score!r:>20}  "
            f"({heur.total_time_s:.1f}s)"
        )

    if not results:
        logger.warning("No clips were actually evaluated (all files missing or unopenable?).")
        return

    results_df = pd.DataFrame(results)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    detail_path = output_dir / "cv_heuristic_eval.csv"
    results_df.to_csv(detail_path, index=False)

    matched_df = results_df[results_df["matched_subset"]]
    full_df = results_df

    print("\n" + "#" * 90)
    print(f"  PRIMARY RESULT — matched_subset == True (n={len(matched_df)})")
    print("  Full-corpus numbers further below are confounded by genre/edit pace — see")
    print("  the module docstring. Treat matched_subset as the real result.")
    print("#" * 90)
    _print_distributions(matched_df, f"Score Distributions by Label — matched_subset (n={len(matched_df)})")

    print("\n" + "#" * 90)
    print(f"  SECONDARY — full corpus (n={len(full_df)}), confounded by genre/edit pace")
    print("#" * 90)
    _print_distributions(full_df, f"Score Distributions by Label — full corpus (n={len(full_df)})")

    print("\n" + "=" * 90)
    print("  ABSTENTION RATES — matched_subset (primary) then full corpus")
    print("=" * 90)
    abstain_rows = _abstention_rates(matched_df, "matched_subset") + _abstention_rates(full_df, "full_corpus")
    abstain_df = pd.DataFrame(abstain_rows)
    print(abstain_df.to_string(index=False))

    print("\n" + "-" * 90)
    print(
        "  NOTE — Blink Rate abstention on SYNTHETIC is missing data, not signal.\n"
        "  It requires >=10.0s of contiguous face tracking (config.MIN_FACE_RUN_FOR_BLINK_S).\n"
        "  Every ai_avatar_ad synthetic clip in this corpus cuts more often than that (Stage 4\n"
        "  face-coverage diagnostic, docs/PROJECT_CONTEXT.md §4) — a genre/edit-pace coverage\n"
        "  confound, not evidence the heuristic distinguishes classes. Do not read a high\n"
        "  synthetic-class abstention rate here as a finding about synthetic media."
    )
    print("-" * 90)

    summary_rows = (
        _distribution_rows(matched_df, "matched_subset") + _distribution_rows(full_df, "full_corpus")
    )
    summary_df = pd.DataFrame(summary_rows).merge(
        abstain_df[["subset", "heuristic", "class", "n", "n_abstained", "abstention_rate"]],
        on=["subset", "heuristic", "class"], how="left",
    )
    summary_path = output_dir / "cv_heuristic_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved per-clip detail to {detail_path}")
    print(f"Saved grouped summary (distributions + abstention) to {summary_path}")
    print(f"\n{len(results_df)}/{len(eligible)} eligible clips evaluated.\n")


if __name__ == "__main__":
    run_evaluation()
