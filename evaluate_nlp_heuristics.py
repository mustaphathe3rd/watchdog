# =============================================================================
# evaluate_nlp_heuristics.py  —  Day 3 Real-Data Validation
# =============================================================================
# PURPOSE:
#   Runs the perplexity + burstiness heuristics from watchdog/nlp_engine.py
#   against every LABELLED TEXTUAL_CORPUS sample in your actual
#   data/manifest.csv — not illustrative examples, your real, verified,
#   student-labelled Nigerian text data (14 samples: 4 authentic, 10
#   synthetic, as of this writing).
#
#   This is the first genuine evidence of whether perplexity/burstiness
#   have any discriminative power on real Nigerian-context text BEFORE any
#   expensive DistilBERT fine-tuning happens (Day 8-9). Small N (14) means
#   this is a sanity check, not a publishable result — but it's real,
#   traceable signal from your own dataset, which is exactly what §3.14's
#   evaluation strategy calls for at every stage.
#
# USAGE:
#   python evaluate_nlp_heuristics.py
#
# OUTPUT:
#   - A per-sample table printed to the terminal
#   - A summary comparing mean perplexity/burstiness by class (label=0 vs 1)
#   - results/nlp_heuristic_eval.csv — saved for use in your Chapter 4
#     results tables/figures
#
# NOTE ON RUNTIME:
#   The first run downloads DistilGPT-2 (~350MB) from Hugging Face — this
#   needs your machine's internet access (this step can't run inside a
#   restricted sandbox, which is why this script wasn't pre-run for you).
#   Subsequent runs use the cached model and are fast.
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


def run_evaluation():
    import pandas as pd
    from config import MANIFEST_PATH
    from watchdog.nlp_engine import analyze_text
    from watchdog.dataset_manager import (
        CATEGORY_TEXTUAL_CORPUS, LABEL_AUTHENTIC, LABEL_SYNTHETIC, require_manifest_exists,
    )

    try:
        require_manifest_exists(MANIFEST_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    text_rows = df[
        (df["category"] == CATEGORY_TEXTUAL_CORPUS) &
        (df["label"].isin([str(LABEL_AUTHENTIC), str(LABEL_SYNTHETIC)]))
    ].copy()

    if text_rows.empty:
        logger.warning("No labelled TEXTUAL_CORPUS samples found in the manifest.")
        return

    logger.info(f"Evaluating {len(text_rows)} labelled text samples...")
    print()

    results = []
    for i, row in text_rows.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            logger.warning(f"File missing on disk, skipping: {local_path}")
            continue

        text = local_path.read_text(encoding="utf-8", errors="replace")
        analysis = analyze_text(text)

        label = int(row["label"])
        label_name = "AUTHENTIC" if label == LABEL_AUTHENTIC else "SYNTHETIC"

        results.append({
            "sample_id": row["sample_id"][:8],
            "true_label": label_name,
            "perplexity": analysis["perplexity"],
            "burstiness": analysis["burstiness"],
            "heuristic_vote": analysis["heuristic_vote"],
            "word_count": analysis["word_count"],
            "publisher": row.get("publisher_name", ""),
        })

        correct = (
            (analysis["heuristic_vote"] == "AI" and label == LABEL_SYNTHETIC) or
            (analysis["heuristic_vote"] == "HUMAN" and label == LABEL_AUTHENTIC)
        )
        marker = "✓" if correct else ("~" if analysis["heuristic_vote"] == "UNCERTAIN" else "✗")
        print(
            f"  {marker} {label_name:10s} | perplexity={analysis['perplexity']:>8.2f} | "
            f"burstiness={analysis['burstiness']:>+.3f} | vote={analysis['heuristic_vote']:9s} | "
            f"{row.get('publisher_name', '')[:30]}"
        )

    if not results:
        logger.warning("No samples were actually evaluated (all files missing?).")
        return

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("  SUMMARY — Mean Heuristic Scores by True Label")
    print("=" * 75)
    summary = results_df.groupby("true_label")[["perplexity", "burstiness"]].agg(["mean", "std"])
    print(summary.to_string())

    # Simple accuracy of the heuristic vote alone (NOT the final system —
    # this is pre-fusion, pre-fine-tuning signal per §3.14's staged
    # evaluation approach)
    votable = results_df[results_df["heuristic_vote"] != "UNCERTAIN"]
    if not votable.empty:
        correct = (
            ((votable["heuristic_vote"] == "AI") & (votable["true_label"] == "SYNTHETIC")) |
            ((votable["heuristic_vote"] == "HUMAN") & (votable["true_label"] == "AUTHENTIC"))
        ).sum()
        print(f"\n  Heuristic-only accuracy (excluding UNCERTAIN): {correct}/{len(votable)} "
              f"({100 * correct / len(votable):.1f}%)")
        print(f"  UNCERTAIN votes: {len(results_df) - len(votable)}/{len(results_df)}")
    print("=" * 75)

    # Save for Chapter 4
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "nlp_heuristic_eval.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to {output_path} for use in Chapter 4 tables/figures.\n")


if __name__ == "__main__":
    run_evaluation()
