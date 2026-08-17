# =============================================================================
# collect_dataset.py  —  Day 2 Dataset Collection Orchestrator
# =============================================================================
# Wires together three modules built so far:
#   watchdog/scraper.py          (Day 1) — fetches content from the web
#   watchdog/dataset_manager.py  (Day 2) — records it in manifest.csv
#   seed_sources.py              (Day 2) — tells it WHERE to look
#
# WORKFLOW:
#   1. Populate seed_sources.py's *_SPECIFIC_URLS lists with real URLs
#      (see that file's docstring for guidance).
#   2. Run:  python collect_dataset.py
#   3. Each URL is scraped, saved under data/raw/{video,text}/, and logged
#      as an unlabelled row in data/manifest.csv.
#   4. Run:  python collect_dataset.py --label   to enter the interactive
#      labelling CLI and assign authentic(0)/synthetic(1) to each sample.
#   5. Run:  python collect_dataset.py --stats   any time to see progress
#      against the Table 3.4 targets.
#
# USAGE:
#   python collect_dataset.py            # scrape all queued seed URLs
#   python collect_dataset.py --stats    # show manifest progress only
#   python collect_dataset.py --label    # interactive labelling session
#   python collect_dataset.py --dry-run  # preview without downloading/saving
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_collection(dry_run: bool = False) -> None:
    """Scrapes every URL registered in seed_sources.py and updates the manifest."""
    from config import RAW_VIDEO_DIR, RAW_TEXT_DIR, MANIFEST_PATH
    from watchdog.scraper import scrape_url
    from watchdog.dataset_manager import (
        init_manifest, add_entry_from_scraped_artifact, load_manifest, print_stats
    )
    from seed_sources import get_all_specific_urls, print_seed_summary

    print_seed_summary()
    queued = get_all_specific_urls()

    if not queued:
        logger.warning(
            "No URLs queued in seed_sources.py — nothing to collect.\n"
            "Edit seed_sources.py's *_SPECIFIC_URLS lists first, then re-run."
        )
        return

    init_manifest(MANIFEST_PATH)
    existing = load_manifest(MANIFEST_PATH)
    already_scraped_urls = set(existing["source_url"]) if not existing.empty else set()

    scraped_count, skipped_count, failed_count = 0, 0, 0

    for i, item in enumerate(queued, 1):
        url = item["url"]
        category = item["category"]

        if url in already_scraped_urls:
            logger.info(f"[{i}/{len(queued)}] Already in manifest, skipping: {url[:70]}")
            skipped_count += 1
            continue

        logger.info(f"[{i}/{len(queued)}] Scraping ({category}): {url[:70]}")

        if dry_run:
            logger.info("  [dry-run] would scrape and add to manifest")
            continue

        artifact = scrape_url(url, download_video=True)
        if artifact is None:
            logger.warning(f"  ✗ Scrape failed: {url[:70]}")
            failed_count += 1
            continue

        # Determine save location and relative path for the manifest
        if artifact.is_video() and artifact.local_path:
            local_path = str(artifact.local_path.relative_to(Path.cwd())) \
                if artifact.local_path.is_absolute() else str(artifact.local_path)
        elif artifact.is_text():
            # Text artifacts aren't saved as files by scraper.py — persist here
            safe_name = f"text_{i:04d}.txt"
            save_path = RAW_TEXT_DIR / safe_name
            save_path.write_text(artifact.raw_text, encoding="utf-8")
            local_path = str(save_path.relative_to(Path.cwd()))
        else:
            logger.warning(f"  ✗ No usable content extracted: {url[:70]}")
            failed_count += 1
            continue

        entry = add_entry_from_scraped_artifact(
            artifact, category=category, local_path=local_path
        )
        logger.info(f"  ✓ Added to manifest: {entry.sample_id} ({entry.category})")
        scraped_count += 1

    print("\n" + "=" * 65)
    print("  COLLECTION RUN COMPLETE")
    print("=" * 65)
    print(f"  Newly scraped:  {scraped_count}")
    print(f"  Already had:    {skipped_count}")
    print(f"  Failed:         {failed_count}")
    print("=" * 65)

    if not dry_run:
        print_stats(MANIFEST_PATH)


def run_labelling_session() -> None:
    """
    Interactive terminal CLI for a human to review each unlabelled sample
    and assign authentic(0)/synthetic(1). Ground-truth labelling requires
    human judgement — §3.9's dataset cannot be labelled automatically
    without defeating the purpose of building a supervised training set.
    """
    from config import MANIFEST_PATH
    from watchdog.dataset_manager import get_unlabeled, label_entry, LABEL_AUTHENTIC, LABEL_SYNTHETIC

    unlabeled = get_unlabeled(MANIFEST_PATH)
    if unlabeled.empty:
        print("\nNo unlabelled samples remaining — manifest is fully labelled.\n")
        return

    print(f"\n{len(unlabeled)} unlabelled sample(s) found. For each one:")
    print("  [0] = Authentic/Human   [1] = Synthetic/AI   [s] = skip   [q] = quit\n")

    for _, row in unlabeled.iterrows():
        print("-" * 65)
        print(f"  sample_id : {row['sample_id']}")
        print(f"  category  : {row['category']}")
        print(f"  publisher : {row['publisher_name']}")
        print(f"  source    : {row['source_url']}")
        print(f"  local     : {row['local_path']}")

        choice = input("  Label [0/1/s/q]: ").strip().lower()

        if choice == "q":
            print("Stopping labelling session.")
            break
        elif choice == "s":
            continue
        elif choice in ("0", "1"):
            notes = input("  Notes (optional, Enter to skip): ").strip()
            label_entry(row["sample_id"], int(choice), notes=notes, path=MANIFEST_PATH)
            print("  ✓ Saved.")
        else:
            print("  Not recognised — skipping.")

    from watchdog.dataset_manager import print_stats
    print_stats(MANIFEST_PATH)


def main():
    parser = argparse.ArgumentParser(
        description="Naija AI Watchdog — Dataset Collection (Day 2)"
    )
    parser.add_argument("--stats", action="store_true",
                       help="Show manifest progress against Table 3.4 targets and exit")
    parser.add_argument("--label", action="store_true",
                       help="Start an interactive labelling session")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview what would be scraped without downloading anything")
    args = parser.parse_args()

    if args.stats:
        from config import MANIFEST_PATH
        from watchdog.dataset_manager import print_stats
        print_stats(MANIFEST_PATH)
        return

    if args.label:
        run_labelling_session()
        return

    run_collection(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
