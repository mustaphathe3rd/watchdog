# =============================================================================
# watchdog/dataset_manager.py  —  Dataset Manifest Manager
# =============================================================================
# PURPOSE:
#   Manages data/manifest.csv — the single source of truth mapping every
#   collected sample to its label, category, and metadata. This is the
#   "manifest.csv — Label index" file shown in the Day 1 project structure.
#
# SCHEMA matches Table 3.4 (Summary of Datasets Used for Training and
# Evaluation) exactly — the three dataset categories defined there become
# the allowed `category` values here:
#
#   Table 3.4 Category          | media_type | this project's category key
#   -----------------------------|------------|------------------------------
#   Nigerian Telecom/FMCG Ads    | video      | TELECOM_FMCG_ADS
#   Naija-Political Media        | video      | POLITICAL_MEDIA
#   Local Textual Corpus         | text       | TEXTUAL_CORPUS
#
# LABELLING CONVENTION (§3.9, Table 3.4 columns "Authentic (Human)" /
# "Synthetic (AI)"):
#   label = 0  ->  Authentic / Human
#   label = 1  ->  Synthetic / AI-generated
#   label = None (empty in CSV) -> not yet labelled, awaiting human review
#
# WHY CSV AND NOT DIRECTLY INTO POSTGRESQL?
#   The manifest is a TRAINING dataset index (§3.9), not a compliance record.
#   It never contains a violation, a statute, or a Psynth score — that only
#   exists post-deployment, produced by the trained model (Day 5+). Keeping
#   it as a flat CSV also makes it trivial to upload to Google Drive and
#   mount directly inside a Colab training notebook (see notebooks/02 and
#   notebooks/03), which cannot easily reach a local PostgreSQL instance.
# =============================================================================

import csv
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import MANIFEST_PATH, DATA_DIR

logger = logging.getLogger(__name__)

# ── Allowed category keys (Table 3.4) ─────────────────────────────────────────
CATEGORY_TELECOM_FMCG_ADS = "TELECOM_FMCG_ADS"     # video — MTN, Golden Terra
CATEGORY_POLITICAL_MEDIA  = "POLITICAL_MEDIA"       # video — press releases vs deepfakes
CATEGORY_TEXTUAL_CORPUS   = "TEXTUAL_CORPUS"        # text  — journalism, Pidgin, bureaucratic

VALID_CATEGORIES = {
    CATEGORY_TELECOM_FMCG_ADS,
    CATEGORY_POLITICAL_MEDIA,
    CATEGORY_TEXTUAL_CORPUS,
}

# ── Label convention ──────────────────────────────────────────────────────────
LABEL_AUTHENTIC = 0   # Human-made — matches Table 3.4 "Authentic (Human)" column
LABEL_SYNTHETIC = 1   # AI-generated — matches Table 3.4 "Synthetic (AI)" column

MANIFEST_COLUMNS = [
    "sample_id",       # UUID — stable identifier, independent of file path
    "local_path",      # Relative path under data/raw/{video,text}/
    "source_url",      # Full scraped URL (manifest is a LOCAL training file,
                        #  not the compliance DB — NDPA minimisation (NFR-05)
                        #  applies to the production pipeline's audit log, not
                        #  to a local, non-published research dataset index)
    "media_type",       # 'video' | 'text'
    "category",         # One of VALID_CATEGORIES (Table 3.4)
    "label",            # 0 | 1 | '' (unlabelled)
    "publisher_name",   # e.g. "MTN Nigeria"
    "scraped_at",       # ISO 8601 timestamp
    "notes",            # Free-text — e.g. "verified deepfake, ARCON case ref"
]


@dataclass
class ManifestEntry:
    sample_id:      str
    local_path:     str
    source_url:     str
    media_type:     str
    category:       str
    label:          Optional[int] = None
    publisher_name: str = ""
    scraped_at:     str = ""
    notes:          str = ""

    def __post_init__(self):
        if self.media_type not in ("video", "text"):
            raise ValueError(f"media_type must be 'video' or 'text', got '{self.media_type}'")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {VALID_CATEGORIES}, got '{self.category}'"
            )
        if self.label is not None and self.label not in (0, 1):
            raise ValueError(f"label must be 0, 1, or None — got {self.label}")
        if not self.scraped_at:
            self.scraped_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# MANIFEST INITIALISATION
# =============================================================================

def init_manifest(path: Path = MANIFEST_PATH) -> Path:
    """
    Creates an empty manifest.csv with the correct header row if one doesn't
    already exist. Safe to call repeatedly — never overwrites existing data.
    """
    if path.exists():
        logger.info(f"Manifest already exists at {path} — leaving untouched.")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_COLUMNS)

    logger.info(f"Initialised empty manifest at {path}")
    return path


# =============================================================================
# WRITE OPERATIONS
# =============================================================================

def add_entry(entry: ManifestEntry, path: Path = MANIFEST_PATH) -> None:
    """
    Appends a single ManifestEntry as a new row.

    Deliberately append-only at the file level for simplicity and to avoid
    read-modify-write races if multiple scraper processes run concurrently
    (§3.9 scraping is I/O-bound; a student may reasonably parallelise it).
    Label UPDATES go through label_entry(), which does a full safe rewrite.
    """
    init_manifest(path)   # no-op if it already exists

    row_dict = asdict(entry)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writerow(row_dict)

    logger.debug(f"Manifest entry added: {entry.sample_id} ({entry.category})")


def add_entry_from_scraped_artifact(
    artifact,
    category: str,
    local_path: str,
    path: Path = MANIFEST_PATH,
) -> ManifestEntry:
    """
    Convenience wrapper: builds a ManifestEntry directly from a
    watchdog.scraper.ScrapedArtifact (Day 1) and appends it.

    Args:
        artifact:   A ScrapedArtifact instance (see watchdog/scraper.py).
        category:   One of the VALID_CATEGORIES constants.
        local_path: Where the artifact was saved, relative to data/.
        path:       Manifest file to write to.

    Returns:
        The ManifestEntry that was written.
    """
    entry = ManifestEntry(
        sample_id=str(uuid.uuid4()),
        local_path=local_path,
        source_url=artifact.source_url,
        media_type=artifact.media_type,
        category=category,
        label=None,   # Always starts unlabelled — see label_entry()
        publisher_name=artifact.publisher_name or "",
        scraped_at=artifact.scraped_at.isoformat(),
        notes="",
    )
    add_entry(entry, path=path)
    return entry


def label_entry(sample_id: str, label: int, notes: str = "", path: Path = MANIFEST_PATH) -> bool:
    """
    Sets the label (0=authentic, 1=synthetic) for an existing sample.

    This is a genuine update, unlike the production `violations` table
    (NFR-04), because the manifest is a research dataset index, not a legal
    audit record — mislabels found during review MUST be correctable, or
    the model in Phase 2/3 trains on bad ground truth.

    Implementation: safe full rewrite via pandas (manifest files here are at
    most a few thousand rows — Table 3.4 totals 10,000 — so this is fast).

    Returns:
        True if the sample_id was found and updated, False otherwise.
    """
    if label not in (LABEL_AUTHENTIC, LABEL_SYNTHETIC):
        raise ValueError(f"label must be {LABEL_AUTHENTIC} or {LABEL_SYNTHETIC}, got {label}")

    df = load_manifest(path)
    if df.empty or sample_id not in df["sample_id"].values:
        logger.warning(f"label_entry: sample_id '{sample_id}' not found in manifest")
        return False

    df.loc[df["sample_id"] == sample_id, "label"] = label
    if notes:
        df.loc[df["sample_id"] == sample_id, "notes"] = notes

    df.to_csv(path, index=False)
    logger.info(f"Labelled {sample_id} as {label} ({'synthetic' if label else 'authentic'})")
    return True


# =============================================================================
# READ OPERATIONS
# =============================================================================

def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """
    Loads the manifest as a pandas DataFrame. Returns an empty DataFrame with
    the correct columns if the manifest doesn't exist yet (never raises).

    BUG FIX (discovered via manual verification, Day 2):
    A freshly-initialised manifest has entirely empty `notes` and
    `publisher_name` columns. Left to infer types on its own, pandas reads
    an all-empty column as float64 (NaN), and a later `df.loc[...] = "some
    string"` assignment then raises `TypeError: Invalid value ... for dtype
    'float64'` under pandas 2.x's stricter type-setting rules. Every text
    column is now explicitly typed as string on load to prevent this.
    """
    if not path.exists():
        logger.info(f"No manifest at {path} yet — returning empty DataFrame")
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    string_columns = [c for c in MANIFEST_COLUMNS if c != "label"]
    dtype_map = {col: str for col in string_columns}

    df = pd.read_csv(path, dtype=dtype_map, keep_default_na=False)
    # keep_default_na=False means truly empty cells load as '' (empty string),
    # not NaN — correct for text columns. label needs the opposite treatment
    # (missing = genuinely unlabelled), so it's converted separately below.
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    return df


def require_manifest_exists(path: Path = MANIFEST_PATH) -> None:
    """
    Raises FileNotFoundError with an actionable message if the manifest
    doesn't exist at `path`. For analysis/evaluation entry points ONLY
    (evaluate_nlp_heuristics.py, evaluate_cv_heuristics.py, compute_strata.py,
    etc.) — deliberately NOT called from load_manifest() itself, since a
    missing manifest is the normal, correct state on a first run before
    collect_dataset.py has ever executed (load_manifest() returning an empty
    DataFrame in that case is intentional library behaviour, not a bug).

    Added after a near-miss (2026-08-04, see docs/PROJECT_CONTEXT.md bug
    #11): the real data/ directory was found moved to dashboard/data/, so
    config.MANIFEST_PATH resolved to nothing. Nothing had actually been RUN
    against the empty path yet — it was caught by inspection, not by a
    script failing loudly — which would have been a fourth instance of the
    silent-failure pattern behind bugs #6/#9/#10 if a script had run first.
    Every evaluation script should call this before doing anything else so
    a misplaced/missing manifest is a loud, immediate error instead of a
    quietly empty result set.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No manifest found at {path}. Either data/ has been moved/deleted, "
            f"or collect_dataset.py hasn't been run yet. Refusing to proceed "
            f"against a missing manifest rather than silently evaluating zero samples."
        )


def get_unlabeled(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Returns all rows where label is still empty — the human review queue."""
    df = load_manifest(path)
    if df.empty:
        return df
    return df[df["label"].isna()]


def get_by_category(category: str, path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Returns all rows for one Table 3.4 category (e.g. TEXTUAL_CORPUS)."""
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Valid: {VALID_CATEGORIES}")
    df = load_manifest(path)
    if df.empty:
        return df
    return df[df["category"] == category]


def get_stats(path: Path = MANIFEST_PATH) -> dict:
    """
    Returns a summary matching the shape of Table 3.4, so progress toward
    the target sample counts can be checked at any point during collection:

        Table 3.4 targets:
          Nigerian Telecom/FMCG Ads : 3,000 total (1,500 / 1,500)
          Naija-Political Media     : 1,000 total (500 / 500)
          Local Textual Corpus      : 6,000 total (3,000 / 3,000)
    """
    df = load_manifest(path)
    if df.empty:
        return {"total": 0, "labelled": 0, "unlabelled": 0, "by_category": {}}

    stats = {
        "total":       len(df),
        "labelled":    int(df["label"].notna().sum()),
        "unlabelled":  int(df["label"].isna().sum()),
        "by_category": {},
    }

    for category in VALID_CATEGORIES:
        cat_df = df[df["category"] == category]
        stats["by_category"][category] = {
            "total":     len(cat_df),
            "authentic": int((cat_df["label"] == LABEL_AUTHENTIC).sum()),
            "synthetic": int((cat_df["label"] == LABEL_SYNTHETIC).sum()),
            "unlabelled": int(cat_df["label"].isna().sum()),
        }

    return stats


def print_stats(path: Path = MANIFEST_PATH) -> None:
    """Pretty-prints get_stats() to the terminal — used by collect_dataset.py."""
    stats = get_stats(path)

    print("\n" + "=" * 65)
    print("  DATASET MANIFEST — Current Status (cf. Table 3.4)")
    print("=" * 65)
    print(f"  Total samples:      {stats['total']}")
    print(f"  Labelled:           {stats['labelled']}")
    print(f"  Awaiting label:     {stats['unlabelled']}")
    print("-" * 65)

    target_map = {
        CATEGORY_TELECOM_FMCG_ADS: 3000,
        CATEGORY_POLITICAL_MEDIA:  1000,
        CATEGORY_TEXTUAL_CORPUS:   6000,
    }

    for category, target in target_map.items():
        cat_stats = stats["by_category"].get(category, {
            "total": 0, "authentic": 0, "synthetic": 0, "unlabelled": 0
        })
        pct = (cat_stats["total"] / target * 100) if target else 0
        print(
            f"  {category:22s} {cat_stats['total']:>5}/{target:<5} ({pct:5.1f}%)  "
            f"| authentic={cat_stats['authentic']:<4} synthetic={cat_stats['synthetic']:<4} "
            f"unlabelled={cat_stats['unlabelled']}"
        )
    print("=" * 65 + "\n")
