# =============================================================================
# compute_strata.py  —  Per-Sample Stratification for CV-Engine Bias Checking
# =============================================================================
# PURPOSE:
#   Before evaluating the CV engine (facial-artifact detector, FR-02/FR-03),
#   we need to know whether it's actually keying on synthesis artifacts, or
#   on confounds that happen to correlate with label in this small corpus —
#   e.g. every synthetic sample happens to be a 2026 Instagram reel, so a
#   detector that just learns "portrait 2026 Instagram video" would score
#   well without detecting anything about synthesis at all.
#
#   This script derives six stratum fields per VIDEO sample:
#     era             — upload year bucket (pre_2020 / 2020_2023 / 2024_2025 / 2026)
#     platform        — Instagram / X / YouTube, from source_url domain
#     duration_bucket — <30s / 30-60s / 1-3min / 3min+
#     orientation     — landscape / portrait / square
#     genre           — content-type classification (see GENRE_MAP below)
#     matched_subset  — True for the platform/orientation-matched evaluation
#                        stratum (see MATCHED-SUBSET FLAG below)
#
#   Output: data/strata.csv, one row per video sample_id, joinable against
#   data/manifest.csv. A future eval script should group/stratify on these
#   fields (e.g. "does accuracy hold within each era bucket, not just
#   overall?") rather than just reporting a single aggregate accuracy number.
#
# MATCHED-SUBSET FLAG (added after the Instagram authentic-sample expansion):
#   The full video corpus remains confounded on platform/orientation — most
#   authentic samples are landscape YouTube, most synthetic samples are
#   portrait Instagram (see PROJECT_CONTEXT.md for the full history of that
#   confound and the expansion that partially closed it). `matched_subset`
#   marks the one cell where both labels now have real representation:
#   Instagram + portrait (4 authentic, 9 synthetic as of this writing). The
#   CV-engine evaluation should report full-corpus and matched-subset
#   metrics SEPARATELY — a detector that only works outside the matched
#   subset is likely keying on platform/orientation, not synthesis artifacts.
#
# WHY NOT A MANIFEST COLUMN?
#   data/manifest.csv has a strict, validated schema (see
#   watchdog/dataset_manager.py — MANIFEST_COLUMNS, ManifestEntry.__post_init__).
#   Every field there today is either ground truth (label) or scrape
#   provenance (source_url, scraped_at) — written once by the collection
#   pipeline. Stratum fields are diagnostic metadata for bias-checking, not
#   ground truth, and are 100% re-derivable from the manifest + the video
#   files themselves. Keeping them in a separate, regenerable file avoids a
#   schema migration to dataset_manager.py for what is purely an
#   evaluation-time concern, and this script can just be re-run whenever new
#   samples are added rather than touching the manifest write path.
#
# HOW ERA AND GENRE WERE DERIVED (NOT MECHANICALLY DERIVABLE):
#   `orientation`, `duration_bucket`, and `platform` are computed directly
#   from ffprobe output and the source_url — no judgement calls.
#   `era` requires each video's real upload date, which isn't in the
#   manifest. These were fetched individually via `yt-dlp --print
#   %(upload_date)s` against each source_url (2026-08-01) and are hardcoded
#   in UPLOAD_DATE below — re-run that lookup if new samples are added.
#   `genre` is a content-type judgement call made during manual review of
#   each video (thumbnails / full downloads) earlier in this project's
#   history — hardcoded in GENRE_MAP below with the reasoning inline.
#   Both tables are keyed on the first 8 chars of sample_id (matches the
#   short IDs used throughout this project's notes) and will raise loudly
#   for any video sample not covered, rather than silently guessing.
# =============================================================================

import csv
import json
import re
import subprocess
import sys

from config import MANIFEST_PATH, DATA_DIR
from watchdog.dataset_manager import load_manifest

STRATA_PATH = DATA_DIR / "strata.csv"

# ── era: upload_date (YYYYMMDD), fetched via yt-dlp against each source_url ──
UPLOAD_DATE = {
    "b1ce1815": "20260612", "6d95436c": "20260611", "3a0af80c": "20260621",
    "50420433": "20260620", "c4bc80a6": "20260422", "84f30c5c": "20260408",
    "f9bf1f34": "20251003", "2dc8a1f6": "20260101", "ec69163f": "20260415",
    "158dbbda": "20260408", "d39b79fc": "20260313", "f35abdb5": "20250530",
    "b7c35b27": "20260707", "43392d3d": "20251024", "88e00c7c": "20130422",
    "d5e94745": "20160516", "00bbba4d": "20151004", "3572306b": "20151001",
    "f2fc19dd": "20241105", "a8e45112": "20101006", "f030bd5b": "20101006",
    "a7716d7a": "20101006", "835b5341": "20101006", "fc4b1a55": "20260117",
    "33bc415e": "20260731", "83a9ffbb": "20260731", "b8cbac67": "20260731",
    "5f52f83f": "20260326", "1aade253": "20251114",
    "febde9f3": "20241105", "f02c7b8f": "20241019",
    "e2af117c": "20250612", "ed3c9aad": "20250203",
}


def era_bucket(upload_date: str) -> str:
    year = int(upload_date[:4])
    if year < 2020:
        return "pre_2020"
    if year <= 2023:
        return "2020_2023"
    if year <= 2025:
        return "2024_2025"
    return "2026"


# ── genre: content-type classification from manual review ───────────────────
# ai_avatar_ad            — synthetic AI-avatar-presenter ad (the confound
#                            this whole corpus expansion exists to isolate)
# celebrity_endorsement_ad— live-action ad built around a named human
#                            celebrity as the on-camera talent
# brand_tvc               — traditional scripted narrative TV commercial,
#                            no single celebrity face as the selling point
# corporate_interview     — on-camera talking-head interview, corporate context
# studio_panel_interview  — TV studio interview/panel show, anchor + guest(s)
# studio_interview        — single-subject direct-to-camera interview,
#                            non-studio setting (e.g. office)
# news_package            — news report/ceremony coverage; camera cuts
#                            between speaker and event B-roll
# event_coverage           — media/lifestyle outlet's candid coverage of a
#                            real branded activation/event featuring a
#                            confirmed brand ambassador or named guest — not
#                            a produced ad, but genuine event footage
GENRE_MAP = {
    # 7 MTN AI-avatar Instagram reels
    "b1ce1815": "ai_avatar_ad", "6d95436c": "ai_avatar_ad",
    "3a0af80c": "ai_avatar_ad", "50420433": "ai_avatar_ad",
    "c4bc80a6": "ai_avatar_ad", "84f30c5c": "ai_avatar_ad",
    "f9bf1f34": "ai_avatar_ad",
    "2dc8a1f6": "ai_avatar_ad",   # TasteMaker
    "ec69163f": "ai_avatar_ad", "158dbbda": "ai_avatar_ad",
    "d39b79fc": "ai_avatar_ad",   # 3x bet9ja
    "f35abdb5": "ai_avatar_ad",   # Golden Terra — producer-confirmed AI ad
    "b7c35b27": "ai_avatar_ad",   # iLOTBET
    "43392d3d": "ai_avatar_ad",   # Supramult

    "88e00c7c": "celebrity_endorsement_ad",  # MTN ft. SAKA
    "d5e94745": "celebrity_endorsement_ad",  # MTN ft. Falz
    "00bbba4d": "celebrity_endorsement_ad",  # Glo ft. Korede Bello
    "3572306b": "celebrity_endorsement_ad",  # Glo ft. Di'Ja
    "f2fc19dd": "celebrity_endorsement_ad",  # Airtel ft. Farooq Oreagba

    "a8e45112": "brand_tvc", "f030bd5b": "brand_tvc",
    "a7716d7a": "brand_tvc", "835b5341": "brand_tvc",  # 4x Indomie

    "fc4b1a55": "corporate_interview",     # Dangote / Fatima Aliko-Dangote
    "33bc415e": "news_package",            # Channels — healthcare ceremony
    "83a9ffbb": "studio_panel_interview",  # Channels — Politics Today (trim)
    "b8cbac67": "studio_panel_interview",  # Channels — Hard Copy (trim)
    "5f52f83f": "studio_interview",        # NTA — office interview (trim)
    "1aade253": "news_package",            # Arise News — MTN exec at launch

    "febde9f3": "news_package",     # MediaRoomHub — Airtel/Oreagba launch event
    "f02c7b8f": "event_coverage",   # BellaNaija — Star Radler ft. Erica Nlewedim
    "e2af117c": "brand_tvc",        # Amstel Malta — Democracy Day narrative ad
    "ed3c9aad": "brand_tvc",        # Amstel Malta — Valentine's Day narrative ad
}

# ── campaign_family: samples known to share production/campaign lineage ─────
# NOT a duplicate-footage flag (frame comparison on 2026-08-01 confirmed the
# 4 Indomie clips use 4 different actors/sets/scenes despite sharing a
# tagline and release date — see notes on that sample_id below). This is a
# softer signal: these 4 share campaign-level production style/era/genre and
# may still be more correlated with each other than with the rest of the
# corpus, worth knowing about even though they don't need dedup/grouping for
# train/test splitting.
CAMPAIGN_FAMILY = {
    "a8e45112": "indomie_mum_like_no_other_2010",
    "f030bd5b": "indomie_mum_like_no_other_2010",
    "a7716d7a": "indomie_mum_like_no_other_2010",
    "835b5341": "indomie_mum_like_no_other_2010",
}


def platform_of(source_url: str) -> str:
    if "instagram.com" in source_url:
        return "Instagram"
    if "x.com" in source_url or "twitter.com" in source_url:
        return "X"
    if "youtube.com" in source_url or "youtu.be" in source_url:
        return "YouTube"
    return "unknown"


def duration_bucket(seconds: float) -> str:
    if seconds < 30:
        return "under_30s"
    if seconds < 60:
        return "30s_60s"
    if seconds < 180:
        return "1min_3min"
    return "3min_plus"


def orientation_of(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def ffprobe_wh_duration(path: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    data = json.loads(out.stdout)
    fmt = data.get("format", {})
    vstream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
    if vstream is None:
        raise ValueError(f"No video stream found in {path}")
    return int(vstream["width"]), int(vstream["height"]), float(fmt.get("duration", 0))


def compute_strata() -> None:
    df = load_manifest(MANIFEST_PATH)
    vids = df[df["media_type"] == "video"]

    rows = []
    missing_era, missing_genre = [], []
    matched_labels = []   # (sample_id, label) for matched-subset summary only — label
                           # itself is NOT persisted to strata.csv (it's mutable ground
                           # truth that lives in manifest.csv; duplicating it here risks
                           # the two files silently drifting apart after a relabel)

    for _, r in vids.iterrows():
        sample_id = r["sample_id"]
        short_id = sample_id[:8]

        width, height, duration = ffprobe_wh_duration(r["local_path"])

        upload_date = UPLOAD_DATE.get(short_id)
        if upload_date is None:
            missing_era.append(sample_id)
            era = "unknown"
        else:
            era = era_bucket(upload_date)

        genre = GENRE_MAP.get(short_id)
        if genre is None:
            missing_genre.append(sample_id)
            genre = "unknown"

        platform = platform_of(r["source_url"])
        orientation = orientation_of(width, height)
        matched_subset = (platform == "Instagram" and orientation == "portrait")

        rows.append({
            "sample_id": sample_id,
            "era": era,
            "upload_date": upload_date or "",
            "platform": platform,
            "duration_bucket": duration_bucket(duration),
            "duration_s": round(duration, 1),
            "orientation": orientation,
            "resolution": f"{width}x{height}",
            "genre": genre,
            "campaign_family": CAMPAIGN_FAMILY.get(short_id, ""),
            "matched_subset": matched_subset,
        })
        if matched_subset:
            matched_labels.append(r["label"])

    if missing_era:
        print(f"WARNING: no verified upload_date for {len(missing_era)} sample(s) "
              f"— era set to 'unknown', NOT guessed: {missing_era}", file=sys.stderr)
    if missing_genre:
        print(f"WARNING: no genre classification for {len(missing_genre)} sample(s) "
              f"— genre set to 'unknown', NOT guessed: {missing_genre}", file=sys.stderr)

    fieldnames = ["sample_id", "era", "upload_date", "platform", "duration_bucket",
                  "duration_s", "orientation", "resolution", "genre", "campaign_family",
                  "matched_subset"]
    with open(STRATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {STRATA_PATH}")

    matched_auth = sum(1 for lbl in matched_labels if lbl == 0)
    matched_synth = sum(1 for lbl in matched_labels if lbl == 1)
    print(f"Matched subset (Instagram + portrait): {len(matched_labels)} samples "
          f"({matched_auth} authentic, {matched_synth} synthetic) — "
          f"join against manifest.csv for label, this file only carries the flag")


if __name__ == "__main__":
    compute_strata()
