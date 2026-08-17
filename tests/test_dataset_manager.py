# =============================================================================
# tests/test_dataset_manager.py  —  Dataset Manifest Test Suite
# =============================================================================
# Covers §3.9 (Dataset Description) and Table 3.4's schema/target structure.
#
# RUN:
#   pytest tests/test_dataset_manager.py -v
# =============================================================================

import pytest

from watchdog.dataset_manager import (
    ManifestEntry, init_manifest, add_entry, label_entry,
    load_manifest, get_unlabeled, get_by_category, get_stats,
    CATEGORY_TEXTUAL_CORPUS, CATEGORY_TELECOM_FMCG_ADS, CATEGORY_POLITICAL_MEDIA,
    LABEL_AUTHENTIC, LABEL_SYNTHETIC,
)


@pytest.fixture()
def manifest_path(tmp_path):
    """Each test gets its own throwaway manifest.csv — no shared state."""
    return tmp_path / "manifest.csv"


@pytest.fixture()
def populated_manifest(manifest_path):
    """Three entries: one per Table 3.4 category, one already labelled."""
    entries = [
        ManifestEntry(
            sample_id="text-1", local_path="data/raw/text/a.txt",
            source_url="https://example-news.ng/a", media_type="text",
            category=CATEGORY_TEXTUAL_CORPUS,
        ),
        ManifestEntry(
            sample_id="video-1", local_path="data/raw/video/b.mp4",
            source_url="https://example.com/mtn-ad.mp4", media_type="video",
            category=CATEGORY_TELECOM_FMCG_ADS,
        ),
        ManifestEntry(
            sample_id="video-2", local_path="data/raw/video/c.mp4",
            source_url="https://example.com/press-release.mp4", media_type="video",
            category=CATEGORY_POLITICAL_MEDIA, label=LABEL_SYNTHETIC,
        ),
    ]
    for e in entries:
        add_entry(e, path=manifest_path)
    return manifest_path


# =============================================================================
# ManifestEntry validation
# =============================================================================

class TestManifestEntryValidation:

    def test_rejects_invalid_media_type(self):
        with pytest.raises(ValueError, match="media_type"):
            ManifestEntry(
                sample_id="x", local_path="p", source_url="u",
                media_type="audio", category=CATEGORY_TEXTUAL_CORPUS,
            )

    def test_rejects_invalid_category(self):
        with pytest.raises(ValueError, match="category"):
            ManifestEntry(
                sample_id="x", local_path="p", source_url="u",
                media_type="text", category="NOT_A_REAL_CATEGORY",
            )

    def test_rejects_invalid_label(self):
        with pytest.raises(ValueError, match="label"):
            ManifestEntry(
                sample_id="x", local_path="p", source_url="u",
                media_type="text", category=CATEGORY_TEXTUAL_CORPUS, label=99,
            )

    def test_auto_fills_scraped_at_when_omitted(self):
        entry = ManifestEntry(
            sample_id="x", local_path="p", source_url="u",
            media_type="text", category=CATEGORY_TEXTUAL_CORPUS,
        )
        assert entry.scraped_at != ""

    def test_accepts_valid_entry(self):
        entry = ManifestEntry(
            sample_id="x", local_path="p", source_url="u",
            media_type="video", category=CATEGORY_POLITICAL_MEDIA, label=LABEL_AUTHENTIC,
        )
        assert entry.label == 0


# =============================================================================
# init_manifest / add_entry / load_manifest round-trip
# =============================================================================

class TestManifestIO:

    def test_init_creates_file_with_header(self, manifest_path):
        init_manifest(manifest_path)
        assert manifest_path.exists()
        df = load_manifest(manifest_path)
        assert list(df.columns) == [
            "sample_id", "local_path", "source_url", "media_type",
            "category", "label", "publisher_name", "scraped_at", "notes",
        ]
        assert len(df) == 0

    def test_init_is_idempotent(self, manifest_path):
        init_manifest(manifest_path)
        add_entry(
            ManifestEntry(sample_id="s1", local_path="p", source_url="u",
                         media_type="text", category=CATEGORY_TEXTUAL_CORPUS),
            path=manifest_path,
        )
        init_manifest(manifest_path)   # must NOT wipe existing data
        df = load_manifest(manifest_path)
        assert len(df) == 1

    def test_load_manifest_missing_file_returns_empty_df(self, tmp_path):
        never_created = tmp_path / "nonexistent.csv"
        df = load_manifest(never_created)
        assert len(df) == 0
        assert "sample_id" in df.columns

    def test_add_entry_round_trips_all_fields(self, manifest_path):
        entry = ManifestEntry(
            sample_id="round-trip-1", local_path="data/raw/text/z.txt",
            source_url="https://example.com/z", media_type="text",
            category=CATEGORY_TEXTUAL_CORPUS, label=LABEL_AUTHENTIC,
            publisher_name="Test Times", notes="a note with, a comma",
        )
        add_entry(entry, path=manifest_path)

        df = load_manifest(manifest_path)
        row = df[df["sample_id"] == "round-trip-1"].iloc[0]
        assert row["local_path"] == entry.local_path
        assert row["source_url"] == entry.source_url
        assert row["publisher_name"] == "Test Times"
        assert row["notes"] == "a note with, a comma"   # CSV comma-escaping works
        assert int(row["label"]) == LABEL_AUTHENTIC

    def test_notes_column_accepts_string_after_load_even_when_all_empty(
        self, manifest_path
    ):
        """
        Regression test for the pandas float64-inference bug found on Day 2:
        a manifest where every `notes` value is empty must still allow a
        later string write via label_entry() without raising TypeError.
        """
        add_entry(
            ManifestEntry(sample_id="s1", local_path="p", source_url="u",
                         media_type="text", category=CATEGORY_TEXTUAL_CORPUS),
            path=manifest_path,
        )
        # This must not raise TypeError('Invalid value ... for dtype float64')
        result = label_entry("s1", LABEL_AUTHENTIC, notes="now has a note", path=manifest_path)
        assert result is True


# =============================================================================
# label_entry
# =============================================================================

class TestLabelEntry:

    def test_label_updates_existing_row(self, populated_manifest):
        result = label_entry("text-1", LABEL_SYNTHETIC, path=populated_manifest)
        assert result is True

        df = load_manifest(populated_manifest)
        row = df[df["sample_id"] == "text-1"].iloc[0]
        assert int(row["label"]) == LABEL_SYNTHETIC

    def test_label_returns_false_for_unknown_id(self, populated_manifest):
        result = label_entry("does-not-exist", LABEL_AUTHENTIC, path=populated_manifest)
        assert result is False

    def test_label_rejects_out_of_range_value(self, populated_manifest):
        with pytest.raises(ValueError):
            label_entry("text-1", 2, path=populated_manifest)

    def test_relabelling_overwrites_previous_label(self, populated_manifest):
        label_entry("video-2", LABEL_AUTHENTIC, path=populated_manifest)  # was SYNTHETIC
        df = load_manifest(populated_manifest)
        row = df[df["sample_id"] == "video-2"].iloc[0]
        assert int(row["label"]) == LABEL_AUTHENTIC


# =============================================================================
# Query helpers
# =============================================================================

class TestQueryHelpers:

    def test_get_unlabeled_excludes_labelled_rows(self, populated_manifest):
        unlabeled = get_unlabeled(populated_manifest)
        ids = set(unlabeled["sample_id"])
        assert "video-2" not in ids     # this one was pre-labelled
        assert "text-1" in ids
        assert "video-1" in ids

    def test_get_by_category_filters_correctly(self, populated_manifest):
        video_ads = get_by_category(CATEGORY_TELECOM_FMCG_ADS, populated_manifest)
        assert len(video_ads) == 1
        assert video_ads.iloc[0]["sample_id"] == "video-1"

    def test_get_by_category_rejects_unknown_category(self, populated_manifest):
        with pytest.raises(ValueError):
            get_by_category("NOT_REAL", populated_manifest)

    def test_get_stats_totals_match_manifest_size(self, populated_manifest):
        stats = get_stats(populated_manifest)
        assert stats["total"] == 3
        assert stats["labelled"] == 1
        assert stats["unlabelled"] == 2

    def test_get_stats_by_category_breakdown(self, populated_manifest):
        stats = get_stats(populated_manifest)
        political = stats["by_category"][CATEGORY_POLITICAL_MEDIA]
        assert political["total"] == 1
        assert political["synthetic"] == 1
        assert political["authentic"] == 0

    def test_get_stats_on_empty_manifest(self, manifest_path):
        init_manifest(manifest_path)
        stats = get_stats(manifest_path)
        assert stats["total"] == 0
        assert stats["by_category"] == {}
