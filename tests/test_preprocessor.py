# =============================================================================
# tests/test_preprocessor.py  —  Preprocessing Pipeline Test Suite
# =============================================================================
# Covers Table 3.5 (§3.10) end to end:
#   Step 1  Frame Extraction (OpenCV)          -> extract_frames()
#   Step 2  Facial Alignment (MTCNN)           -> extract_faces()
#   Step 3  Tokenization & Stop-Word Removal   -> preprocess_text()
#   Step 4  Noise Reduction (Gaussian Blur)    -> extract_frames() kernel arg
#
# All video fixtures are generated synthetically with cv2.VideoWriter — no
# external media files are required to run this suite, and no student-specific
# ad footage needs to be committed to the repo just to run tests.
#
# RUN:
#   pytest tests/test_preprocessor.py -v
# =============================================================================

import shutil

import cv2
import numpy as np
import pytest

from watchdog import preprocessor
from watchdog.preprocessor import extract_frames, extract_faces, preprocess_text, _fallback_preprocess


# =============================================================================
# Fixtures: synthetic test video generation
# =============================================================================

def _write_synthetic_video(path, num_frames=50, fps=25.0, size=(64, 64), pattern="solid"):
    """
    Generates a tiny synthetic MP4 for testing — no real footage needed.

    pattern='solid': every frame is a flat colour that cycles — fast, useful
                      for frame-count / blur tests where content doesn't matter.
    pattern='noise':  random noise per frame — used to sanity-check that
                      Gaussian blur actually changes pixel values.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    rng = np.random.default_rng(seed=42)

    for i in range(num_frames):
        if pattern == "noise":
            frame = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        else:
            fill = i % 256
            frame = np.full((size[1], size[0], 3), fill_value=fill, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path


@pytest.fixture()
def synthetic_video_25fps(tmp_path):
    """50 frames @ 25fps = 2 seconds of video, solid colour pattern."""
    path = tmp_path / "synthetic_ad_25fps.mp4"
    return _write_synthetic_video(path, num_frames=50, fps=25.0)


@pytest.fixture()
def synthetic_video_noise(tmp_path):
    """30 frames of random noise — for verifying Gaussian blur has an effect."""
    path = tmp_path / "synthetic_noise.mp4"
    return _write_synthetic_video(path, num_frames=30, fps=25.0, pattern="noise")


# =============================================================================
# STEP 1 + 4 — extract_frames() : sampling rate + Gaussian blur
# =============================================================================

class TestExtractFrames:

    def test_missing_file_returns_empty_list(self, tmp_path):
        missing = tmp_path / "does_not_exist.mp4"
        assert extract_frames(missing) == []

    def test_sample_rate_reduces_frame_count(self, synthetic_video_25fps):
        """
        §3.10 rationale: sampling at 1 fps instead of the native 25 fps keeps
        CPU-only inference tractable (§1.7 Limitations). 50 frames @ 25fps,
        sampled at 1 fps, should yield ~2 frames (one per second of video).
        """
        frames = extract_frames(synthetic_video_25fps, sample_fps=1)
        # 50 frames at 25fps = 2.0s of video -> expect 2 sampled frames
        assert 1 <= len(frames) <= 3   # small tolerance for rounding at boundaries

    def test_full_sample_rate_returns_all_frames(self, synthetic_video_25fps):
        """Sampling at the native fps should return (approximately) every frame."""
        frames = extract_frames(synthetic_video_25fps, sample_fps=25)
        assert len(frames) == 50

    def test_frames_have_correct_shape(self, synthetic_video_25fps):
        frames = extract_frames(synthetic_video_25fps, sample_fps=25)
        assert frames[0].shape == (64, 64, 3)   # (H, W, 3) BGR, per docstring

    def test_gaussian_blur_changes_pixel_values(self, synthetic_video_noise):
        """
        §3.10 Step 4: blur must actually suppress noise (change pixel values),
        or the WhatsApp/Facebook compression-artifact mitigation described in
        the literature isn't really happening.
        """
        frames = extract_frames(
            synthetic_video_noise, sample_fps=25, gaussian_kernel=(5, 5)
        )
        # Compare against the frame re-read WITHOUT blur applied
        cap = cv2.VideoCapture(str(synthetic_video_noise))
        _, raw_frame = cap.read()
        cap.release()

        assert not np.array_equal(frames[0], raw_frame), (
            "Gaussian blur had no effect — noise-frame pixels are identical "
            "to the unprocessed source frame."
        )

    def test_larger_kernel_smooths_more_than_smaller_kernel(self, synthetic_video_noise):
        """Sanity check on the blur parameter itself: bigger kernel = more smoothing."""
        light_blur = extract_frames(synthetic_video_noise, sample_fps=25, gaussian_kernel=(3, 3))
        heavy_blur = extract_frames(synthetic_video_noise, sample_fps=25, gaussian_kernel=(9, 9))

        # Standard deviation of pixel values should drop as blur increases
        assert heavy_blur[0].std() <= light_blur[0].std()


# =============================================================================
# STEP 2 — extract_faces() : MTCNN facial alignment
# =============================================================================

class TestExtractFaces:

    def test_no_faces_in_solid_colour_frames_returns_empty(self, synthetic_video_25fps):
        """
        A flat-colour synthetic frame has no face-like structure at all.
        MTCNN should correctly detect zero faces rather than erroring out —
        this proves the function handles the "no face found" path safely,
        which matters because real scraped content sometimes has no visible
        face (e.g. voice-over-only ads, B-roll, or an obscured presenter).
        """
        frames = extract_frames(synthetic_video_25fps, sample_fps=25)
        results = extract_faces(frames[:5])   # only check a few frames — MTCNN is slow on CPU
        assert results == []

    def test_returns_list_of_tuples_when_present(self, synthetic_video_25fps):
        """
        Structural contract check: even on a no-face input, the return type
        must remain a list (not None), so callers (cv_engine.py, Day 5+) can
        safely iterate without a None-check at every call site.
        """
        frames = extract_frames(synthetic_video_25fps, sample_fps=25)
        results = extract_faces(frames[:3])
        assert isinstance(results, list)


# =============================================================================
# STEP 3 — preprocess_text() : tokenisation + stop-word removal
# =============================================================================

SAMPLE_TEXT = (
    "MTN Nigeria today announced a new AI-powered advertising campaign. "
    "The campaign features synthetic avatars instead of human actors. "
    "Critics say this displaces local creative talent."
)


class TestPreprocessText:

    def test_returns_expected_keys(self):
        result = preprocess_text(SAMPLE_TEXT)
        expected_keys = {"cleaned", "sentences", "tokens", "word_count",
                         "sentence_count", "original"}
        assert expected_keys.issubset(result.keys())

    def test_original_text_preserved_for_ip_similarity_check(self):
        """
        §3.12 B: the IP similarity matrix needs the ORIGINAL (non-lowercased,
        non-stripped) text to compute meaningful sentence-transformer
        embeddings — cleaning must not destroy the source for that step.
        """
        result = preprocess_text(SAMPLE_TEXT)
        assert result["original"] == SAMPLE_TEXT

    def test_sentence_count_matches_input_structure(self):
        """SAMPLE_TEXT has exactly 3 sentences (3 periods, no abbreviations)."""
        result = preprocess_text(SAMPLE_TEXT)
        assert result["sentence_count"] == 3

    def test_cleaned_text_is_lowercase(self):
        result = preprocess_text(SAMPLE_TEXT)
        assert result["cleaned"] == result["cleaned"].lower()

    def test_empty_string_does_not_crash(self):
        result = preprocess_text("")
        assert result["sentence_count"] == 0
        assert result["cleaned"] == ""

    def test_gracefully_handles_missing_spacy_model(self, monkeypatch):
        """
        Regression coverage for a real scenario: a student who has run
        `pip install -r requirements.txt` but forgotten the separate
        `python -m spacy download en_core_web_sm` step (a common setup
        mistake — it's not a pip dependency, it's a separate model download).
        preprocess_text() must degrade to _fallback_preprocess(), not crash
        the whole pipeline.

        Patches preprocessor._load_spacy_model() itself (same idiom
        test_nlp_engine.py already uses for nlp_engine._load_perplexity_model())
        rather than spacy.load() directly — since the 2026-08-13 caching fix,
        _load_spacy_model() caches its result at module level, so a real
        successful load earlier in the test session (e.g. the sentence_count
        tests above) would make patching spacy.load() alone a no-op: the
        cached pipeline would already be sitting in preprocessor._NLP and
        this test would silently stop testing the fallback path it claims to.
        """
        def _raise_os_error(*args, **kwargs):
            raise OSError("model not found (simulated)")

        monkeypatch.setattr(preprocessor, "_load_spacy_model", _raise_os_error)

        result = preprocess_text(SAMPLE_TEXT)   # must not raise
        assert result["sentence_count"] > 0
        assert "mtn" in result["cleaned"]


class TestFallbackPreprocess:
    """
    Direct tests of _fallback_preprocess() — the dependency-free code path.
    These always run, regardless of whether en_core_web_sm is installed on
    the machine executing the test suite.
    """

    def test_splits_sentences_on_terminal_punctuation(self):
        result = _fallback_preprocess(SAMPLE_TEXT)
        assert result["sentence_count"] == 3

    def test_strips_punctuation_and_numbers_from_cleaned_text(self):
        result = _fallback_preprocess("Ad campaign #2026 costs ₦500,000 total!")
        assert not any(ch.isdigit() for ch in result["cleaned"])

    def test_filters_short_tokens(self):
        """Tokens of length <= 2 are dropped (e.g. 'a', 'to', 'is')."""
        result = _fallback_preprocess("It is a big ad for MTN Nigeria")
        assert "is" not in result["tokens"]
        assert "big" in result["tokens"]

    def test_original_field_is_untouched(self):
        raw = "MTN Nigeria's NEW campaign — 2026!"
        result = _fallback_preprocess(raw)
        assert result["original"] == raw
