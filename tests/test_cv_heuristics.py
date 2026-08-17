# =============================================================================
# tests/test_cv_heuristics.py  —  Stage 4 Heuristics Test Suite
# =============================================================================
# Covers watchdog/cv_engine.py's four §3.12 A heuristics: blink rate,
# emotional rigidity, viseme-phoneme mismatch, blending artifacts.
#
# Two things every test here is checking for, per the 2026-08-03 amendment:
#   1. Each heuristic returns score=None (not a default number) when the clip
#      can't actually support the measurement — no face, no audio, or too
#      short a contiguous face-present run. This is the primary thing worth
#      testing: a heuristic that silently fabricates a confident score from
#      insufficient data is worse than one that abstains (NFR-01).
#   2. The raw math (EAR, MAR, blink counting, displacement variance) is
#      correct on hand-constructed landmark arrays where the right answer is
#      known, not just "doesn't crash on real video."
#
# Real-video integration coverage (run_heuristics on an actual corpus clip)
# is deliberately NOT duplicated here — that was verified interactively
# during calibration (see docs/PROJECT_CONTEXT.md §4) and is expensive to
# run in a unit-test loop. These tests use synthetic landmark arrays and
# synthetic (cv2.VideoWriter-generated) videos, same convention as
# tests/test_cv_engine.py and tests/test_preprocessor.py.
# =============================================================================

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from watchdog.cv_engine import (
    _TrackedFrame,
    _ear,
    _mar,
    _count_blinks,
    _longest_face_present_segment,
    _extract_audio_wav,
    compute_blink_rate,
    compute_emotional_rigidity,
    compute_viseme_mismatch,
    compute_blending_artifacts,
    run_heuristics,
    RIGHT_EYE_EAR_IDX,
    LEFT_EYE_EAR_IDX,
    MOUTH_MAR_IDX,
)
from config import (
    FACE_MESH_SAMPLE_RATE_FPS,
    MIN_FACE_RUN_FOR_BLINK_S,
    MIN_FACE_RUN_FOR_RIGIDITY_S,
    MIN_FACE_RUN_FOR_VISEME_S,
    MIN_FRAMES_FOR_BLENDING,
    EAR_BLINK_THRESHOLD,
    EAR_MIN_CONSECUTIVE_FRAMES,
    HUMAN_BLINK_RATE_MIN,
    HUMAN_BLINK_RATE_MAX,
)


# =============================================================================
# Helpers to build synthetic landmark arrays with a known, computable answer
# =============================================================================

def _dummy_landmarks(overrides: dict) -> np.ndarray:
    """478x2 zero array with specific point indices overridden."""
    lm = np.zeros((478, 2), dtype=np.float32)
    for idx, (x, y) in overrides.items():
        lm[idx] = (x, y)
    return lm


def _eye_landmarks(idx, open_amount: float) -> dict:
    """
    Builds override points for one 6-point EAR eye contour with a chosen
    vertical opening. open_amount=0 -> fully closed (EAR=0); larger values
    -> more open. Horizontal (corner-to-corner) distance is fixed at 1.0,
    so EAR == open_amount by construction (vertical/horizontal, corners
    contribute the full horizontal denominator).
    """
    p1, p2, p3, p4, p5, p6 = idx
    return {
        p1: (0.0, 0.0), p4: (1.0, 0.0),                      # corners, distance 1.0
        p2: (0.3, open_amount / 2), p6: (0.3, -open_amount / 2),
        p3: (0.7, open_amount / 2), p5: (0.7, -open_amount / 2),
    }


def _open_face_landmarks() -> np.ndarray:
    overrides = {}
    overrides.update(_eye_landmarks(RIGHT_EYE_EAR_IDX, open_amount=0.4))
    overrides.update(_eye_landmarks(LEFT_EYE_EAR_IDX, open_amount=0.4))
    return _dummy_landmarks(overrides)


def _closed_face_landmarks() -> np.ndarray:
    overrides = {}
    overrides.update(_eye_landmarks(RIGHT_EYE_EAR_IDX, open_amount=0.0))
    overrides.update(_eye_landmarks(LEFT_EYE_EAR_IDX, open_amount=0.0))
    return _dummy_landmarks(overrides)


def _make_segment(n: int, landmarks_fn, bbox=(0, 0, 100, 100)) -> list:
    """n consecutive face-present _TrackedFrame at the Face-Mesh tick rate."""
    return [
        _TrackedFrame(t_s=i / FACE_MESH_SAMPLE_RATE_FPS, face_present=True,
                      landmarks_xy=landmarks_fn(i), bbox=bbox)
        for i in range(n)
    ]


# =============================================================================
# Raw math: EAR / MAR / blink counting / segment-finding
# =============================================================================

class TestEAR:

    def test_open_eye_gives_expected_ratio(self):
        lm = _dummy_landmarks(_eye_landmarks(RIGHT_EYE_EAR_IDX, open_amount=0.4))
        assert _ear(lm, RIGHT_EYE_EAR_IDX) == pytest.approx(0.4, abs=1e-5)

    def test_closed_eye_gives_zero(self):
        lm = _dummy_landmarks(_eye_landmarks(RIGHT_EYE_EAR_IDX, open_amount=0.0))
        assert _ear(lm, RIGHT_EYE_EAR_IDX) == pytest.approx(0.0, abs=1e-5)

    def test_zero_horizontal_distance_does_not_divide_by_zero(self):
        p1, p2, p3, p4, p5, p6 = RIGHT_EYE_EAR_IDX
        lm = _dummy_landmarks({p1: (0, 0), p4: (0, 0), p2: (0, 1), p6: (0, -1), p3: (0, 1), p5: (0, -1)})
        assert _ear(lm, RIGHT_EYE_EAR_IDX) == 0.0


class TestMAR:

    def test_open_mouth_gives_expected_ratio(self):
        idx = MOUTH_MAR_IDX
        lm = _dummy_landmarks({idx["left"]: (0, 0), idx["right"]: (1, 0),
                                idx["upper"]: (0.5, 0.3), idx["lower"]: (0.5, -0.3)})
        assert _mar(lm) == pytest.approx(0.6, abs=1e-5)

    def test_closed_mouth_gives_zero(self):
        idx = MOUTH_MAR_IDX
        lm = _dummy_landmarks({idx["left"]: (0, 0), idx["right"]: (1, 0),
                                idx["upper"]: (0.5, 0), idx["lower"]: (0.5, 0)})
        assert _mar(lm) == pytest.approx(0.0, abs=1e-5)


class TestCountBlinks:

    def test_no_closures_counts_zero(self):
        assert _count_blinks([0.4] * 20, threshold=0.21, min_consecutive=2) == 0

    def test_single_sub_threshold_frame_is_not_a_blink(self):
        """A lone noisy frame below threshold shouldn't count — needs min_consecutive."""
        series = [0.4, 0.4, 0.1, 0.4, 0.4]
        assert _count_blinks(series, threshold=0.21, min_consecutive=2) == 0

    def test_two_consecutive_closures_count_as_one_blink(self):
        series = [0.4, 0.4, 0.1, 0.1, 0.4, 0.4]
        assert _count_blinks(series, threshold=0.21, min_consecutive=2) == 1

    def test_three_separate_blinks_counted_separately(self):
        series = [0.4, 0.1, 0.1, 0.4, 0.4, 0.1, 0.1, 0.4, 0.1, 0.1, 0.4]
        assert _count_blinks(series, threshold=0.21, min_consecutive=2) == 3

    def test_closure_still_open_at_series_end_still_counts(self):
        series = [0.4, 0.4, 0.1, 0.1, 0.1]
        assert _count_blinks(series, threshold=0.21, min_consecutive=2) == 1


class TestLongestFacePresentSegment:

    def test_empty_list_returns_empty(self):
        assert _longest_face_present_segment([]) == []

    def test_finds_longest_run_not_first_or_last(self):
        frames = [
            _TrackedFrame(t_s=0, face_present=True),
            _TrackedFrame(t_s=1, face_present=False),
            _TrackedFrame(t_s=2, face_present=True),
            _TrackedFrame(t_s=3, face_present=True),
            _TrackedFrame(t_s=4, face_present=True),
            _TrackedFrame(t_s=5, face_present=False),
        ]
        segment = _longest_face_present_segment(frames)
        assert [f.t_s for f in segment] == [2, 3, 4]


# =============================================================================
# compute_blink_rate — abstention gate + rate math
# =============================================================================

class TestComputeBlinkRate:

    def test_abstains_below_minimum_run_length(self):
        n = int((MIN_FACE_RUN_FOR_BLINK_S - 1) * FACE_MESH_SAMPLE_RATE_FPS)
        frames = _make_segment(n, lambda i: _open_face_landmarks())
        result = compute_blink_rate(frames)
        assert result.score is None
        assert "below the" in result.abstain_reason

    def test_normal_blink_rate_scores_zero_deviation(self):
        """3 blinks over 12s = 15/min, inside HUMAN_BLINK_RATE_MIN..MAX (12-25)."""
        n = 12 * FACE_MESH_SAMPLE_RATE_FPS
        blink_frames = {20, 21, 50, 51, 80, 81}   # 3 blinks, 2 frames each

        def landmarks_fn(i):
            return _closed_face_landmarks() if i in blink_frames else _open_face_landmarks()

        frames = _make_segment(n, landmarks_fn)
        result = compute_blink_rate(frames)
        assert result.score == pytest.approx(0.0)
        assert result.raw["n_blinks"] == 3
        assert HUMAN_BLINK_RATE_MIN <= result.raw["blink_rate_per_min"] <= HUMAN_BLINK_RATE_MAX

    def test_zero_blinks_scores_above_zero(self):
        n = 12 * FACE_MESH_SAMPLE_RATE_FPS
        frames = _make_segment(n, lambda i: _open_face_landmarks())
        result = compute_blink_rate(frames)
        assert result.raw["n_blinks"] == 0
        assert result.score > 0.0


# =============================================================================
# compute_emotional_rigidity — abstention gate + variance math
# =============================================================================

class TestComputeEmotionalRigidity:

    def test_abstains_below_minimum_run_length(self):
        n = int((MIN_FACE_RUN_FOR_RIGIDITY_S - 0.5) * FACE_MESH_SAMPLE_RATE_FPS)
        frames = _make_segment(n, lambda i: _open_face_landmarks())
        result = compute_emotional_rigidity(frames)
        assert result.score is None

    def test_identical_landmarks_across_frames_scores_maximally_rigid(self):
        n = int((MIN_FACE_RUN_FOR_RIGIDITY_S + 1) * FACE_MESH_SAMPLE_RATE_FPS)
        fixed = _open_face_landmarks()
        frames = _make_segment(n, lambda i: fixed)
        result = compute_emotional_rigidity(frames)
        assert result.raw["displacement_variance"] == pytest.approx(0.0, abs=1e-9)
        assert result.score == pytest.approx(1.0)

    def test_varying_landmarks_score_lower_than_identical(self):
        n = int((MIN_FACE_RUN_FOR_RIGIDITY_S + 1) * FACE_MESH_SAMPLE_RATE_FPS)
        rng = np.random.default_rng(seed=7)

        def landmarks_fn(i):
            lm = _open_face_landmarks().copy()
            lm += rng.normal(scale=15.0, size=lm.shape).astype(np.float32)
            return lm

        varying = compute_emotional_rigidity(_make_segment(n, landmarks_fn))
        rigid = compute_emotional_rigidity(_make_segment(n, lambda i: _open_face_landmarks()))
        assert varying.raw["displacement_variance"] > rigid.raw["displacement_variance"]
        assert varying.score < rigid.score


# =============================================================================
# compute_viseme_mismatch — abstention gates (short run, no audio)
# =============================================================================

def _write_silent_video(path, num_frames=150, fps=25.0, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _mux_sine_wave_audio(video_path, duration_s=6.0):
    """
    Adds a synthetic sine-wave audio track to an existing (silent, no-audio)
    video via ffmpeg, for exercising the audio-present path of
    _extract_audio_wav() / compute_viseme_mismatch() without needing a real
    recorded clip. Returns the new path (video_path with "_audio" suffix).
    """
    video_path = str(video_path)
    out_path = video_path.replace(".mp4", "_audio.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
         "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return out_path


class TestExtractAudioWav:
    """
    Regression coverage for the viseme audio-decode cost fix
    (docs/PROJECT_CONTEXT.md — "viseme audio-decode cost"): librosa.load()
    has no fast native path for MP4, so audio is now extracted to WAV once
    via ffmpeg (_extract_audio_wav()) instead of loading the MP4 directly on
    every compute_viseme_mismatch() call.
    """

    def test_video_with_no_audio_stream_returns_none(self, tmp_path):
        video_path = _write_silent_video(tmp_path / "silent.mp4")
        assert _extract_audio_wav(video_path) is None

    def test_missing_file_returns_none_not_an_exception(self, tmp_path):
        assert _extract_audio_wav(tmp_path / "does_not_exist.mp4") is None

    def test_video_with_audio_stream_produces_a_real_wav_file(self, tmp_path):
        silent = _write_silent_video(tmp_path / "base.mp4")
        with_audio = _mux_sine_wave_audio(silent)

        wav_path = _extract_audio_wav(with_audio)
        try:
            assert wav_path is not None
            assert wav_path.exists()
            assert wav_path.stat().st_size > 44   # more than a bare WAV header
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)


class TestComputeVisemeMismatch:

    def test_abstains_below_minimum_run_length(self, tmp_path):
        n = int((MIN_FACE_RUN_FOR_VISEME_S - 0.5) * FACE_MESH_SAMPLE_RATE_FPS)
        frames = _make_segment(n, lambda i: _open_face_landmarks())
        result = compute_viseme_mismatch(frames, tmp_path / "unused.mp4")
        assert result.score is None
        assert "below the" in result.abstain_reason

    def test_abstains_cleanly_when_no_audio_track(self, tmp_path):
        """A video with no audio stream must abstain, not raise."""
        video_path = _write_silent_video(tmp_path / "silent.mp4")
        n = int((MIN_FACE_RUN_FOR_VISEME_S + 1) * FACE_MESH_SAMPLE_RATE_FPS)
        frames = _make_segment(n, lambda i: _open_face_landmarks())
        result = compute_viseme_mismatch(frames, video_path)
        assert result.score is None
        assert result.abstain_reason is not None

    def test_abstains_cleanly_when_audio_path_is_none(self):
        """
        run_heuristics() passes None straight through when
        _extract_audio_wav() found no audio track — must abstain without
        attempting to load anything.
        """
        n = int((MIN_FACE_RUN_FOR_VISEME_S + 1) * FACE_MESH_SAMPLE_RATE_FPS)
        frames = _make_segment(n, lambda i: _open_face_landmarks())
        result = compute_viseme_mismatch(frames, None)
        assert result.score is None
        assert result.abstain_reason == "no audio track"


# =============================================================================
# compute_blending_artifacts — abstention gate + ratio/discontinuity scoring
# =============================================================================

class TestComputeBlendingArtifacts:

    def test_abstains_below_minimum_frame_count(self):
        frames = [_TrackedFrame(t_s=i, face_present=True, face_lap_var=100.0, surround_lap_var=100.0)
                  for i in range(MIN_FRAMES_FOR_BLENDING - 1)]
        result = compute_blending_artifacts(frames)
        assert result.score is None

    def test_equal_sharpness_scores_low(self):
        frames = [_TrackedFrame(t_s=i, face_present=True, face_lap_var=100.0, surround_lap_var=100.0,
                                 edge_discontinuity=0.0)
                  for i in range(MIN_FRAMES_FOR_BLENDING + 5)]
        result = compute_blending_artifacts(frames)
        assert result.raw["mean_face_surround_lap_ratio"] == pytest.approx(1.0)
        assert result.score == pytest.approx(0.0, abs=1e-6)

    def test_large_sharpness_mismatch_scores_high(self):
        frames = [_TrackedFrame(t_s=i, face_present=True, face_lap_var=500.0, surround_lap_var=50.0,
                                 edge_discontinuity=0.0)
                  for i in range(MIN_FRAMES_FOR_BLENDING + 5)]
        result = compute_blending_artifacts(frames)
        assert result.score > 0.5

    def test_frames_without_pixel_stats_are_excluded(self):
        usable = [_TrackedFrame(t_s=i, face_present=True, face_lap_var=100.0, surround_lap_var=100.0,
                                 edge_discontinuity=0.0) for i in range(MIN_FRAMES_FOR_BLENDING)]
        no_face = [_TrackedFrame(t_s=i, face_present=False) for i in range(20)]
        result = compute_blending_artifacts(usable + no_face)
        assert result.score is not None
        assert result.raw["n_frames"] == MIN_FRAMES_FOR_BLENDING


# =============================================================================
# run_heuristics — orchestrator integration
# =============================================================================

class TestRunHeuristics:

    def test_missing_file_returns_none(self, tmp_path):
        assert run_heuristics(tmp_path / "does_not_exist.mp4") is None

    def test_face_free_clip_abstains_on_everything_without_crashing(self, tmp_path):
        """
        A face-free clip should make every heuristic abstain cleanly (no
        face at all means no contiguous run, and no usable pixel stats for
        blending either) rather than raise.
        """
        video_path = _write_silent_video(tmp_path / "face_free.mp4", num_frames=100)
        result = run_heuristics(video_path)
        assert result is not None
        assert result.blink_rate.score is None
        assert result.emotional_rigidity.score is None
        assert result.viseme_mismatch.score is None
        assert result.blending_artifacts.score is None
        assert result.total_time_s >= result.tracking_time_s

    def test_no_temp_wav_file_leaks_after_running(self, tmp_path):
        """
        run_heuristics() extracts audio to a temp WAV via _extract_audio_wav()
        and must clean it up afterward regardless of outcome — otherwise
        every clip evaluated (e.g. across the full 32-clip CV-eligible
        corpus) leaks one file into the OS temp directory.
        """
        import tempfile

        before = set(Path(tempfile.gettempdir()).glob("watchdog_audio_*"))

        video_path = _write_silent_video(tmp_path / "face_free.mp4", num_frames=100)
        with_audio = _mux_sine_wave_audio(video_path)
        result = run_heuristics(with_audio)
        assert result is not None

        after = set(Path(tempfile.gettempdir()).glob("watchdog_audio_*"))
        assert after == before
