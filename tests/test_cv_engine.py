# =============================================================================
# tests/test_cv_engine.py  —  Sparse-Detection / Dense-Landmark Pipeline Tests
# =============================================================================
# Covers watchdog/cv_engine.py (Stage 2, FR-02/FR-03):
#   letterbox() / unletterbox_box()  — fixed-resolution normalisation, no stretch
#   analyze_clip()                   — sparse MTCNN + dense Face Mesh, face-free
#                                       frames handled as the normal case
#
# All video fixtures are generated synthetically with cv2.VideoWriter, same
# approach as tests/test_preprocessor.py — no real footage needed to run this
# suite. analyze_clip() still loads the real facenet-pytorch and MediaPipe
# models (no mocking), consistent with this project's "test against real
# behaviour" convention (see docs/PROJECT_CONTEXT.md §6) — it's just pointed
# at synthetic, guaranteed-face-free frames so the outcome is deterministic.
#
# RUN:
#   pytest tests/test_cv_engine.py -v
# =============================================================================

import cv2
import numpy as np
import pytest

from watchdog.cv_engine import (
    letterbox,
    unletterbox_box,
    analyze_clip,
    _longest_true_run,
    _padded_crop,
    _iou,
)


# =============================================================================
# Fixtures: synthetic test video generation (mirrors test_preprocessor.py)
# =============================================================================

def _write_synthetic_video(path, num_frames=50, fps=25.0, size=(64, 64)):
    """Flat-colour synthetic MP4 — guaranteed to contain no face-like structure."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        fill = i % 256
        frame = np.full((size[1], size[0], 3), fill_value=fill, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def synthetic_video_25fps(tmp_path):
    """50 frames @ 25fps = 2 seconds, no faces."""
    path = tmp_path / "cv_engine_synthetic_25fps.mp4"
    return _write_synthetic_video(path, num_frames=50, fps=25.0)


# =============================================================================
# letterbox() / unletterbox_box() — fixed-resolution normalisation
# =============================================================================

class TestLetterbox:

    @pytest.mark.parametrize("orientation,shape", [
        ("landscape", (720, 1280, 3)),   # (H, W, 3)
        ("portrait", (1280, 720, 3)),
        ("square", (900, 900, 3)),
    ])
    def test_output_matches_target_canvas_size(self, orientation, shape):
        """
        Regardless of source orientation (the corpus spans landscape,
        portrait, and square — see data/strata.csv), letterbox() must always
        return the fixed target canvas size so the detector sees one
        consistent input shape.
        """
        frame = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        target = (640, 640)
        canvas, scale, pad = letterbox(frame, target_size=target)
        assert canvas.shape == (target[1], target[0], 3)

    def test_does_not_stretch_aspect_ratio(self):
        """
        A 720x1280 portrait frame letterboxed into a square canvas must be
        scaled uniformly (same factor on both axes), not stretched — a
        non-uniform scale would distort facial geometry, exactly the kind of
        artifact this pipeline is trying to detect, not introduce.
        """
        frame = np.random.randint(0, 256, size=(1280, 720, 3), dtype=np.uint8)  # (H, W, 3)
        canvas, scale, pad = letterbox(frame, target_size=(640, 640))
        h, w = frame.shape[:2]
        expected_scale = min(640 / w, 640 / h)
        assert scale == pytest.approx(expected_scale)

    def test_landscape_frame_gets_vertical_padding(self):
        frame = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)  # (H, W, 3)
        canvas, scale, (pad_x, pad_y) = letterbox(frame, target_size=(640, 640))
        assert pad_y > 0
        assert pad_x == 0

    def test_portrait_frame_gets_horizontal_padding(self):
        frame = np.random.randint(0, 256, size=(1280, 720, 3), dtype=np.uint8)  # (H, W, 3)
        canvas, scale, (pad_x, pad_y) = letterbox(frame, target_size=(640, 640))
        assert pad_x > 0
        assert pad_y == 0

    def test_square_frame_gets_no_padding(self):
        frame = np.random.randint(0, 256, size=(900, 900, 3), dtype=np.uint8)
        canvas, scale, pad = letterbox(frame, target_size=(640, 640))
        assert pad == (0, 0)

    def test_unletterbox_box_inverts_letterbox_scaling(self):
        """
        A box detected in letterboxed coordinates, mapped back via
        unletterbox_box(), must recover the original-frame coordinates —
        this is the mapping detect_face_bbox() relies on to return boxes
        callers can use directly against the un-letterboxed frame.
        """
        frame = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)  # (H, W, 3)
        target = (640, 640)
        _, scale, pad = letterbox(frame, target_size=target)

        orig_box = (100.0, 50.0, 200.0, 150.0)   # (x, y, w, h) in original coords
        ox, oy, ow, oh = orig_box
        pad_x, pad_y = pad
        lb_box = (ox * scale + pad_x, oy * scale + pad_y, ow * scale, oh * scale)

        recovered = unletterbox_box(lb_box, scale, pad)
        assert recovered == pytest.approx(orig_box)


# =============================================================================
# _longest_true_run() / _padded_crop() — small pure helpers
# =============================================================================

class TestLongestTrueRun:

    def test_empty_list_returns_zero(self):
        assert _longest_true_run([]) == 0

    def test_no_true_values_returns_zero(self):
        assert _longest_true_run([False, False, False]) == 0

    def test_finds_longest_contiguous_run(self):
        assert _longest_true_run([True, True, False, True, True, True, False]) == 3

    def test_all_true_returns_full_length(self):
        assert _longest_true_run([True] * 5) == 5


class TestPaddedCrop:

    def test_crop_within_bounds(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _padded_crop(frame, (40, 40, 20, 20), margin_frac=0.3)
        assert result is not None
        crop, origin = result
        assert crop.shape[0] > 20 and crop.shape[1] > 20   # margin applied
        assert origin == (34, 34)   # x - margin(6), y - margin(6)

    def test_crop_clamped_to_frame_edges(self):
        """A box near the frame edge must clamp, not raise or wrap around."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _padded_crop(frame, (90, 90, 20, 20), margin_frac=0.3)
        assert result is not None
        crop, origin = result
        # unclamped would be 26x26 (20 + 2*margin(6)); clamped to the frame
        # edge at x=100,y=100 leaves a 16x16 crop starting at x=y=84
        assert crop.shape[0] == 16 and crop.shape[1] == 16
        assert origin == (84, 84)

    def test_degenerate_box_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _padded_crop(frame, (200, 200, 10, 10), margin_frac=0.3) is None


# =============================================================================
# _iou() / detect_face_bbox() continuity — identity-flicker regression
# =============================================================================
# Regression coverage for the bug in docs/PROJECT_CONTEXT.md §4/bug log:
# detect_face_bbox()'s IoU-continuity branch referenced
# config.MTCNN_TRACK_IOU_THRESHOLD before that constant existed anywhere,
# which would have raised NameError the first time this function ran with a
# prev_box. Exercises the actual selection logic with a stubbed detector
# (no real face image needed) rather than just checking the import works.

class TestIoU:

    def test_identical_boxes_have_iou_one(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_have_iou_zero(self):
        assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    def test_partial_overlap_computed_correctly(self):
        # a: x[0,10] y[0,10]; b: x[5,15] y[0,10] -> intersection 5x10=50, union=100+100-50=150
        assert _iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


class TestDetectFaceBboxContinuity:

    @staticmethod
    def _identity_frame():
        # Frame already at LETTERBOX_TARGET_SIZE (640x640) -> letterbox() is a
        # no-op (scale=1.0, pad=(0,0)), so candidate boxes returned by the
        # stub detector map straight through to detect_face_bbox()'s output.
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def test_picks_candidate_overlapping_previous_box_over_higher_confidence(self, monkeypatch):
        """
        The whole point of the fix: a higher-confidence candidate that does
        NOT overlap the previous track must lose to a lower-confidence one
        that does — this is what keeps identity from flickering to a
        different face just because MTCNN scored it higher this tick.
        """
        import watchdog.cv_engine as cv_engine

        prev_box = (100.0, 100.0, 50.0, 50.0)
        near_candidate = ((105.0, 105.0, 50.0, 50.0), 0.80)   # overlaps prev_box
        far_candidate = ((400.0, 400.0, 50.0, 50.0), 0.99)    # higher confidence, no overlap

        monkeypatch.setattr(
            cv_engine, "_get_detector",
            lambda: (lambda rgb: [near_candidate, far_candidate]),
        )

        box = cv_engine.detect_face_bbox(self._identity_frame(), prev_box=prev_box)
        assert box == pytest.approx(near_candidate[0])

    def test_falls_back_to_highest_confidence_when_nothing_overlaps(self, monkeypatch):
        """A real cut / fresh track: no candidate overlaps prev_box, so highest confidence wins."""
        import watchdog.cv_engine as cv_engine

        prev_box = (100.0, 100.0, 50.0, 50.0)
        far_low = ((400.0, 400.0, 50.0, 50.0), 0.70)
        far_high = ((450.0, 450.0, 50.0, 50.0), 0.95)

        monkeypatch.setattr(
            cv_engine, "_get_detector",
            lambda: (lambda rgb: [far_low, far_high]),
        )

        box = cv_engine.detect_face_bbox(self._identity_frame(), prev_box=prev_box)
        assert box == pytest.approx(far_high[0])

    def test_no_previous_box_picks_highest_confidence(self, monkeypatch):
        import watchdog.cv_engine as cv_engine

        low = ((10.0, 10.0, 50.0, 50.0), 0.5)
        high = ((200.0, 200.0, 50.0, 50.0), 0.9)

        monkeypatch.setattr(cv_engine, "_get_detector", lambda: (lambda rgb: [low, high]))

        box = cv_engine.detect_face_bbox(self._identity_frame(), prev_box=None)
        assert box == pytest.approx(high[0])

    def test_no_candidates_returns_none(self, monkeypatch):
        import watchdog.cv_engine as cv_engine

        monkeypatch.setattr(cv_engine, "_get_detector", lambda: (lambda rgb: []))
        assert cv_engine.detect_face_bbox(self._identity_frame(), prev_box=(0, 0, 10, 10)) is None


# =============================================================================
# analyze_clip() — face-free frames are the normal case, not exceptional
# =============================================================================

class TestAnalyzeClip:

    def test_missing_file_returns_none(self, tmp_path):
        missing = tmp_path / "does_not_exist.mp4"
        assert analyze_clip(missing) is None

    def test_face_free_clip_reports_zero_faces_not_an_error(self, synthetic_video_25fps):
        """
        A flat-colour synthetic clip has no face-like structure anywhere.
        analyze_clip() must complete normally (not raise) and report zero
        face-present frames — this is the expected outcome for B-roll/
        product-shot-heavy footage, not an error condition (see module
        docstring).
        """
        result = analyze_clip(synthetic_video_25fps)
        assert result is not None
        assert result.total_frames_sampled > 0
        assert result.frames_with_face == 0
        assert result.longest_contiguous_face_run_s == 0.0

    def test_timing_fields_are_recorded_and_positive(self, synthetic_video_25fps):
        """
        The whole point of Stage 2 is to measure real stage costs rather
        than assume them — every timing field must reflect real elapsed
        time, and total_time_s must equal the sum of the three stages.

        Face Mesh only ever runs against a tracked MTCNN crop (see
        analyze_clip's docstring), so on this face-free fixture
        face_mesh_calls is correctly 0, not an error — MTCNN never hands it
        a box to work with.
        """
        result = analyze_clip(synthetic_video_25fps)
        assert result.mtcnn_calls > 0
        assert result.mtcnn_time_s >= 0
        assert result.face_mesh_calls == 0
        assert result.face_mesh_time_s == 0
        assert result.frame_extraction_time_s >= 0
        assert result.total_time_s == pytest.approx(
            result.mtcnn_time_s + result.face_mesh_time_s + result.frame_extraction_time_s
        )

    def test_duration_matches_native_fps_and_frame_count(self, synthetic_video_25fps):
        """50 frames @ 25fps = 2.0s of video."""
        result = analyze_clip(synthetic_video_25fps)
        assert result.duration_s == pytest.approx(2.0, abs=0.2)


class TestFaceLandmarkerRealFace:
    """
    The face-free fixture above proves analyze_clip() doesn't crash or
    misreport when there's nothing to find, but it can never exercise the
    Face Mesh call path (face_mesh_calls stays 0 whenever MTCNN never hands
    it a box). This uses detect_face_bbox() + the real FaceLandmarker
    directly against a real corpus frame to prove that path actually works
    end to end — this is exactly the gap that let the mediapipe.solutions
    API removal go undetected until analyze_clip() was run for real.
    """

    def test_detects_landmarks_on_real_face_crop(self):
        import mediapipe as mp

        from watchdog.cv_engine import detect_face_bbox, _get_face_landmarker, _padded_crop

        frame_path = (
            "/tmp/claude-1000/-home-arukwe-watchdog/32334010-de4e-4707-8c53-ff3a6b9b3099"
            "/scratchpad/bench_frames/f01_720x888.jpg"
        )
        frame = cv2.imread(frame_path)
        if frame is None:
            pytest.skip(f"benchmark corpus frame not available at {frame_path}")

        box = detect_face_bbox(frame)
        assert box is not None

        result = _padded_crop(frame, box, margin_frac=0.3)
        assert result is not None
        crop, origin = result
        assert crop.size > 0

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        result = _get_face_landmarker().detect(mp_image)
        assert len(result.face_landmarks) > 0
        assert len(result.face_landmarks[0]) == 478   # FaceLandmarker's fixed mesh size
