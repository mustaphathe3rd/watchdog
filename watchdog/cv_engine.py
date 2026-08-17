# =============================================================================
# watchdog/cv_engine.py  —  Sparse-Detection / Dense-Landmark CV Pipeline
# =============================================================================
# PURPOSE:
#   Stage 2 of the CV/deepfake engine (FR-02, FR-03). Extends
#   watchdog/preprocessor.py's per-frame extract_faces() (which runs MTCNN on
#   every sampled frame) with a two-tier design driven by real CPU-cost
#   measurements: MTCNN is the expensive stage on CPU
#   (benchmark_face_detection.py), so it only runs sparsely
#   (config.MTCNN_SAMPLE_RATE_FPS, ~1 fps) to locate the face. Between MTCNN
#   hits, the bounding box is reused for a MediaPipe Face Mesh pass on just
#   that crop, running much denser (config.FACE_MESH_SAMPLE_RATE_FPS, ~10
#   fps) since Face Mesh is cheap relative to MTCNN. This gives blink-rate /
#   viseme-sync analysis (FR-03) the temporal resolution it needs without
#   paying MTCNN's cost at that same rate.
#
# WHY NOT JUST RUN MTCNN AT 10FPS DIRECTLY?
#   Stage 0 measured MTCNN at ~0.8s/frame on CPU. At 10fps that's 8 seconds
#   of MTCNN time per second of video — already 8x realtime before any other
#   pipeline stage runs. Sparse detection + dense cheap landmarking is what
#   makes CPU-only inference (§1.7 Limitations) tractable at all.
#
# FACE-FREE FRAMES ARE NORMAL, NOT EXCEPTIONAL:
#   Nigerian ad/event footage cuts to B-roll, product shots, and text cards
#   constantly (see compute_strata.py's genre classifications — news_package,
#   brand_tvc B-roll, etc.). A clip with zero faces for stretches of its
#   runtime is the expected case, not an error condition. This module never
#   raises on a face-free frame; it counts it and moves on. The longest
#   contiguous face-present run is reported explicitly because it — not the
#   clip's total duration — determines whether blink-rate analysis is even
#   possible for a given clip (you need a continuous run of frames with a
#   tracked face to measure a blink).
# =============================================================================

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import (
    MTCNN_SAMPLE_RATE_FPS,
    FACE_MESH_SAMPLE_RATE_FPS,
    MTCNN_MIN_FACE_SIZE,
    MTCNN_TRACK_IOU_THRESHOLD,
    LETTERBOX_TARGET_SIZE,
    FACE_LANDMARKER_MODEL_PATH,
    MIN_FACE_RUN_FOR_BLINK_S,
    MIN_FACE_RUN_FOR_RIGIDITY_S,
    MIN_FACE_RUN_FOR_VISEME_S,
    MIN_FRAMES_FOR_BLENDING,
    EAR_BLINK_THRESHOLD,
    EAR_MIN_CONSECUTIVE_FRAMES,
    HUMAN_BLINK_RATE_MIN,
    HUMAN_BLINK_RATE_MAX,
    EMOTIONAL_RIGIDITY_THRESHOLD,
    VISEME_CORRELATION_THRESHOLD,
    BLENDING_SHARPNESS_RATIO_DEVIATION_THRESHOLD,
    BLENDING_EDGE_DISCONTINUITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Padding applied around a tracked MTCNN bounding box before cropping for
# Face Mesh — a tight face-only crop clips the jawline/forehead landmarks
# Face Mesh needs; this margin gives it room without re-running MTCNN.
_CROP_MARGIN_FRAC = 0.3


# =============================================================================
# Letterbox normalisation
# =============================================================================

def letterbox(frame: np.ndarray, target_size: tuple[int, int] = LETTERBOX_TARGET_SIZE):
    """
    Resizes `frame` to fit inside `target_size` preserving aspect ratio, then
    pads with black borders to exactly `target_size` — never stretches.

    Why letterbox and not stretch: the corpus spans 11 distinct resolutions
    across landscape/portrait/square (see data/strata.csv). Stretching a
    720x1280 portrait frame into a square canvas would distort facial
    geometry (exactly the kind of artifact this pipeline is trying to
    detect, not introduce). Letterboxing also means every call into the
    TF-based detector sees the same input shape regardless of source
    resolution, collapsing what would otherwise be a per-resolution
    retrace/recompile cost to a single occurrence.

    Args:
        frame:       BGR or RGB numpy array, shape (H, W, 3).
        target_size: (width, height) of the fixed output canvas.

    Returns:
        (letterboxed, scale, (pad_x, pad_y)) — `scale` and the pad offsets
        let callers map a detected box in letterboxed-frame coordinates back
        to the original frame via: orig_x = (lb_x - pad_x) / scale.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def unletterbox_box(box, scale, pad):
    """Maps an (x, y, w, h) box in letterboxed coordinates back to the original frame."""
    x, y, w, h = box
    pad_x, pad_y = pad
    return ((x - pad_x) / scale, (y - pad_y) / scale, w / scale, h / scale)


# =============================================================================
# Face detector (sparse, MTCNN) — winning implementation from
# benchmark_face_detection.py wired in here
# =============================================================================

_detector = None


def _get_detector():
    """
    Lazy-loads the MTCNN detector. Which package this wraps is decided by
    benchmark_face_detection.py's real CPU-time comparison on this project's
    own corpus frames, not by a priori preference — see that script's
    docstring and requirements.txt for the outcome.
    """
    global _detector
    if _detector is None:
        _detector = _load_winning_mtcnn()
    return _detector


def _load_winning_mtcnn():
    """
    benchmark_face_detection.py compared facenet-pytorch's MTCNN against the
    TensorFlow-based `mtcnn` package on 10 real frames pulled from this
    project's own video corpus (not synthetic test images). facenet-pytorch
    won on median per-frame CPU time (0.228s vs 0.946s — ~4x faster) with
    mean IoU 0.76 on the 7/10 frames where both detected a face (the
    remaining 3 were frames neither detector found a face in — genuine
    agreement, not disagreement). facenet-pytorch also means one DL
    framework (torch) instead of two, since TensorFlow was carried solely
    for the `mtcnn` package. requirements.txt has been updated accordingly.

    keep_all=True (not False): returns every candidate face, not just the
    highest-confidence one. Bug found during Stage 4 rigidity calibration
    (see docs/PROJECT_CONTEXT.md §4) — with keep_all=False, on a frame with
    two visible faces the "highest confidence" face can differ from one
    sparse MTCNN tick to the next, so the tracked identity silently
    flickers between two different people. detect_face_bbox() now does
    IoU-based continuity selection against the previous tracked box instead
    of blindly taking the top candidate — see that function.
    """
    import torch
    from facenet_pytorch import MTCNN as FacenetMTCNN

    torch.set_num_threads(1)   # single-threaded CPU assumption used throughout this pipeline
    detector = FacenetMTCNN(keep_all=True, device="cpu", min_face_size=MTCNN_MIN_FACE_SIZE)

    def _detect(rgb_frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        boxes, probs = detector.detect(rgb_frame)
        if boxes is None:
            return []
        candidates = []
        for box, prob in zip(boxes, probs):
            if prob is None:
                continue
            x1, y1, x2, y2 = box
            candidates.append(((x1, y1, x2 - x1, y2 - y1), float(prob)))
        return candidates

    return _detect


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# =============================================================================
# Dense landmarker (MediaPipe Face Mesh, via the Tasks API)
# =============================================================================

_face_landmarker = None


def _get_face_landmarker():
    """
    Lazy-loads a singleton FaceLandmarker so it's created once per process,
    not once per analyze_clip() call.

    mediapipe==1.0.0 fully removed the old `solutions.face_mesh` API (which
    bundled its model file inside the package) in favour of the Tasks API,
    which requires an explicit downloaded model bundle — see
    config.FACE_LANDMARKER_MODEL_PATH. This was only caught by actually
    running analyze_clip() against a real video; the old API call raised
    ModuleNotFoundError at runtime despite looking correct on read.
    """
    global _face_landmarker
    if _face_landmarker is None:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def detect_face_bbox(
    frame_bgr: np.ndarray, prev_box: Optional[tuple[float, float, float, float]] = None,
) -> Optional[tuple[float, float, float, float]]:
    """
    Runs the sparse MTCNN stage on one frame (via letterbox normalisation)
    and returns a face box in ORIGINAL frame coordinates, or None if no
    face was found.

    Box selection (IoU-based continuity, not "just take the top candidate"):
    if `prev_box` is given and at least one candidate overlaps it by more
    than MTCNN_TRACK_IOU_THRESHOLD, the best-overlapping candidate is
    returned — this keeps the track on the SAME person across sparse MTCNN
    ticks. Otherwise (no previous box, or no candidate overlaps enough —
    the previous face left frame, or this is a fresh track) falls back to
    the highest-confidence candidate. Without this, on a multi-face frame
    "highest confidence" can flip between two different people from one
    tick to the next, corrupting anything that assumes continuous identity
    (rigidity, blink rate) — see _load_winning_mtcnn()'s docstring.
    """
    letterboxed, scale, pad = letterbox(frame_bgr)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    detector = _get_detector()
    candidates = detector(rgb)
    if not candidates:
        return None

    candidates = [(unletterbox_box(box, scale, pad), prob) for box, prob in candidates]

    if prev_box is not None:
        best_box, best_iou = max(
            ((box, _iou(box, prev_box)) for box, _ in candidates), key=lambda x: x[1],
        )
        if best_iou >= MTCNN_TRACK_IOU_THRESHOLD:
            return best_box

    return max(candidates, key=lambda c: c[1])[0]


# =============================================================================
# Shared tracking core
# =============================================================================
# Stage 4's four heuristics (blink rate, emotional rigidity, viseme mismatch,
# blending artifacts) all need per-frame data from the SAME sparse-MTCNN /
# dense-Face-Mesh pass Stage 2 already runs — landmark coordinates for the
# first three, face/surrounding pixel statistics for the fourth. Re-running
# detection per heuristic would multiply the already-measured MTCNN/Face-Mesh
# cost by 4x for no reason. _track_clip() runs the pass ONCE and returns a
# per-frame record any heuristic can consume; analyze_clip() (Stage 2's public
# entry point, unchanged in its return type) is now a thin wrapper over it.

@dataclass
class _TrackedFrame:
    t_s: float
    face_present: bool
    landmarks_xy: Optional[np.ndarray] = None    # (478, 2) float32, ORIGINAL frame pixel coords
    bbox: Optional[tuple] = None                 # (x, y, w, h), tight MTCNN box, original frame coords
    face_lap_var: Optional[float] = None         # Laplacian variance inside bbox (blending heuristic)
    surround_lap_var: Optional[float] = None     # Laplacian variance in the ring around bbox
    edge_discontinuity: Optional[float] = None   # boundary-band edge density minus interior/background


def _extract_audio_wav(video_path: str | Path) -> Optional[Path]:
    """
    Extracts the audio track to a temporary WAV file via a single ffmpeg
    subprocess call. Returns the temp file path, or None (not an exception)
    if the clip has no audio stream, ffmpeg isn't available, or extraction
    otherwise fails — compute_viseme_mismatch() treats None exactly like
    today's "no audio track" abstain.

    WHY THIS EXISTS (found timing d5e94745 — see docs/PROJECT_CONTEXT.md):
    librosa.load() has no fast native path for an MP4 container — PySoundFile
    can't read it, so librosa silently falls back to audioread's
    ffmpeg-subprocess backend on EVERY call. Calling ffmpeg here instead,
    once per clip, and handing compute_viseme_mismatch() a plain WAV file
    lets librosa's fast native soundfile path handle the actual load. Kept
    mono, native sample rate (no `-ar` resample) to match the semantics of
    the `librosa.load(path, sr=None, mono=True)` call this replaces.

    Caller is responsible for deleting the returned path when done with it —
    this only creates the file, it never cleans up after itself, since the
    caller (_track_clip()'s caller) is the one that knows when the clip's
    heuristics have all finished reading it.
    """
    video_path = Path(video_path)
    fd, wav_path_str = tempfile.mkstemp(suffix=".wav", prefix="watchdog_audio_")
    os.close(fd)
    wav_path = Path(wav_path_str)

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-acodec", "pcm_s16le", str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning(f"ffmpeg audio extraction failed for {video_path}: {e}")
        wav_path.unlink(missing_ok=True)
        return None

    # A bare WAV header with zero audio samples is 44 bytes — treat anything
    # at or below that (plus a nonzero ffmpeg exit code) as "no usable audio".
    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size <= 44:
        wav_path.unlink(missing_ok=True)
        return None

    return wav_path


def _track_clip(video_path: str | Path, extract_audio: bool = True):
    """
    Runs the sparse-MTCNN / dense-Face-Mesh pipeline over one video ONCE and
    returns `(frames, timing, duration_s, audio_wav_path)`, or None if the
    video can't be opened.

    `audio_wav_path` (new 2026-08-04, see `_extract_audio_wav()`) is a
    temporary WAV file with this clip's audio track, extracted once via
    ffmpeg, or None if the clip has no audio / extraction failed. Callers
    that don't need audio (analyze_clip(), Stage 2's diagnostic entry point)
    pass extract_audio=False to skip paying that cost at all — it exists
    purely for compute_viseme_mismatch() via run_heuristics(). Whoever
    receives a non-None path is responsible for unlinking it when finished.

    `frames` is a list of `_TrackedFrame`, one per Face-Mesh-sampled tick
    (FACE_MESH_SAMPLE_RATE_FPS), in order — this is the same temporal grid
    analyze_clip()'s `per_frame_face_present` was already built from.

    `timing` buckets:
      mtcnn_time_s / face_mesh_time_s / frame_extraction_time_s — Stage 2's
        original three buckets, unchanged in meaning.
      blending_pixel_time_s — the EXTRA per-frame Laplacian/Canny computation
        Stage 4's blending-artifacts heuristic needs. Kept separate rather
        than folded into face_mesh_time_s so its cost is independently
        attributable (see config.py's Stage 4 timing note) instead of
        silently inflating Stage 2's already-verified NFR-02 numbers.
      audio_extraction_time_s — the one ffmpeg subprocess call that produces
        audio_wav_path (see _extract_audio_wav()), only paid when
        extract_audio=True. Kept separate for the same attribution reason as
        blending_pixel_time_s — run_heuristics() folds it into
        viseme_mismatch.time_s, the heuristic that actually needs it.

    Design (see module docstring for the "why" of the two-tier approach):
      - MTCNN runs at MTCNN_SAMPLE_RATE_FPS (~1/s) to (re)locate the face.
        The resulting box is padded by _CROP_MARGIN_FRAC and held as the
        current "track" until the next MTCNN hit updates or clears it.
      - MediaPipe Face Mesh runs at FACE_MESH_SAMPLE_RATE_FPS (~10/s) on the
        current tracked crop, if any. A frame with no active track (MTCNN
        hasn't found a face yet, or lost it) is recorded as face-absent —
        this is the normal path for B-roll/product-shot frames, not an
        error.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"cv2.VideoCapture could not open: {video_path}")
        return None

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_native_frames / native_fps if native_fps > 0 else 0.0

    mtcnn_step = max(1, round(native_fps / MTCNN_SAMPLE_RATE_FPS))
    mesh_step = max(1, round(native_fps / FACE_MESH_SAMPLE_RATE_FPS))

    timing = dict(mtcnn_calls=0.0, mtcnn_time_s=0.0, face_mesh_calls=0.0,
                  face_mesh_time_s=0.0, frame_extraction_time_s=0.0,
                  blending_pixel_time_s=0.0, audio_extraction_time_s=0.0)

    audio_wav_path = None
    if extract_audio:
        t0 = time.perf_counter()
        audio_wav_path = _extract_audio_wav(video_path)
        timing["audio_extraction_time_s"] = time.perf_counter() - t0

    current_box = None   # (x, y, w, h) in original-frame coords, or None
    frames: list[_TrackedFrame] = []

    face_mesh = _get_face_landmarker()

    frame_idx = 0
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            timing["frame_extraction_time_s"] += time.perf_counter() - t0
            if not ret:
                break

            is_mtcnn_tick = frame_idx % mtcnn_step == 0
            is_mesh_tick = frame_idx % mesh_step == 0

            if is_mtcnn_tick:
                t0 = time.perf_counter()
                current_box = detect_face_bbox(frame, prev_box=current_box)
                timing["mtcnn_time_s"] += time.perf_counter() - t0
                timing["mtcnn_calls"] += 1

            if is_mesh_tick:
                t_s = frame_idx / native_fps
                tracked = _TrackedFrame(t_s=t_s, face_present=False)
                if current_box is not None:
                    padded = _padded_crop(frame, current_box, _CROP_MARGIN_FRAC)
                    if padded is not None:
                        crop, (ox, oy) = padded
                        if crop.size > 0:
                            t0 = time.perf_counter()
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                            result = face_mesh.detect(mp_image)
                            timing["face_mesh_time_s"] += time.perf_counter() - t0
                            timing["face_mesh_calls"] += 1

                            if result.face_landmarks:
                                lm = result.face_landmarks[0]
                                crop_h, crop_w = crop.shape[:2]
                                landmarks_xy = np.array(
                                    [(ox + p.x * crop_w, oy + p.y * crop_h) for p in lm],
                                    dtype=np.float32,
                                )
                                tracked.face_present = True
                                tracked.landmarks_xy = landmarks_xy
                                tracked.bbox = current_box

                                t0 = time.perf_counter()
                                metrics = _blending_pixel_metrics(frame, current_box, _CROP_MARGIN_FRAC)
                                timing["blending_pixel_time_s"] += time.perf_counter() - t0
                                if metrics is not None:
                                    tracked.face_lap_var = metrics["face_lap_var"]
                                    tracked.surround_lap_var = metrics["surround_lap_var"]
                                    tracked.edge_discontinuity = metrics["edge_discontinuity"]
                frames.append(tracked)

            frame_idx += 1
    finally:
        cap.release()

    return frames, timing, duration_s, audio_wav_path


def _blending_pixel_metrics(frame: np.ndarray, bbox, margin_frac: float) -> Optional[dict]:
    """
    Per-frame pixel statistics for the blending-artifacts heuristic (§3.12 A:
    "synthetic blending artifacts, specifically ... anomalies around the
    eyes"). Face-swap and avatar-generation methods commonly leave a
    sharpness or texture seam where the generated face meets the original
    background/body, so this compares:

      face_lap_var:      Laplacian variance (a standard sharpness proxy)
                          inside the tight face bbox.
      surround_lap_var:   Laplacian variance in the ring around the bbox
                          (the padded-crop area minus the bbox itself) —
                          real footage has continuous camera/lighting noise
                          across that boundary; a generated face pasted onto
                          real background often does not.
      edge_discontinuity: Canny edge density in a thin band straddling the
                          bbox boundary, minus the higher of the interior or
                          background edge density. A blending seam shows up
                          as an edge-density spike right at the boundary
                          that neither the interior nor the background
                          otherwise has.

    Returns None if the regions are too small to measure meaningfully.
    """
    x, y, w, h = [int(round(v)) for v in bbox]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return None

    mx, my = int(w * margin_frac), int(h * margin_frac)
    ox1, oy1 = max(0, x - mx), max(0, y - my)
    ox2, oy2 = min(frame.shape[1], x2 + mx), min(frame.shape[0], y2 + my)
    if ox2 <= ox1 or oy2 <= oy1:
        return None

    face_region = frame[y:y2, x:x2]
    outer_region = frame[oy1:oy2, ox1:ox2]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    gray_outer = cv2.cvtColor(outer_region, cv2.COLOR_BGR2GRAY)

    face_lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

    inner_y1, inner_x1 = y - oy1, x - ox1
    inner_y2, inner_x2 = inner_y1 + (y2 - y), inner_x1 + (x2 - x)

    surround_mask = np.ones(gray_outer.shape, dtype=bool)
    surround_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    if surround_mask.sum() < 16:
        return None
    lap_outer = cv2.Laplacian(gray_outer, cv2.CV_64F)
    surround_lap_var = float(lap_outer[surround_mask].var())

    edges = cv2.Canny(gray_outer, 50, 150)
    interior_density = float(edges[inner_y1:inner_y2, inner_x1:inner_x2].mean() / 255.0)
    background_density = float(edges[surround_mask].mean() / 255.0)

    band_px = max(2, int(0.08 * min(w, h)))
    by1, by2 = max(0, inner_y1 - band_px), min(gray_outer.shape[0], inner_y2 + band_px)
    bx1, bx2 = max(0, inner_x1 - band_px), min(gray_outer.shape[1], inner_x2 + band_px)
    band_mask = np.zeros(gray_outer.shape, dtype=bool)
    band_mask[by1:by2, bx1:bx2] = True
    band_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    if band_mask.sum() < 16:
        edge_discontinuity = None
    else:
        boundary_density = float(edges[band_mask].mean() / 255.0)
        edge_discontinuity = boundary_density - max(interior_density, background_density)

    return dict(face_lap_var=face_lap_var, surround_lap_var=surround_lap_var,
                edge_discontinuity=edge_discontinuity)


# =============================================================================
# Per-clip result (Stage 2 public entry point — unchanged contract)
# =============================================================================

@dataclass
class ClipFaceTrackResult:
    sample_path: str
    duration_s: float
    total_frames_sampled: int          # dense (Face Mesh) samples taken
    frames_with_face: int              # of those, how many had a landmark hit
    longest_contiguous_face_run_s: float
    mtcnn_calls: int
    mtcnn_time_s: float
    face_mesh_calls: int
    face_mesh_time_s: float
    frame_extraction_time_s: float
    per_frame_face_present: list[bool] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        return self.mtcnn_time_s + self.face_mesh_time_s + self.frame_extraction_time_s


def analyze_clip(video_path: str | Path) -> Optional[ClipFaceTrackResult]:
    """
    Runs the sparse-MTCNN / dense-Face-Mesh pipeline over one video and
    returns per-clip face-presence statistics + real measured stage timings.
    Thin wrapper over _track_clip() — see that function's docstring for the
    tracking design. Returns None if the video cannot be opened.

    Calls _track_clip() with extract_audio=False — this Stage 2 entry point
    never needs the audio track (viseme mismatch is a Stage 4 heuristic), so
    there's no reason to pay the ffmpeg extraction cost here and no wav file
    is produced to clean up.
    """
    tracked = _track_clip(video_path, extract_audio=False)
    if tracked is None:
        return None
    frames, timing, duration_s, _audio_wav_path = tracked

    per_frame_face_present = [f.face_present for f in frames]
    total_frames_sampled = len(per_frame_face_present)
    frames_with_face = sum(per_frame_face_present)
    longest_run_frames = _longest_true_run(per_frame_face_present)
    longest_run_s = longest_run_frames / FACE_MESH_SAMPLE_RATE_FPS

    return ClipFaceTrackResult(
        sample_path=str(video_path),
        duration_s=duration_s,
        total_frames_sampled=total_frames_sampled,
        frames_with_face=frames_with_face,
        longest_contiguous_face_run_s=longest_run_s,
        mtcnn_calls=timing["mtcnn_calls"],
        mtcnn_time_s=timing["mtcnn_time_s"],
        face_mesh_calls=timing["face_mesh_calls"],
        face_mesh_time_s=timing["face_mesh_time_s"],
        frame_extraction_time_s=timing["frame_extraction_time_s"],
        per_frame_face_present=per_frame_face_present,
    )


def _padded_crop(frame: np.ndarray, box, margin_frac: float) -> Optional[tuple[np.ndarray, tuple[int, int]]]:
    """
    Crops `frame` to `box` expanded by `margin_frac` on each side, clamped to
    frame bounds. Returns `(crop, (x1, y1))` — the origin is needed by
    callers that must map coordinates measured in the crop (e.g. Face Mesh
    landmarks, which MediaPipe returns normalised to whatever image it was
    given) back to original-frame pixel coordinates.
    """
    x, y, w, h = box
    mx, my = w * margin_frac, h * margin_frac
    x1 = max(0, int(x - mx))
    y1 = max(0, int(y - my))
    x2 = min(frame.shape[1], int(x + w + mx))
    y2 = min(frame.shape[0], int(y + h + my))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2], (x1, y1)


def _longest_true_run(bools: list[bool]) -> int:
    """Longest run of consecutive True values in a list — 0 if none."""
    longest = current = 0
    for b in bools:
        current = current + 1 if b else 0
        longest = max(longest, current)
    return longest


def _longest_face_present_segment(frames: "list[_TrackedFrame]") -> "list[_TrackedFrame]":
    """
    Same counting logic as _longest_true_run(), but returns the actual
    _TrackedFrame objects in the longest contiguous face_present run instead
    of just its length — this is the window every Stage 4 heuristic that
    needs temporal continuity (blink rate, rigidity, viseme) operates on.
    """
    best_start = best_len = 0
    cur_start = cur_len = 0
    for i, f in enumerate(frames):
        if f.face_present:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return frames[best_start:best_start + best_len]


# =============================================================================
# Stage 4 — the four heuristics (§3.12 A / FR-02 / FR-03)
# =============================================================================
# Every heuristic returns a HeuristicResult with score=None (not a default
# number) when the clip can't actually support that measurement — no face,
# no audio track, or too little contiguous face-present run for a rate
# estimate. This mirrors nlp_engine.analyze_text()'s conservative UNCERTAIN
# vote and exists for the same NFR-01 reason: a fabricated blink rate is
# exactly the kind of thing that ends up stated as fact in an auto-generated
# legal complaint (§3.13). The minimum-run gates are config.py constants set
# from the real per-clip face-coverage diagnostic (2026-08-03), not guessed —
# see config.py's Stage 4 section and docs/PROJECT_CONTEXT.md §4.

# MediaPipe Face Mesh's 478-point topology, mapped to the classic 6-point EAR
# eye contour (Soukupová & Čech, 2016): p1/p4 = corners, p2/p3 = upper lid,
# p5/p6 = lower lid. These specific indices are the commonly-used mapping for
# MediaPipe's landmark set in EAR-based blink detection implementations.
RIGHT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)

# Simplified 4-point Mouth Aspect Ratio: inner-lip vertical opening over
# mouth-corner horizontal width.
MOUTH_MAR_IDX = dict(upper=13, lower=14, left=61, right=291)


@dataclass
class HeuristicResult:
    score: Optional[float]          # [0,1], higher = more synthetic-like; None = abstained
    raw: dict                       # forensic detail — a legal complaint must state what was observed
    abstain_reason: Optional[str]
    time_s: float


def _ear(landmarks_xy: np.ndarray, idx: tuple[int, int, int, int, int, int]) -> float:
    """Eye Aspect Ratio (Soukupová & Čech, 2016) from 6 eye-contour points."""
    p1, p2, p3, p4, p5, p6 = (landmarks_xy[i] for i in idx)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def _mar(landmarks_xy: np.ndarray) -> float:
    """Mouth Aspect Ratio: inner-lip vertical opening / mouth-corner width."""
    upper = landmarks_xy[MOUTH_MAR_IDX["upper"]]
    lower = landmarks_xy[MOUTH_MAR_IDX["lower"]]
    left = landmarks_xy[MOUTH_MAR_IDX["left"]]
    right = landmarks_xy[MOUTH_MAR_IDX["right"]]
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return float(np.linalg.norm(upper - lower) / horizontal)


def _count_blinks(ear_series: list[float], threshold: float, min_consecutive: int) -> int:
    """
    Counts EAR threshold-crossings: a run of >= min_consecutive frames below
    `threshold` followed by recovery above it is one blink. A run still
    below threshold at the end of the series (blink mid-closure when the
    tracked window ends) still counts — the eye was demonstrably closed for
    long enough to be a blink, not noise.
    """
    blinks = 0
    closed_run = 0
    for ear in ear_series:
        if ear < threshold:
            closed_run += 1
        else:
            if closed_run >= min_consecutive:
                blinks += 1
            closed_run = 0
    if closed_run >= min_consecutive:
        blinks += 1
    return blinks


def compute_blink_rate(frames: "list[_TrackedFrame]") -> HeuristicResult:
    """
    1. BLINK RATE (§3.12 A). EAR per frame over the longest contiguous
    face-present run; blinks counted via threshold-crossing; rate compared
    against config.HUMAN_BLINK_RATE_MIN/_MAX (12-25/min). Abstains
    (score=None) below config.MIN_FACE_RUN_FOR_BLINK_S of contiguous
    tracking — see that constant's docstring in config.py for why this
    excludes essentially all synthetic ad footage in the current corpus.
    """
    t0 = time.perf_counter()
    segment = _longest_face_present_segment(frames)
    run_s = len(segment) / FACE_MESH_SAMPLE_RATE_FPS

    if run_s < MIN_FACE_RUN_FOR_BLINK_S:
        return HeuristicResult(
            score=None, raw={"longest_face_run_s": round(run_s, 2)},
            abstain_reason=(f"longest contiguous face run ({run_s:.1f}s) is below the "
                             f"{MIN_FACE_RUN_FOR_BLINK_S}s minimum needed to measure a blink rate"),
            time_s=time.perf_counter() - t0,
        )

    ear_series = [
        (_ear(f.landmarks_xy, LEFT_EYE_EAR_IDX) + _ear(f.landmarks_xy, RIGHT_EYE_EAR_IDX)) / 2.0
        for f in segment
    ]
    n_blinks = _count_blinks(ear_series, EAR_BLINK_THRESHOLD, EAR_MIN_CONSECUTIVE_FRAMES)
    blink_rate_per_min = n_blinks / (run_s / 60.0)

    if blink_rate_per_min < HUMAN_BLINK_RATE_MIN:
        deviation = (HUMAN_BLINK_RATE_MIN - blink_rate_per_min) / HUMAN_BLINK_RATE_MIN
    elif blink_rate_per_min > HUMAN_BLINK_RATE_MAX:
        deviation = (blink_rate_per_min - HUMAN_BLINK_RATE_MAX) / HUMAN_BLINK_RATE_MAX
    else:
        deviation = 0.0
    score = max(0.0, min(1.0, deviation))

    return HeuristicResult(
        score=score,
        raw={"blink_rate_per_min": round(blink_rate_per_min, 2), "n_blinks": n_blinks,
             "window_s": round(run_s, 2)},
        abstain_reason=None,
        time_s=time.perf_counter() - t0,
    )


def compute_emotional_rigidity(frames: "list[_TrackedFrame]") -> HeuristicResult:
    """
    2. EMOTIONAL RIGIDITY (§3.12 A: "unnatural emotional rigidity or lack of
    subtle micro-expressions across consecutive frames"). Variance of
    per-frame landmark displacement between consecutive tracked frames,
    normalised by face bbox size (scale-invariant across the corpus's mixed
    resolutions/orientations — see data/strata.csv), over the longest
    contiguous face run. Low variance -> rigid -> scores toward 1.0.
    Abstains below config.MIN_FACE_RUN_FOR_RIGIDITY_S.
    """
    t0 = time.perf_counter()
    segment = _longest_face_present_segment(frames)
    run_s = len(segment) / FACE_MESH_SAMPLE_RATE_FPS

    if run_s < MIN_FACE_RUN_FOR_RIGIDITY_S:
        return HeuristicResult(
            score=None, raw={"longest_face_run_s": round(run_s, 2)},
            abstain_reason=(f"longest contiguous face run ({run_s:.1f}s) is below the "
                             f"{MIN_FACE_RUN_FOR_RIGIDITY_S}s minimum needed for a displacement-variance estimate"),
            time_s=time.perf_counter() - t0,
        )

    displacements = []
    for prev, cur in zip(segment, segment[1:]):
        bbox_scale = max(prev.bbox[2], prev.bbox[3])
        if bbox_scale <= 0:
            continue
        per_landmark_d = np.linalg.norm(cur.landmarks_xy - prev.landmarks_xy, axis=1)
        displacements.append(float(per_landmark_d.mean()) / bbox_scale)

    if len(displacements) < 2:
        return HeuristicResult(
            score=None, raw={}, abstain_reason="too few consecutive frame pairs for a variance estimate",
            time_s=time.perf_counter() - t0,
        )

    variance = float(np.var(displacements))
    score = max(0.0, min(1.0, 1.0 - (variance / EMOTIONAL_RIGIDITY_THRESHOLD)))

    return HeuristicResult(
        score=score,
        raw={"displacement_variance": variance, "window_s": round(run_s, 2), "n_frame_pairs": len(displacements)},
        abstain_reason=None,
        time_s=time.perf_counter() - t0,
    )


def compute_viseme_mismatch(
    frames: "list[_TrackedFrame]", audio_path: Optional[str | Path],
) -> HeuristicResult:
    """
    3. VISEME-PHONEME MISMATCH (§3.12 A / FR-03). Mouth Aspect Ratio per
    frame correlated against the audio RMS-energy envelope (librosa) over
    the longest contiguous face run. Genuine speech: strong positive
    correlation; poor lip-sync: weak/absent correlation. Returns None
    cleanly (not an exception) when there is no audio track, or below
    config.MIN_FACE_RUN_FOR_VISEME_S of contiguous tracking.

    `audio_path` is normally the WAV file _extract_audio_wav() produced
    during _track_clip() (see run_heuristics()), not the original video —
    librosa has no fast native path for an MP4 container, so loading the
    pre-extracted WAV instead of the source video lets librosa's fast
    native path handle it rather than silently falling back to audioread on
    every call (see docs/PROJECT_CONTEXT.md, "viseme audio-decode cost").
    None (no audio stream, or extraction failed) short-circuits straight to
    an abstain without attempting a load — this is also what any
    audio-free file passed here directly (e.g. in tests) naturally falls
    through to via the load failing.
    """
    t0 = time.perf_counter()
    segment = _longest_face_present_segment(frames)
    run_s = len(segment) / FACE_MESH_SAMPLE_RATE_FPS

    if run_s < MIN_FACE_RUN_FOR_VISEME_S:
        return HeuristicResult(
            score=None, raw={"longest_face_run_s": round(run_s, 2)},
            abstain_reason=(f"longest contiguous face run ({run_s:.1f}s) is below the "
                             f"{MIN_FACE_RUN_FOR_VISEME_S}s minimum needed for a mouth/audio correlation"),
            time_s=time.perf_counter() - t0,
        )

    if audio_path is None:
        return HeuristicResult(
            score=None, raw={}, abstain_reason="no audio track", time_s=time.perf_counter() - t0,
        )

    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as e:
        return HeuristicResult(
            score=None, raw={}, abstain_reason=f"audio load failed (likely no audio track): {e}",
            time_s=time.perf_counter() - t0,
        )
    if y.size == 0:
        return HeuristicResult(
            score=None, raw={}, abstain_reason="no audio track", time_s=time.perf_counter() - t0,
        )

    mar_series = np.array([_mar(f.landmarks_xy) for f in segment])

    hop = max(1, int(sr * 0.05))
    energy = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    energy_times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop)
    visual_times = [f.t_s for f in segment]
    audio_at_visual = np.interp(visual_times, energy_times, energy)

    if np.std(mar_series) == 0 or np.std(audio_at_visual) == 0:
        return HeuristicResult(
            score=None, raw={}, abstain_reason="no variation in mouth movement or audio energy over this window",
            time_s=time.perf_counter() - t0,
        )

    corr = float(np.corrcoef(mar_series, audio_at_visual)[0, 1])
    if np.isnan(corr):
        return HeuristicResult(
            score=None, raw={}, abstain_reason="correlation undefined (degenerate signal)",
            time_s=time.perf_counter() - t0,
        )

    if corr >= VISEME_CORRELATION_THRESHOLD:
        score = 0.0
    else:
        score = max(0.0, min(1.0, (VISEME_CORRELATION_THRESHOLD - corr) / (VISEME_CORRELATION_THRESHOLD + 1.0)))

    return HeuristicResult(
        score=score,
        raw={"mouth_audio_correlation": round(corr, 3), "window_s": round(run_s, 2)},
        abstain_reason=None,
        time_s=time.perf_counter() - t0,
    )


def compute_blending_artifacts(frames: "list[_TrackedFrame]") -> HeuristicResult:
    """
    4. BLENDING ARTIFACTS (§3.12 A: "synthetic blending artifacts,
    specifically ... anomalies around the eyes"). Laplacian-variance
    sharpness ratio between the face region and its surrounding ring, plus
    Canny edge-density discontinuity at the face boundary — both computed
    per-frame during _track_clip() (see _blending_pixel_metrics) and
    aggregated here. Unlike the other three heuristics this doesn't need
    temporal continuity, just enough independent samples
    (config.MIN_FRAMES_FOR_BLENDING) — a face that only appears in scattered
    frames can still be checked for a blending seam in each one.
    """
    t0 = time.perf_counter()
    usable = [f for f in frames if f.face_lap_var is not None and f.surround_lap_var is not None]

    if len(usable) < MIN_FRAMES_FOR_BLENDING:
        return HeuristicResult(
            score=None, raw={"n_usable_frames": len(usable)},
            abstain_reason=(f"only {len(usable)} frame(s) with valid face/surround pixel stats, "
                             f"below the {MIN_FRAMES_FOR_BLENDING}-frame minimum"),
            time_s=time.perf_counter() - t0,
        )

    ratios = [f.face_lap_var / f.surround_lap_var for f in usable if f.surround_lap_var > 0]
    discontinuities = [f.edge_discontinuity for f in usable if f.edge_discontinuity is not None]

    if not ratios:
        return HeuristicResult(
            score=None, raw={}, abstain_reason="surrounding region had zero variance in every usable frame",
            time_s=time.perf_counter() - t0,
        )

    mean_ratio = float(np.mean(ratios))
    mean_discontinuity = float(np.mean(discontinuities)) if discontinuities else 0.0

    ratio_score = min(1.0, abs(mean_ratio - 1.0) / BLENDING_SHARPNESS_RATIO_DEVIATION_THRESHOLD)
    edge_score = min(1.0, max(0.0, mean_discontinuity) / BLENDING_EDGE_DISCONTINUITY_THRESHOLD)
    score = max(ratio_score, edge_score)   # either signal alone can indicate a seam

    return HeuristicResult(
        score=score,
        raw={"mean_face_surround_lap_ratio": round(mean_ratio, 3),
             "mean_edge_discontinuity": round(mean_discontinuity, 4), "n_frames": len(usable)},
        abstain_reason=None,
        time_s=time.perf_counter() - t0,
    )


# =============================================================================
# Orchestrator — runs the shared tracking pass once, then all four heuristics
# =============================================================================

@dataclass
class ClipHeuristics:
    sample_path: str
    blink_rate: HeuristicResult
    emotional_rigidity: HeuristicResult
    viseme_mismatch: HeuristicResult
    blending_artifacts: HeuristicResult
    tracking_time_s: float   # Stage 2 cost (mtcnn+mesh+extract) — context, not a heuristic cost

    @property
    def total_time_s(self) -> float:
        return (self.tracking_time_s + self.blink_rate.time_s + self.emotional_rigidity.time_s
                + self.viseme_mismatch.time_s + self.blending_artifacts.time_s)


def run_heuristics(video_path: str | Path) -> Optional[ClipHeuristics]:
    """
    Runs _track_clip() once, then all four Stage 4 heuristics against the
    same tracked frames, with each heuristic's cost timed independently
    (per this project's 2026-08-03 amendment: report per-heuristic cost from
    the start, not just total pipeline time, so a budget overrun is
    attributable to a specific stage rather than just observed). Returns
    None if the video cannot be opened.

    Note on blending-artifacts' reported time_s: the real per-frame pixel
    cost (Laplacian/Canny) is paid during _track_clip(), not in
    compute_blending_artifacts() itself (which just aggregates numbers
    already computed) — that cost is folded into blending_artifacts.time_s
    here so it's attributed to the heuristic that actually needs it, not
    hidden inside tracking_time_s. Same pattern for viseme_mismatch: the one
    ffmpeg audio-extraction call (_extract_audio_wav(), via
    _track_clip(extract_audio=True)) is folded into viseme_mismatch.time_s
    since that's the only heuristic that needs the audio track.
    """
    tracked = _track_clip(video_path)
    if tracked is None:
        return None
    frames, timing, duration_s, audio_wav_path = tracked
    tracking_time_s = timing["mtcnn_time_s"] + timing["face_mesh_time_s"] + timing["frame_extraction_time_s"]

    try:
        blink = compute_blink_rate(frames)
        rigidity = compute_emotional_rigidity(frames)
        viseme = compute_viseme_mismatch(frames, audio_wav_path)
        viseme.time_s += timing["audio_extraction_time_s"]
        blending = compute_blending_artifacts(frames)
        blending.time_s += timing["blending_pixel_time_s"]
    finally:
        if audio_wav_path is not None:
            Path(audio_wav_path).unlink(missing_ok=True)

    return ClipHeuristics(
        sample_path=str(video_path),
        blink_rate=blink,
        emotional_rigidity=rigidity,
        viseme_mismatch=viseme,
        blending_artifacts=blending,
        tracking_time_s=tracking_time_s,
    )
