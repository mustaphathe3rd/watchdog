# =============================================================================
# watchdog/preprocessor.py  —  Data Preprocessing Pipeline
# =============================================================================
# PURPOSE:
#   Implements Table 3.5 of the project literature — the 4-step preprocessing
#   pipeline that normalises scraped media before it enters the detection modules.
#
# TABLE 3.5 (Data Preprocessing Pipeline):
#   Step 1: Frame Extraction (OpenCV)   — Video → discrete 2D image tensors
#   Step 2: Facial Alignment (MTCNN)    — Isolate facial bounds, discard background
#   Step 3: Tokenization + Stop-Word    — Raw text → uniform sequences for NLP
#   Step 4: Noise Reduction (Gaussian)  — Suppress WhatsApp/Facebook compression
#
# §3.10: "A rigorous preprocessing pipeline was established to normalize the
# diverse range of media scraped from the internet, isolating the exact features
# required for anomaly detection while stripping out background noise."
# =============================================================================

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1 + 4:  Frame Extraction + Gaussian Blur (Video preprocessing)
# =============================================================================

def extract_frames(
    video_path: str | Path,
    sample_fps: int = 1,
    gaussian_kernel: Tuple[int, int] = (5, 5)
) -> List[np.ndarray]:
    """
    Extracts frames from a video file at the specified sample rate and applies
    Gaussian blur to suppress compression artifacts.

    Step 1 — Frame Extraction (§3.10, Table 3.5):
        "Deconstructs continuous video into discrete 2D image tensors for
         granular analysis."
        Uses cv2.VideoCapture as specified in §3.12 A:
        "The custom Python functions utilized cv2.VideoCapture to parse
         video files frame-by-frame."

    Step 4 — Noise Reduction / Gaussian Blur (§3.10, Table 3.5):
        "Suppresses compression artifacts (common in WhatsApp/Facebook videos)
         that could trigger false positives."
        Why this matters: Nigerian social media content is heavily re-shared
        via WhatsApp (which applies lossy compression), creating JPEG artifacts
        that naive CV models can misclassify as deepfake blending artifacts.
        The Gaussian blur removes this noise before facial analysis.

    Args:
        video_path:       Path to the MP4/AVI video file.
        sample_fps:       Frames to extract per second of video.
                          Default 1 fps — for a 30s ad = 30 frames to analyse.
                          Practical for CPU-only inference (§1.7 Limitations).
        gaussian_kernel:  Kernel size for cv2.GaussianBlur.
                          (5, 5) from config.GAUSSIAN_BLUR_KERNEL.

    Returns:
        List of numpy arrays, each shape (H, W, 3) BGR, one per sampled frame.
        Returns empty list if video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"cv2.VideoCapture could not open: {video_path}")
        return []

    # Get native video FPS to calculate step interval
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0   # Assume 25 fps if metadata unavailable
        logger.warning(f"FPS metadata missing for {video_path.name}, assuming 25 fps")

    # How many native frames to skip between each sample
    # e.g. native_fps=25, sample_fps=1 → skip every 25 frames (1 per second)
    step = max(1, int(native_fps / sample_fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break   # End of video

        # Only keep frames at the sample interval
        if frame_idx % step == 0:
            # Step 4: Apply Gaussian blur to suppress compression noise
            blurred = cv2.GaussianBlur(frame, gaussian_kernel, 0)
            frames.append(blurred)

        frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {len(frames)} frames from '{video_path.name}' "
        f"(native {native_fps:.1f} fps, sampled at {sample_fps} fps)"
    )
    return frames


# =============================================================================
# STEP 2: Facial Alignment (MTCNN — lazy import to keep startup fast)
# =============================================================================

def extract_faces(
    frames: List[np.ndarray],
    min_face_size: int = 40
) -> List[Tuple[np.ndarray, dict]]:
    """
    Runs MTCNN on each frame to detect, align, and extract facial regions.

    Step 2 — Facial Alignment (§3.10, Table 3.5):
        "Isolates human/avatar facial boundaries, discarding irrelevant
         background pixels to focus on micro-expressions."

    FR-02 (Functional Requirement):
        "The system shall employ MTCNN to automatically identify, bound,
         and extract facial features from ingested video frames."

    §3.12 A — Artifact Detection:
        "A lightweight classification model scrutinized the extracted MTCNN
         facial bounds for synthetic blending artifacts..."

    Why MTCNN?
        Multi-Task Cascaded CNN detects faces in 3 stages (P-Net, R-Net, O-Net),
        returning both a bounding box and 5 facial landmark points (eyes, nose,
        mouth corners). The landmark points are essential for the viseme-phoneme
        sync analysis in the CV engine (FR-03).

    Which package?
        FR-02 specifies the MTCNN algorithm, not a specific package.
        benchmark_face_detection.py compared the TensorFlow-based `mtcnn`
        package against facenet-pytorch's MTCNN on 10 real frames from this
        project's own corpus; facenet-pytorch won on median per-frame CPU
        time (~4x faster) with good box agreement (mean IoU 0.76) — see that
        script and watchdog/cv_engine.py._load_winning_mtcnn() for the full
        result. Using it here too avoids carrying TensorFlow in this codebase
        at all, alongside the torch stack already required for the
        classifiers.

    Args:
        frames:         List of BGR numpy arrays from extract_frames().
        min_face_size:  Minimum face size in pixels to detect.
                        40px is sufficient for standard-resolution ad content.

    Returns:
        List of (face_crop, detection_dict) tuples.
        face_crop:      Cropped face region as numpy array.
        detection_dict: {'box': [x, y, w, h], 'confidence': float, 'keypoints':
                        {'left_eye', 'right_eye', 'nose', 'mouth_left',
                        'mouth_right'}} for sync analysis — same shape as the
                        original `mtcnn` package's per-detection dict.
        Frames with no detected face are silently skipped.
    """
    # Lazy import: model init takes a moment; only load when needed
    try:
        import torch
        from facenet_pytorch import MTCNN
    except ImportError:
        logger.error(
            "facenet-pytorch not installed. Run: pip install facenet-pytorch\n"
            "This is required for FR-02 facial extraction."
        )
        return []

    torch.set_num_threads(1)
    detector = MTCNN(keep_all=True, device="cpu", min_face_size=min_face_size)
    results = []

    for frame_num, frame in enumerate(frames):
        # MTCNN expects RGB; OpenCV loads BGR — convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            boxes, probs, landmarks = detector.detect(rgb_frame, landmarks=True)
        except Exception as e:
            logger.warning(f"MTCNN failed on frame {frame_num}: {e}")
            continue

        if boxes is None:
            continue

        for box, prob, lm in zip(boxes, probs, landmarks):
            if prob is None or prob < 0.90:
                # Discard low-confidence face detections to reduce noise
                continue

            x1, y1, x2, y2 = box
            # Clamp coordinates to frame bounds (handles edge-detected faces)
            x, y = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

            face_crop = frame[y:y2, x:x2]
            if face_crop.size == 0:
                continue

            detection = {
                "box": [x, y, x2 - x, y2 - y],
                "confidence": float(prob),
                "keypoints": {
                    "left_eye": tuple(lm[0]),
                    "right_eye": tuple(lm[1]),
                    "nose": tuple(lm[2]),
                    "mouth_left": tuple(lm[3]),
                    "mouth_right": tuple(lm[4]),
                },
            }
            results.append((face_crop, detection))

    logger.info(f"MTCNN: detected {len(results)} face regions across {len(frames)} frames")
    return results


# =============================================================================
# STEP 3: Tokenisation + Stop-Word Removal (Text preprocessing)
# =============================================================================

# Module-level cache — the spaCy pipeline is loaded once, not on every
# call. Same lazy-singleton pattern as nlp_engine._load_perplexity_model()/
# _load_classifier() and ip_similarity._get_embedder(). Before this fix,
# preprocess_text() called spacy.load("en_core_web_sm") fresh on every
# single invocation — the only model loader in this codebase that didn't
# cache. Found during the 2026-08-13 NFR-02 profiling pass
# (docs/PROJECT_CONTEXT.md §4): measured at ~0.35-0.76s repeated on every
# call, real and avoidable, though too small alone to explain that pass's
# 16.9s "warm" anomaly (which turned out to be a post-restart cold load
# of the DistilBERT classifier, not this).
_NLP = None


def _load_spacy_model():
    """
    Lazily loads spaCy's en_core_web_sm pipeline, caching it at module
    level. Called automatically by preprocess_text() — callers never need
    to invoke this directly.

    Raises ImportError if spaCy isn't installed, or OSError if
    en_core_web_sm hasn't been downloaded — preprocess_text() catches both
    and falls back to _fallback_preprocess(), unchanged from before this
    fix. Neither failure is cached: a missing model can start succeeding
    later (e.g. `python -m spacy download en_core_web_sm` run mid-session)
    without needing a process restart to pick it up.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    import spacy
    _NLP = spacy.load("en_core_web_sm")
    return _NLP


def preprocess_text(text: str, language: str = "en") -> dict:
    """
    Tokenises and cleans raw text for the NLP detection engine.

    Step 3 — Tokenization & Stop-Word Removal (§3.10, Table 3.5):
        "Standardizes raw text into uniform sequences recognizable by the
         NLP embeddings model."

    FR-04 (Functional Requirement):
        "The system shall evaluate text for perplexity and burstiness metrics
         to determine AI-authorship probability."

    §3.12 B — Textual and Plagiarism Detection Engine:
        The NLP module "analyzed the perplexity (the randomness of word choice)
        and burstiness (the variation in sentence length)..."

    The returned dict provides both the cleaned string (for perplexity scoring)
    and the sentence list (for burstiness calculation) in one pass.

    Args:
        text:      Raw text string (marketing copy, caption, article, etc.)
        language:  SpaCy language model identifier. 'en' uses en_core_web_sm,
                   which handles formal Nigerian English adequately for
                   tokenisation. NLP classification itself uses the fine-tuned
                   DistilBERT model, not SpaCy.

    Returns:
        dict with keys:
            'cleaned':      Lowercased text with stop words removed (for NLP model input).
            'sentences':    List of raw sentences (for burstiness calculation).
            'tokens':       List of meaningful tokens.
            'word_count':   Total word count of original text.
            'sentence_count': Number of sentences detected.
    """
    try:
        nlp = _load_spacy_model()
    except ImportError:
        logger.error("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
        return _fallback_preprocess(text)
    except OSError:
        logger.warning(
            "SpaCy model 'en_core_web_sm' not found.\n"
            "Run: python -m spacy download en_core_web_sm\n"
            "Falling back to basic preprocessing."
        )
        return _fallback_preprocess(text)

    doc = nlp(text)

    # Extract sentences (needed for burstiness metric)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Extract meaningful tokens: no stop words, no punctuation, no whitespace
    # Keep case for IP similarity matrix (sentence-transformers is case-sensitive)
    tokens_raw    = [token.text for token in doc
                     if not token.is_stop and not token.is_punct and not token.is_space]

    # Lowercase cleaned version for DistilBERT (tokenizer handles subwords)
    tokens_lower  = [t.lower() for t in tokens_raw]
    cleaned_text  = " ".join(tokens_lower)

    return {
        "cleaned":        cleaned_text,
        "sentences":      sentences,
        "tokens":         tokens_raw,
        "word_count":     len(list(doc)),
        "sentence_count": len(sentences),
        "original":       text,       # Keep original for IP similarity check
    }


def _fallback_preprocess(text: str) -> dict:
    """
    Basic whitespace tokenisation — used only if spaCy is not installed.
    Functional but less accurate than the spaCy pipeline.
    """
    import re

    stripped = text.strip()

    # BUG FIX (discovered via tests/test_preprocessor.py, Day 2):
    # re.split(pattern, "") returns [''] — a list containing ONE empty
    # string — not an empty list. Left unguarded, empty/whitespace-only
    # scraped content (e.g. a video-only page with no article body) was
    # being counted as "1 sentence", which would silently corrupt the
    # burstiness metric (§3.12 B) by injecting a phantom zero-length
    # sentence into the variance calculation.
    if not stripped:
        return {
            "cleaned": "", "sentences": [], "tokens": [],
            "word_count": 0, "sentence_count": 0, "original": text,
        }

    # Split into sentences on common punctuation
    sentences = re.split(r'(?<=[.!?])\s+', stripped)
    # Remove numbers, punctuation, lowercase
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens  = [t for t in cleaned.split() if len(t) > 2]
    return {
        "cleaned":        " ".join(tokens),
        "sentences":      sentences,
        "tokens":         tokens,
        "word_count":     len(text.split()),
        "sentence_count": len(sentences),
        "original":       text,
    }
