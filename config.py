# =============================================================================
# config.py  —  Central Configuration for Naija AI Watchdog
# =============================================================================
# All thresholds, paths, and constants are defined here in one place.
# Every module imports from config.py — no hardcoded magic numbers anywhere.
#
# §3.5.2 System Outputs: "The system is programmed with a high threshold (85%)
# to trigger further action, thereby mitigating the risk of erroneously
# flagging human creators." — this 0.85 value lives here and only here.
# =============================================================================

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Load .env into the process environment ────────────────────────────────────
# BUG FIX (found in production — a live cookie-auth test showed
# YTDLP_COOKIES_FROM_BROWSER wasn't being picked up despite being set in
# .env). Root cause: nothing in this project ever called load_dotenv().
# python-dotenv was listed in requirements.txt but never actually invoked,
# so every os.getenv() call below only ever saw real shell-level environment
# variables — .env's contents were silently ignored from Day 1 onward.
# This went unnoticed because DATABASE_URL's hardcoded fallback below
# happens to exactly match what most people put in .env, so it *looked*
# like .env was being read when it was actually just coincidentally
# falling back to the same hardcoded default the whole time.
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")


# ── Detection Threshold (§3.5.2 / §3.13) ─────────────────────────────────────
# The Synthetic Probability Score (Psynth) must EXCEED this value to trigger
# the Compliance Mapping and Auto-Complaint modules.
# Rationale: VSD principle (§2.2.3) — protect innocent human creators from
# false accusations. High threshold = low false-positive rate (NFR-01, NFR-03).
PSYNTH_THRESHOLD = 0.85


# ── Fusion Weights (§3.11 Mathematical Framework) ────────────────────────────
# PROVISIONAL — UNVALIDATED. Guessed before any real fusion evidence existed:
# the CV and NLP heuristics it would weight between were both calibrated (Day
# 3 for NLP, 2026-08-03 for CV — see docs/PROJECT_CONTEXT.md §4) only AFTER
# this value was set, and the fusion module itself (§3.11, combining a CV
# score and an NLP score into one Psynth) has not been built yet — nothing
# has ever exercised this weight against real per-modality scores. Same
# category as the Stage 4 CV thresholds below: flagged rather than treated as
# settled. Do not tune this to make a future fusion result look better
# without first checking whether the two modalities it's weighting between
# have any discriminative power at all (Day 3's NLP finding and the Stage 4
# CV calibration both suggest that's an open question, not a given).
FUSION_ALPHA = 0.6


# ── Data Directories (§3.9 Dataset Description) ───────────────────────────────
DATA_DIR          = BASE_DIR / "data"
RAW_VIDEO_DIR     = DATA_DIR / "raw" / "video"
RAW_TEXT_DIR      = DATA_DIR / "raw" / "text"
PROCESSED_DIR     = DATA_DIR / "processed"
PROTECTED_IP_DIR  = DATA_DIR / "protected_ip"
MANIFEST_PATH     = DATA_DIR / "manifest.csv"


# ── Model Checkpoints ─────────────────────────────────────────────────────────
MODELS_DIR        = BASE_DIR / "models"
NLP_MODEL_PATH    = MODELS_DIR / "nlp_watchdog_v1.pt"
CV_MODEL_PATH     = MODELS_DIR / "cv_watchdog_v1.pt"
IP_EMBEDDINGS_PATH = MODELS_DIR / "ip_embeddings.pkl"

# CV_MODEL_PATH's companion metrics file — the FF++ held-out numbers
# (accuracy/precision/recall/F1, frame- and video-level) that
# notebooks/train_cv_classifier_colab.ipynb measures BEFORE saving the
# checkpoint (its explicit >=90%-accuracy gate). Saved alongside the
# checkpoint on Drive and downloaded together. evaluate_cv_transfer.py reads
# this to print the FF++-vs-Nigerian-corpus transfer gap automatically
# rather than a hardcoded number that could go stale if the notebook is
# re-run with different data/hyperparameters — if this file is missing,
# evaluate_cv_transfer.py still runs, it just skips the gap comparison.
CV_MODEL_FF_METRICS_PATH = MODELS_DIR / "cv_watchdog_v1_ff_metrics.json"

# MediaPipe's legacy `solutions.face_mesh` API (static_image_mode, bundled model)
# was fully removed by mediapipe==1.0.0 in favour of the Tasks API, which
# requires an explicit downloaded model bundle rather than one bundled in the
# package. Downloaded once from Google's model zoo (see watchdog/cv_engine.py):
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"


# ── Reports Output Directory (§3.13 Document Generation) ─────────────────────
REPORTS_DIR = BASE_DIR / "reports"


# ── NLP Settings (§3.12 B — Textual and Plagiarism Detection Engine) ──────────
NLP_BASE_MODEL = "distilbert-base-uncased"

# PROVISIONAL — UNVALIDATED. Guessed before any real data existed. The Day 3
# evaluation (14 real labelled TEXTUAL_CORPUS samples — see
# docs/PROJECT_CONTEXT.md §4, results/nlp_heuristic_eval.csv) found
# perplexity runs BACKWARDS on this corpus relative to the assumption behind
# this threshold: synthetic mean 60.9 vs authentic mean 30.3 — AI text scored
# MORE surprising to DistilGPT-2, not less, plausibly because the model is
# trained on Western English and the synthetic samples deliberately use
# Nigerian phrasing ("wahala", "Owambe", "Detty December") that's
# out-of-distribution regardless of author, while real Nigerian tech
# journalism sits closer to the model's training distribution. That's a
# genuine finding for the project's own thesis (§1.1/§2.3.1), not just noise
# — but it also means this threshold's *direction* (lower perplexity = more
# AI-like) is actively contradicted by the only real data run against it so
# far. Not retuned per CLAUDE.md rule 7 / the Day 3 decision to let the Day
# 8-9 fine-tuned classifier learn the boundary rather than fit this number to
# current data.
PERPLEXITY_THRESHOLD = 35.0

BURSTINESS_THRESHOLD = 0.3

# VALIDATED (2026-08-04) against the real 34-work protected corpus —
# see validate_ip_similarity_threshold.py and docs/PROJECT_CONTEXT.md §2's
# "IP-similarity-matrix work" entry for the full method and numbers.
# Negative control (leave-one-out among the 34 works): 5/34 unrelated pairs
# scored >= 0.82, but all 5 trace to two identified, structural confounds
# (same-event convergent journalism quoting an official statement verbatim;
# same-publisher beat clustering) rather than the matcher actually failing —
# excluding those, the genuine ceiling is 0.7615. Positive control (4
# hand-written substantial paraphrases of protected works): floor 0.8676,
# 4/4 correctly attributed and flagged. 0.82 sits inside the resulting
# [0.7615, 0.8676] margin and is defensible as measured. The two confound
# clusters are a real §1.7 limitation of whole-document cosine similarity
# (it cannot distinguish shared-quote/shared-beat convergence from actual
# copying) — not fixable by picking a different threshold number, so not
# addressed by moving this value. Do not re-tune without new evidence
# per CLAUDE.md rule 7.
IP_SIMILARITY_THRESHOLD = 0.82


# ── CV Settings (§3.12 A — Visual/Video Detection Engine) ────────────────────
FRAME_SAMPLE_RATE_FPS = 1
HUMAN_BLINK_RATE_MIN = 12
HUMAN_BLINK_RATE_MAX = 25
GAUSSIAN_BLUR_KERNEL = (5, 5)

# ── Sparse-detection / dense-landmark pipeline (Stage 2, watchdog/cv_engine.py) ──
# MTCNN is the expensive stage on CPU (Stage 2 benchmark: see
# benchmark_face_detection.py) so it only runs sparsely, at
# FRAME_SAMPLE_RATE_FPS (1 fps), to locate the face. Between MTCNN hits, the
# last bounding box is reused/tracked. MediaPipe Face Mesh is cheap by
# comparison, so it runs much denser — FACE_MESH_SAMPLE_RATE_FPS — on the
# (small) face crop only, to get the temporal resolution blink-rate/viseme
# analysis actually needs.
MTCNN_SAMPLE_RATE_FPS = FRAME_SAMPLE_RATE_FPS   # ~1 fps — matches existing frame sampling
FACE_MESH_SAMPLE_RATE_FPS = 10                  # dense landmarks on the tracked crop
MTCNN_MIN_FACE_SIZE = 40                        # matches extract_faces() default

# Identity-flicker fix (found during Stage 4 rigidity calibration, see
# docs/PROJECT_CONTEXT.md §4): facenet-pytorch MTCNN's keep_all=False picked
# whichever candidate face had the highest confidence at each sparse
# detection tick, independent of which face was tracked at the previous
# tick. On a frame with two visible faces this let the "tracked" identity
# silently switch between two different people mid-run, producing a huge
# spurious landmark-displacement spike at the switch rather than a real
# micro-expression signal (the 0.03/0.12 rigidity outliers on fc4b1a55 and
# 6d95436c). Fix: keep_all=True (see _load_winning_mtcnn()) plus IoU-based
# continuity in detect_face_bbox() — a detection tick only replaces the
# current track if its IoU against the previous box is >= this threshold;
# otherwise it falls back to highest confidence (fresh track / previous face
# left frame). 0.3 is the common association threshold in tracking-by-
# detection literature (e.g. SORT); a stricter 0.5 risks losing a genuine
# same-person track between sparse ~1s-apart MTCNN ticks (handheld motion,
# zooms), while going much lower risks matching a different face that
# happens to overlap slightly. Not independently calibrated against this
# corpus beyond confirming it resolves the two known flicker outliers.
MTCNN_TRACK_IOU_THRESHOLD = 0.3

# Fixed square canvas for letterboxing detector input (see cv_engine.letterbox()).
# The corpus spans 11 distinct resolutions across landscape/portrait/square —
# feeding the TF-based detector a different shape on every call forces a
# retrace/recompile each time; letterboxing to one fixed size collapses that
# to a single occurrence regardless of source resolution or orientation.
LETTERBOX_TARGET_SIZE = (640, 640)   # (width, height)

# NFR-02 (§3.4 Non-Functional Requirements): video analysis latency budget.
# README.md's "28.6s Achieved" figure predates the real duration-spread
# measurement below and was back-calculated assuming a uniform ~30s clip —
# see the Stage 2 per-clip timing projection for the real, non-uniform figure.
VIDEO_LATENCY_BUDGET_S = 30

# NFR-02's other half — text pipeline latency budget. README.md's NFR table
# has quoted "≤ 5s" for this since Day 1, but no named constant existed for
# it anywhere in this file until the Days 21-22 dashboard's /upload route
# became the first thing to actually time a full end-to-end run
# (main_pipeline.process_media()/process_manifest_sample(), not just one
# sub-stage) and needed a real budget to compare against rather than a
# magic 5 hardcoded into dashboard/app.py (CLAUDE.md: "ALL thresholds ...
# in config.py. No magic numbers elsewhere.").
TEXT_LATENCY_BUDGET_S = 5

# ── Stage 4 heuristics (§3.12 A four sub-detectors, watchdog/cv_engine.py) ───
#
# COVERAGE GATES — minimum longest-contiguous-face-present run required before
# a heuristic will report a score at all; below it, the heuristic returns
# None rather than a number computed from too little signal (same conservative
# posture as nlp_engine.analyze_text()'s UNCERTAIN vote, and for the same
# NFR-01 reason: a fabricated blink rate is exactly the kind of thing that
# ends up cited in a legal complaint). Set from a real coverage diagnostic run
# across all 32 CV-eligible clips (2026-08-03, see docs/PROJECT_CONTEXT.md
# §4) — not guessed:
#   - at >=8s, only 6/32 clips qualify, and ALL SIX ARE AUTHENTIC. Every
#     `ai_avatar_ad` synthetic sample in the corpus tops out at 6.5s of
#     contiguous tracking (cuts too often for MTCNN to hold a track longer).
#   - at >=3s, coverage is far more balanced: 12 authentic / 10 synthetic.
# This is a real corpus-coverage confound, not a threshold-tuning choice —
# the blink-rate heuristic will structurally abstain on every synthetic clip
# in the current corpus at any defensible floor above ~6.5s. Chapter 4 should
# state this plainly rather than let a silent None read as "no opinion."
MIN_FACE_RUN_FOR_BLINK_S = 10.0      # ~1-2 average inter-blink intervals at HUMAN_BLINK_RATE_MIN
MIN_FACE_RUN_FOR_RIGIDITY_S = 3.0    # needs a handful of consecutive frame-pairs, not a full blink cycle
MIN_FACE_RUN_FOR_VISEME_S = 3.0      # needs enough paired mouth/audio samples for a correlation
MIN_FRAMES_FOR_BLENDING = 5          # blending doesn't need contiguity, just enough independent samples

# BLINK RATE (Soukupová & Čech, 2016 — EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)).
# 0.2-0.25 is the commonly cited closed-eye threshold in EAR-based blink
# detection; 2 consecutive sub-threshold Face-Mesh ticks (~200ms at
# FACE_MESH_SAMPLE_RATE_FPS) approximates a real blink's closure duration
# without counting single-frame landmark noise as a blink.
EAR_BLINK_THRESHOLD = 0.21
EAR_MIN_CONSECUTIVE_FRAMES = 2

# EMOTIONAL RIGIDITY — variance of per-frame landmark displacement, normalised
# by face bbox size (scale-invariant), over the longest contiguous face run.
# The write-up's EMOTIONAL_RIGIDITY_THRESHOLD=3 (removed from this file) was a
# guess made before any real measurement existed, and is not even on the right
# scale: this project's calibration run (9 clips, 2026-08-03 — see
# docs/PROJECT_CONTEXT.md §4) measured real normalised-displacement variance
# in the 1.3e-5 to 3.3e-4 range, never anywhere near 3. Threshold set to
# 0.0001, the geometric mean of six clean calibration values, as a
# provisional real-scale starting point (CLAUDE.md rule 7: fix the root cause
# — the old value was wrong-scale, not just uncalibrated — rather than widen
# a guess to make a number work). At this sample size the six clean values do
# NOT separate cleanly by label (authentic: 9.0e-5, 1.3e-4, 2.5e-4; synthetic:
# 1.3e-5, 7.3e-5, 3.3e-4) — flagged honestly rather than tuned to look better.
#
# Two clips (fc4b1a55 authentic, 6d95436c synthetic) originally measured
# wildly higher (0.12, 0.03) than the six clean values above. 2026-08-04
# follow-up after adding MTCNN_TRACK_IOU_THRESHOLD-based continuity
# (detect_face_bbox() in cv_engine.py): fc4b1a55 resolved cleanly
# (0.12 -> 7.7e-5, now inside the clean range) — that one genuinely was the
# keep_all=False multi-face misattribution this fix targets. 6d95436c did
# NOT resolve (0.03 -> 0.027, essentially unchanged) because its outlier has
# a DIFFERENT root cause that IoU continuity cannot fix: a hard scene cut
# landing inside a single ~1s MTCNN sampling interval (verified by frame
# inspection at t=9.0-9.2s — same close-up shot at 9.0/9.1s, a completely
# different shot of the same actor at 9.2s). MTCNN only re-detects once per
# second, so Face Mesh spent one dense tick reading landmarks from a stale,
# now-wrong bounding box carried over from before the cut. There is no
# competing same-tick candidate for IoU continuity to disambiguate here —
# the frame content changed between ticks, not the count of faces within a
# tick — so this needs a different fix (e.g. a per-tick appearance/histogram
# sanity check on the crop, or MTCNN re-verification when Face Mesh
# confidence drops) which has NOT been implemented. Fast-cut ad footage
# (the ai_avatar_ad genre this clip belongs to) is exactly where this
# failure mode is most likely, so treat any single-clip rigidity score on
# that genre as provisional until this is addressed.
EMOTIONAL_RIGIDITY_THRESHOLD = 0.0001

# VISEME-PHONEME MISMATCH — Pearson correlation between per-frame Mouth
# Aspect Ratio and the audio RMS-energy envelope over the same window.
# Genuine speech is expected to show strong positive correlation; the same
# 9-clip calibration run found the opposite of "strong" across the board —
# correlations clustered near zero for BOTH classes (authentic: -0.055 to
# 0.127; synthetic: -0.161 to 0.411), with no clean separation. This may be a
# real limitation of coarse (50ms) RMS energy without band-pass filtering to
# speech frequencies, or of the simplified 2-point MAR — worth investigating
# before trusting this heuristic's scores, not papering over. Threshold set
# to 0.15 (roughly where the longest, most reliable authentic interview clips
# in the calibration sample landed) as a provisional floor, not a validated
# decision boundary — see docs/PROJECT_CONTEXT.md §4, which flags this
# alongside the NLP engine's Day-3 "burstiness contributed nothing" finding
# as the same category of honest null result.
VISEME_CORRELATION_THRESHOLD = 0.15

# BLENDING ARTIFACTS — Laplacian-variance sharpness ratio (face region vs.
# surrounding ring) and Canny edge-density discontinuity at the face
# boundary. ratio=1.0 means face and surroundings are equally sharp (real
# footage baseline); deviation either direction is scored. Same 9-clip
# calibration run: observed ratios overlapped heavily between classes
# (authentic 0.47-1.29, synthetic 0.61-0.92) and edge discontinuity was
# small and mostly negative for both. Thresholds below are provisional,
# set near the calibration sample's typical deviation — not a validated
# decision boundary. See docs/PROJECT_CONTEXT.md §4.
BLENDING_SHARPNESS_RATIO_DEVIATION_THRESHOLD = 0.35
BLENDING_EDGE_DISCONTINUITY_THRESHOLD = 0.03

# ── CV classifier transfer evaluation (Days 10-15/16) ────────────────────────
# notebooks/train_cv_classifier_colab.ipynb trains the Days 10-15 EfficientNet-B0
# classifier on FaceForensics++ in Colab (no local GPU — §1.7); evaluate_cv_transfer.py
# then runs the downloaded checkpoint zero-shot against this project's own
# Nigerian CV-eligible clips. Both use the SAME frame-sampling/aggregation
# scheme so the measured FF++-vs-Nigerian-corpus gap reflects domain
# transfer, not a methodology difference between the two evaluations.
CV_TRANSFER_FRAMES_PER_CLIP = 10     # evenly-spaced frames sampled per video/clip in both places
# Below this many valid face crops out of CV_TRANSFER_FRAMES_PER_CLIP sampled,
# a clip's aggregate score is too noisy to trust — abstain (same "abstain
# rather than fabricate" posture as every other CV heuristic in this project,
# see cv_engine.py's MIN_FACE_RUN_FOR_* gates) rather than average 1-2 crops
# into a confident-looking number. Provisional, not independently validated
# at this specific value.
CV_TRANSFER_MIN_FACE_CROPS = 3
# The gate notebooks/train_cv_classifier_colab.ipynb enforces on its own FF++
# held-out video-level accuracy before it will save a checkpoint at all — "if
# it doesn't reach roughly 90%+ on FF++'s own data, the pipeline is broken and
# any Nigerian-corpus number afterward is meaningless" (the instruction this
# was built under). Duplicated as a plain number inside the notebook itself
# (Colab can't cleanly import this project's config.py) with a comment
# pointing back here — kept here too so evaluate_cv_transfer.py's own gap
# report can cite the same number it was validated against.
CV_TRANSFER_FF_ACCURACY_GATE = 0.90


# ── Database (NFR-04 Auditability / NFR-05 Data Privacy) ─────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://watchdog_user:watchdog_pass@localhost:5432/naija_watchdog"
)
STORE_METADATA_ONLY_ON_VIOLATION = True


# ── yt-dlp Authentication (optional — for Facebook/Instagram downloads) ──────
# Both platforms block anonymous automated video downloads. yt-dlp supports
# two ways to authenticate as YOU (never store someone else's credentials):
#
# OPTION 1 (easiest, but fragile in WSL/non-standard installs): pull cookies
# directly from a browser's live storage. Set to 'chrome', 'firefox', 'edge',
# 'brave', etc. Set via .env: YTDLP_COOKIES_FROM_BROWSER=chrome
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "")

# OPTION 2 (more reliable — recommended if Option 1 fails, e.g. WSL): a
# manually-exported Netscape-format cookies.txt (e.g. via the "Get
# cookies.txt LOCALLY" browser extension). Only used if Option 1 above is
# not set. Set via .env: YTDLP_COOKIES_FILE=cookies.txt
# NEVER commit this file — it contains live session authentication.
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "")


# ── Flask Dashboard ───────────────────────────────────────────────────────────
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")
FLASK_DEBUG      = os.getenv("FLASK_DEBUG", "False").lower() == "true"
FLASK_HOST       = "0.0.0.0"
FLASK_PORT       = 5000


# ── Statutory Mapping (§3.13 Automated Complaint and Enforcement Logic) ───────
STATUTORY_MAP = {
    "DEEPFAKE_IMPERSONATION": {
        "statute": "Nigeria Data Protection Act (NDPA) 2023, Section 37",
        "summary": (
            "Section 37 of the NDPA 2023 grants data subjects the right not to be subject "
            "to decisions based solely on automated processing and profiling."
        ),
        "regulator": "Nigeria Data Protection Commission (NDPC)",
        "action": "Notice and Take-Down; Data Privacy Impact Assessment required.",
    },
    "UNDISCLOSED_SYNTHETIC_AVATAR": {
        "statute": "ARCON Advertising Guidelines (2023) & NDPA 2023, Section 37",
        "summary": (
            "The deployment of an AI-generated avatar to front a commercial advertisement "
            "without explicit disclosure to the consumer constitutes deceptive synthetic "
            "labor substitution."
        ),
        "regulator": "Advertising Regulatory Council of Nigeria (ARCON)",
        "action": "Mandatory AI Disclosure Label; Takedown of non-compliant asset.",
    },
    # CORRECTED 2026-08-04 — was "Section 108" (wrong; see below). Section
    # 108 is the Interpretation section of the Copyright Act 2022 — pure
    # definitions, no cause of action, cannot itself be infringed. The
    # correct provision is Section 36, which enumerates the acts that
    # constitute infringement; s.36(2) specifically covers reproducing "the
    # whole or a substantial part of the work either in its original form or
    # in any form recognisably derived from the original" — exactly what a
    # cosine-similarity threshold over embeddings is a proxy for. Section
    # 37(2) covers the civil remedies (damages, injunction, accounts of
    # profit) a complaint would actually be seeking. Section 108 is still
    # correctly citable, narrowly, for its definition of "copy" as including
    # a reproduction in electronic/digital form — just not as the
    # infringement provision itself. §3.12 B and §3.13 in the write-up still
    # cite the old, wrong section and need the same correction in the
    # Chapter 3 rewrite (see docs/PROJECT_CONTEXT.md).
    "IP_SCRAPING_PLAGIARISM": {
        "statute": "Copyright Act 2022, Section 36(2) (infringement); Section 37(2) (remedies)",
        "summary": (
            "Section 36(2) of the Copyright Act 2022 makes it infringement to reproduce "
            "\"the whole or a substantial part of the work either in its original form or "
            "in any form recognisably derived from the original\" without licence. Section "
            "37(2) provides the civil remedies (damages, injunction, account of profits) "
            "available to the rights-holder."
        ),
        "regulator": "Nigerian Copyright Commission (NCC)",
        "action": "Cease and Desist; Attribution or Licensing required.",
    },
}


# ── Ensure all directories exist on import ────────────────────────────────────
for _dir in [RAW_VIDEO_DIR, RAW_TEXT_DIR, PROCESSED_DIR,
             PROTECTED_IP_DIR, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
