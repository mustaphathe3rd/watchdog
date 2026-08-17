# =============================================================================
# watchdog/nlp_engine.py  —  NLP Detection Engine (Day 3)
# =============================================================================
# PURPOSE:
#   Implements §3.12 B — Textual and Plagiarism Detection Engine:
#   "analyzed the perplexity (the randomness of word choice) and burstiness
#    (the variation in sentence length)..."
#
# DESIGN NOTE — why DistilGPT-2 for perplexity, not DistilBERT:
#   config.py's NLP_BASE_MODEL is "distilbert-base-uncased", matching §3.12 B.
#   However, DistilBERT is a MASKED/bidirectional language model — it has no
#   natural, well-defined autoregressive "perplexity" the way a causal model
#   does (perplexity is fundamentally P(token_i | token_1..i-1), which only
#   a left-to-right model computes directly). Standard AI-text detectors
#   (GPTZero and similar) compute perplexity via a small causal LM for
#   exactly this reason. This module therefore uses a SECOND, lightweight
#   model — DistilGPT-2 (82M params, CPU-friendly) — purely as a feature
#   extractor for perplexity. DistilBERT remains reserved for the actual
#   fine-tuned CLASSIFICATION head built in Day 8-9, per the original spec.
#   This is a hybrid statistical-feature + deep-learning design, not a
#   deviation from §3.12 B — it makes the "perplexity" term in your own
#   literature actually well-defined and computable.
#
# BURSTINESS METRIC — Goh & Barabási (2008) formulation:
#   B = (σ - μ) / (σ + μ)
#   where σ, μ are the standard deviation and mean of sentence lengths
#   (word counts). This is a named, citable metric bounded in [-1, 1]:
#     B → +1  : highly irregular/"bursty" sentence-length distribution
#               (characteristic of human writing)
#     B → -1  : highly regular/uniform sentence lengths
#               (characteristic of LLM-generated text)
#     B ≈  0  : Poisson-random distribution
#   Reference: Goh, K.I. & Barabási, A.L. (2008). "Burstiness and memory in
#   complex systems." EPL, 81(4), 48002.
# =============================================================================

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache — the model is loaded once, not on every call.
# CPU inference of a fresh 82M-param model per call would be far too slow
# for batch evaluation of hundreds of samples (§1.7 CPU-only constraint).
_MODEL = None
_TOKENIZER = None

# DistilGPT-2's trained context window. Nigerian news articles frequently
# exceed this (§3.9 sources have run 2,000-76,000+ characters) — see the
# chunking logic in compute_perplexity() below.
_MAX_CHUNK_TOKENS = 512


def _load_perplexity_model():
    """
    Lazily loads DistilGPT-2 and its tokenizer, caching both at module
    level. Called automatically by compute_perplexity() — callers never
    need to invoke this directly.
    """
    global _MODEL, _TOKENIZER

    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    logger.info("Loading DistilGPT-2 for perplexity scoring (first call only)...")
    _TOKENIZER = GPT2TokenizerFast.from_pretrained("distilgpt2")
    _MODEL = GPT2LMHeadModel.from_pretrained("distilgpt2")
    _MODEL.eval()  # inference only — never trained/fine-tuned itself
    torch.set_num_threads(max(1, torch.get_num_threads()))  # CPU-friendly default

    logger.info("DistilGPT-2 loaded.")
    return _MODEL, _TOKENIZER


def compute_perplexity(text: str) -> Optional[float]:
    """
    Computes perplexity of `text` under DistilGPT-2.

    Low perplexity = the model found the text highly predictable (every
    word was close to what it expected next) = a hallmark of LLM-generated
    text, which tends toward the statistically "safest" word choices.
    High perplexity = more surprising, varied word choice = more
    human-like, per §3.12 B's own framing ("the randomness of word choice").

    Handles arbitrarily long documents via chunking: text is split into
    non-overlapping windows of _MAX_CHUNK_TOKENS tokens, loss is computed
    per chunk, and the final perplexity is derived from the token-weighted
    average loss across all chunks — not just a truncated first slice,
    which would silently ignore most of a long article.

    Args:
        text: Raw or preprocessed text. Works with either, but raw text
              (preserving punctuation/casing) gives DistilGPT-2 more
              natural signal than aggressively cleaned text.

    Returns:
        Perplexity as a float (>= 1.0, lower = more predictable/AI-like).
        None if the text is empty or contains no scoreable tokens.
    """
    text = (text or "").strip()
    if not text:
        return None

    import torch

    model, tokenizer = _load_perplexity_model()

    # verbose=False: suppress HF's "sequence length > model_max_length"
    # warning. It fires on this raw, pre-chunk encode() call, but the loop
    # below already splits into _MAX_CHUNK_TOKENS windows before the text
    # ever reaches the model, so the warning is a false positive here.
    input_ids = tokenizer.encode(text, return_tensors="pt", verbose=False)
    if input_ids.shape[1] < 2:
        # Need at least 2 tokens to compute a next-token prediction loss
        logger.warning("Text too short to score (< 2 tokens) — returning None")
        return None

    total_nll = 0.0  # sum of negative log-likelihood across all chunks
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, input_ids.shape[1], _MAX_CHUNK_TOKENS):
            chunk = input_ids[:, start:start + _MAX_CHUNK_TOKENS]
            if chunk.shape[1] < 2:
                continue  # trailing chunk too short to score, skip

            outputs = model(chunk, labels=chunk)
            # outputs.loss is the MEAN cross-entropy over (chunk_len - 1)
            # predictions. Convert back to a token-weighted SUM so multiple
            # chunks of different lengths average correctly.
            n_predicted = chunk.shape[1] - 1
            total_nll += outputs.loss.item() * n_predicted
            total_tokens += n_predicted

    if total_tokens == 0:
        return None

    mean_nll = total_nll / total_tokens
    perplexity = math.exp(mean_nll)
    return perplexity


def compute_burstiness(sentences: list[str]) -> Optional[float]:
    """
    Computes the Goh & Barabási (2008) burstiness parameter over sentence
    lengths (word counts): B = (σ - μ) / (σ + μ).

    Args:
        sentences: List of sentence strings, e.g. from
                   preprocessor.preprocess_text()['sentences'].

    Returns:
        Burstiness in [-1, 1], or None if fewer than 2 sentences (variance
        is undefined for a single data point — not a bug, a genuine
        limitation of the metric itself, worth noting in Chapter 4 if a
        very short sample produces None).
    """
    if not sentences or len(sentences) < 2:
        return None

    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return None

    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std = math.sqrt(variance)

    if (std + mean) == 0:
        return 0.0  # degenerate case: all-zero-length sentences

    burstiness = (std - mean) / (std + mean)
    return burstiness


def analyze_text(text: str) -> dict:
    """
    Runs the full NLP heuristic analysis on a piece of text: preprocessing,
    perplexity, and burstiness, combined into one result dict. This is the
    Day 3 deliverable — the statistical-feature half of the eventual hybrid
    NLP engine. The fine-tuned DistilBERT classifier (Day 8-9) will later
    combine these same two signals with its own learned representation.

    Args:
        text: Raw text to analyze.

    Returns:
        dict with keys:
            'perplexity':        float or None
            'burstiness':         float or None
            'perplexity_flag':    bool — True if perplexity suggests AI text
                                  (below config.PERPLEXITY_THRESHOLD)
            'burstiness_flag':    bool — True if burstiness suggests AI text
                                  (below config.BURSTINESS_THRESHOLD)
            'heuristic_vote':     'AI' | 'HUMAN' | 'UNCERTAIN' — simple
                                  majority vote across the two flags, NOT
                                  the final system decision (§3.11's fusion
                                  module and the trained classifier make
                                  that call — this is a diagnostic signal).
            'word_count':         int, from preprocessor
            'sentence_count':     int, from preprocessor
    """
    from config import PERPLEXITY_THRESHOLD, BURSTINESS_THRESHOLD
    from watchdog.preprocessor import preprocess_text

    preprocessed = preprocess_text(text)
    perplexity = compute_perplexity(text)
    burstiness = compute_burstiness(preprocessed["sentences"])

    perplexity_flag = (perplexity is not None) and (perplexity < PERPLEXITY_THRESHOLD)
    burstiness_flag = (burstiness is not None) and (burstiness < BURSTINESS_THRESHOLD)

    votes = [v for v in (perplexity_flag, burstiness_flag) if v is not None]
    if perplexity is None and burstiness is None:
        heuristic_vote = "UNCERTAIN"
    elif sum([perplexity_flag, burstiness_flag]) >= 1 and (perplexity is not None and burstiness is not None):
        # Both signals available: require BOTH to agree before calling it
        # "AI" outright, since either heuristic alone has known failure
        # modes (e.g. short quotes naturally read as low-burstiness; formal
        # bureaucratic prose naturally reads as low-perplexity). Matches
        # the conservative, false-positive-averse posture required by
        # NFR-01/03 and the VSD principle in §2.2.3.
        heuristic_vote = "AI" if (perplexity_flag and burstiness_flag) else (
            "UNCERTAIN" if (perplexity_flag or burstiness_flag) else "HUMAN"
        )
    else:
        heuristic_vote = "HUMAN"

    return {
        "perplexity": perplexity,
        "burstiness": burstiness,
        "perplexity_flag": perplexity_flag,
        "burstiness_flag": burstiness_flag,
        "heuristic_vote": heuristic_vote,
        "word_count": preprocessed["word_count"],
        "sentence_count": preprocessed["sentence_count"],
    }


# =============================================================================
# Fine-tuned classifier inference (Day 8-9 model, wired for live use Day 16)
# =============================================================================
# analyze_text() above is the Day 3 STATISTICAL-FEATURE half of this module's
# hybrid design and explicitly documents that its heuristic_vote is "NOT the
# final system decision" (§3.11's fusion module makes that call). The other
# half — train_nlp_classifier.py's fine-tuned DistilBERT, validated at FPR
# 0.088 on real data (docs/PROJECT_CONTEXT.md §4) — had no inference-time
# loader until now; training only ever saved the checkpoint. classify_text()
# below is that loader, and is what watchdog/fusion.py calls for p_text: the
# validated classifier, not the heuristic vote, is the real per-text
# synthetic-probability signal §3.11 fuses.
_CLASSIFIER_MODEL = None
_CLASSIFIER_TOKENIZER = None
_CLASSIFIER_MAX_LENGTH = None


@dataclass
class TextClassification:
    p_synthetic: Optional[float]         # [0,1] — softmax P(label=1=SYNTHETIC); None if abstained
    predicted_label: Optional[str]       # "SYNTHETIC" | "AUTHENTIC" | None if abstained
    abstain_reason: Optional[str] = None


def _load_classifier():
    """
    Lazily loads the fine-tuned checkpoint at config.NLP_MODEL_PATH, caching
    at module level (same pattern as _load_perplexity_model() above). Returns
    (None, None, None) if the checkpoint doesn't exist yet — classify_text()
    turns that into a clean abstain rather than a crash, since "the
    classifier hasn't been trained/downloaded on this machine" is an expected
    state on a fresh checkout, not an error.
    """
    global _CLASSIFIER_MODEL, _CLASSIFIER_TOKENIZER, _CLASSIFIER_MAX_LENGTH

    if _CLASSIFIER_MODEL is not None:
        return _CLASSIFIER_MODEL, _CLASSIFIER_TOKENIZER, _CLASSIFIER_MAX_LENGTH

    from config import NLP_MODEL_PATH

    if not NLP_MODEL_PATH.exists():
        return None, None, None

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading fine-tuned classifier from {NLP_MODEL_PATH} (first call only)...")
    checkpoint = torch.load(NLP_MODEL_PATH, map_location="cpu", weights_only=False)
    base_model = checkpoint["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _CLASSIFIER_MODEL = model
    _CLASSIFIER_TOKENIZER = tokenizer
    _CLASSIFIER_MAX_LENGTH = checkpoint["max_length"]
    logger.info("Fine-tuned classifier loaded.")
    return _CLASSIFIER_MODEL, _CLASSIFIER_TOKENIZER, _CLASSIFIER_MAX_LENGTH


def classify_text(text: str) -> TextClassification:
    """
    Runs the fine-tuned DistilBERT classifier (train_nlp_classifier.py) on
    `text` and returns P(synthetic).

    Abstains (p_synthetic=None) rather than guessing — same conservative
    posture as every other abstain path in this project (NFR-01) — when:
      - `text` is empty, or
      - config.NLP_MODEL_PATH doesn't exist (classifier never trained /
        checkpoint not downloaded onto this machine).
    Never raises for either case.

    Unlike compute_perplexity()'s _MAX_CHUNK_TOKENS chunking, long input is
    simply truncated at the checkpoint's own max_length (matches how the
    classifier was trained in train_nlp_classifier.py — truncation, not
    chunking — so inference-time behaviour matches training-time behaviour
    exactly rather than introducing a new averaging scheme the model never
    saw).
    """
    text = (text or "").strip()
    if not text:
        return TextClassification(p_synthetic=None, predicted_label=None, abstain_reason="empty text")

    model, tokenizer, max_length = _load_classifier()
    if model is None:
        from config import NLP_MODEL_PATH
        return TextClassification(
            p_synthetic=None, predicted_label=None,
            abstain_reason=(f"no trained classifier checkpoint at {NLP_MODEL_PATH} — "
                             f"run train_nlp_classifier.py first"),
        )

    import torch

    with torch.no_grad():
        encoding = tokenizer([text], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        logits = model(**encoding).logits
        p_synthetic = torch.softmax(logits, dim=-1)[0, 1].item()

    predicted_label = "SYNTHETIC" if p_synthetic >= 0.5 else "AUTHENTIC"
    return TextClassification(p_synthetic=p_synthetic, predicted_label=predicted_label)
