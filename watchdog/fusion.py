# =============================================================================
# watchdog/fusion.py  —  Day 16: Weighted Fusion (§3.11 Mathematical Framework)
# =============================================================================
# PURPOSE:
#   §3.11 defines Psynth as a weighted fusion of the CV pathway's p_visual and
#   the NLP pathway's p_text: Psynth = alpha*p_visual + (1-alpha)*p_text
#   (config.FUSION_ALPHA). This module is the first thing that has ever
#   actually exercised config.FUSION_ALPHA against real per-modality scores
#   — config.py's comment on FUSION_ALPHA flagged this as untested until now.
#
# THE ABSTENTION PROBLEM THIS MODULE EXISTS TO SOLVE:
#   Every per-modality signal in this project abstains (returns None) rather
#   than guess when it doesn't have enough to say anything — cv_engine's four
#   Stage 4 heuristics, ip_similarity.check_similarity(), and now
#   nlp_engine.classify_text() all follow this pattern (NFR-01). On the real
#   corpus, abstention is NOT rare or random: docs/PROJECT_CONTEXT.md §4's
#   full 32-clip CV evaluation found blink-rate abstains on 100% of BOTH
#   classes in the matched subset, and the earlier per-clip coverage
#   diagnostic found blink-rate's abstention correlates near-perfectly with
#   label (every ai_avatar_ad synthetic clip cuts too often for a long-enough
#   contiguous face track; authentic interview footage mostly doesn't).
#
#   That correlation is a trap, not a signal: it comes from genre/edit-pace
#   (short-cut ad footage vs. long-form interviews), not from anything about
#   synthetic media itself. If this module let "abstained" silently count as
#   evidence anywhere in the arithmetic — e.g. treating a missing score as
#   0.0 before averaging or fusing — it would be laundering that genre
#   confound into Psynth as if it were a real detection signal, exactly the
#   kind of shortcut CLAUDE.md rule 7 and this project's whole "verify before
#   claiming" culture (docs/PROJECT_CONTEXT.md §1/§6) rules out. So the one
#   invariant every function below is built around:
#
#       AN ABSTAINED MODALITY OR HEURISTIC IS EXCLUDED FROM THE ARITHMETIC,
#       NEVER SUBSTITUTED WITH A NUMBER. A text-only Psynth (p_visual=None)
#       IS THE CORRECT OUTPUT, NOT A DEGRADED ONE.
#
#   Concretely: compute_psynth() never multiplies a missing modality by
#   alpha/(1-alpha) and adds it in — that would silently treat "no visual
#   evidence" as "visual evidence of 0.0 = definitely authentic", which is a
#   false claim this system must never make. When only one modality is
#   present, Psynth **is** that modality's own score, unweighted.
#
# ON p_visual's CURRENT RELIABILITY:
#   combine_cv_heuristics() below averages whichever of the four Stage 4
#   heuristics did NOT abstain on a given clip. Averaging is a real, defined
#   number — but docs/PROJECT_CONTEXT.md §4's full-corpus evaluation found
#   ALL FOUR heuristics fail to cleanly discriminate on the matched subset:
#   one (blink rate) returns no data at all, two (viseme mismatch, blending
#   artifacts) show heavy overlap or run backwards, and the fourth
#   (emotional rigidity) shows no clean separation either. The Days 10-15
#   EfficientNet-B0 classifier that was meant to replace this was built and
#   trained (three schedules), but never cleared its own pre-set 0.90 FF++
#   held-out credibility gate (best: 0.86) — closed 2026-08-12 as a
#   documented negative result (identity-group-count ceiling, not a fixable
#   training bug), not outstanding work. See docs/PROJECT_CONTEXT.md §4's
#   "CV classifier training" entry for the three-run table and full
#   reasoning. combine_cv_heuristics() below is therefore this project's
#   PERMANENT p_visual source for its remaining timeline, not a placeholder
#   awaiting a classifier. p_visual as computed today is a real number but a
#   WEAK one — reported honestly as "uncalibrated heuristic average" in
#   VisualAssessment, not hidden, and complaint_generator.py surfaces this
#   caveat verbatim in any generated complaint that relies on visual
#   evidence. Not fixed here by discarding CV entirely (that would waste the
#   one Stage-4 heuristic — emotional rigidity — that at least measures
#   something on the right real-world scale) or by tuning weights to make it
#   look better (CLAUDE.md rule 7).
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Names of the four Stage 4 heuristics on watchdog.cv_engine.ClipHeuristics,
# in the order combine_cv_heuristics() reads them. Kept as an explicit tuple
# (not introspected from the dataclass) so a future fifth heuristic added to
# ClipHeuristics doesn't silently start counting here without a decision.
_CV_HEURISTIC_NAMES = ("blink_rate", "emotional_rigidity", "viseme_mismatch", "blending_artifacts")


# =============================================================================
# Visual modality (CV pathway)
# =============================================================================

@dataclass
class VisualAssessment:
    p_visual: Optional[float]                                  # [0,1], higher = more synthetic-like; None = abstained
    scored_heuristics: list = field(default_factory=list)       # names that contributed a score
    abstained_heuristics: dict = field(default_factory=dict)    # name -> abstain_reason, for heuristics that didn't
    abstain_reason: Optional[str] = None                         # set only when p_visual itself is None


def combine_cv_heuristics(clip_heuristics) -> VisualAssessment:
    """
    Combines watchdog.cv_engine.ClipHeuristics' four Stage 4 heuristic scores
    into one p_visual by averaging only the ones that did NOT abstain.

    Args:
        clip_heuristics: A watchdog.cv_engine.ClipHeuristics instance (from
                          run_heuristics()), or None (e.g. no video / the
                          clip couldn't be opened — cv_engine.run_heuristics()
                          itself returns None in that case).

    Returns:
        VisualAssessment. p_visual is None (abstained) when clip_heuristics
        is None, or when every one of the four heuristics abstained on this
        specific clip — the real, common case for short-cut ad footage per
        config.py's MIN_FACE_RUN_FOR_BLINK_S comment. See the module
        docstring above for why an abstained heuristic is dropped from the
        average rather than counted as evidence.
    """
    if clip_heuristics is None:
        return VisualAssessment(
            p_visual=None,
            abstain_reason="no video track result — clip could not be opened/tracked (cv_engine.run_heuristics() returned None)",
        )

    scored = []
    abstained = {}
    for name in _CV_HEURISTIC_NAMES:
        result = getattr(clip_heuristics, name)
        if result.score is None:
            abstained[name] = result.abstain_reason or "abstained (no reason recorded)"
        else:
            scored.append((name, result.score))

    if not scored:
        return VisualAssessment(
            p_visual=None,
            abstained_heuristics=abstained,
            abstain_reason="all four Stage 4 heuristics abstained on this clip — see abstained_heuristics for per-heuristic reasons",
        )

    p_visual = sum(score for _, score in scored) / len(scored)
    return VisualAssessment(
        p_visual=p_visual,
        scored_heuristics=[name for name, _ in scored],
        abstained_heuristics=abstained,
    )


# =============================================================================
# Text modality (NLP pathway)
# =============================================================================

@dataclass
class TextAssessment:
    p_text: Optional[float]                    # [0,1], higher = more synthetic-like; None = abstained
    source: Optional[str] = None                # "distilbert_classifier" | None (abstained)
    heuristic_diagnostic: Optional[dict] = None  # nlp_engine.analyze_text() output — logged for context only
    abstain_reason: Optional[str] = None


def assess_text(text: Optional[str]) -> TextAssessment:
    """
    p_text comes from the fine-tuned classifier (nlp_engine.classify_text(),
    validated at FPR 0.088 — docs/PROJECT_CONTEXT.md §4), NOT from
    nlp_engine.analyze_text()'s Day 3 heuristic vote. That heuristic vote's
    own docstring already says it is "NOT the final system decision" — this
    is the fusion module that docstring deferred to. The heuristic result is
    still computed and attached as heuristic_diagnostic purely for
    transparency/logging (e.g. so a reviewer can see perplexity/burstiness
    ran, even though Day 3 found perplexity runs backwards on this corpus)
    — it never feeds into p_text's value.

    KEEP, DON'T PRUNE — WHY analyze_text() STILL RUNS ON EVERY LIVE REQUEST:
    The 2026-08-13 NFR-02 profiling pass (docs/PROJECT_CONTEXT.md §4) found
    this call costs ~0.5-1.0s of a warm ~1s text request — the single
    largest line item in the whole warm pipeline once preprocessor.py's own
    spaCy-caching bug was fixed the same day. That's a real, deliberate
    spend against the 5s TEXT_LATENCY_BUDGET_S, kept for two reasons, not
    an accident someone forgot to strip out:
      1. It is the only place perplexity/burstiness are computed against
         real submitted content, which is what makes NFR-03's register-bias
         finding (README.md: DistilGPT-2 scores real Nigerian journalism as
         unexpectedly low-perplexity, i.e. falsely "AI-like") visible
         per-sample rather than only in the one-off historical
         nlp_heuristic_eval.csv run. Losing this call would mean losing the
         only live signal that finding is still true (or has stopped being
         true) on whatever gets submitted next.
      2. heuristic_diagnostic ends up in evidence_json (via
         complaint_generator's build_complaint_context) attached to every
         violation this system raises. In a legal-evidence system, "the
         perplexity/burstiness were computed and here is what they showed,
         even though the system's decision didn't rely on them" is a
         forensic detail worth a reviewer being able to see and a defence
         being able to contest — cheap insurance against "how do you know
         this wasn't just the classifier guessing" at ~1s/request out of a
         5s budget.
    If a future pass needs to reclaim this budget, the right move is
    speeding up compute_perplexity() itself (e.g. skip DistilGPT-2 entirely
    below some word-count floor where burstiness alone is uninformative
    anyway), not deleting the call — see docs/PROJECT_CONTEXT.md §4's
    2026-08-13 entry before removing this as "dead weight."

    Abstains (p_text=None) when `text` is empty/missing, or when
    classify_text() itself abstains (no trained checkpoint on this machine).
    """
    if not text or not text.strip():
        return TextAssessment(p_text=None, abstain_reason="no text supplied")

    from watchdog.nlp_engine import classify_text, analyze_text

    classification = classify_text(text)

    heuristic_diagnostic = None
    try:
        heuristic_diagnostic = analyze_text(text)
    except Exception as e:
        # Diagnostic-only signal — must never block or alter the real p_text
        # decision above just because e.g. DistilGPT-2 isn't downloaded.
        logger.warning(f"assess_text: heuristic diagnostic failed (non-fatal, p_text unaffected): {e}")

    if classification.p_synthetic is None:
        return TextAssessment(
            p_text=None,
            heuristic_diagnostic=heuristic_diagnostic,
            abstain_reason=classification.abstain_reason,
        )

    return TextAssessment(
        p_text=classification.p_synthetic,
        source="distilbert_classifier",
        heuristic_diagnostic=heuristic_diagnostic,
    )


# =============================================================================
# Fusion — the §3.11 gate
# =============================================================================

@dataclass
class FusionResult:
    psynth: Optional[float]
    p_visual: Optional[float]
    p_text: Optional[float]
    modalities_used: list
    alpha: float
    visual: VisualAssessment
    text: TextAssessment
    abstain_reason: Optional[str] = None


def compute_psynth(p_visual: Optional[float], p_text: Optional[float], alpha: Optional[float] = None):
    """
    The §3.11 fusion arithmetic, with correct missing-modality handling.

    Returns (psynth, modalities_used):
      - both present : alpha*p_visual + (1-alpha)*p_text   -> ["visual", "text"]
      - visual only  : p_visual (unweighted)                -> ["visual"]
      - text only    : p_text (unweighted)                  -> ["text"]
      - neither      : None                                 -> []

    Deliberately NOT `alpha*(p_visual or 0) + (1-alpha)*(p_text or 0)` — that
    would silently convert "no visual evidence" into "visual evidence of
    0.0" (i.e. active counter-evidence pulling Psynth down), which is not
    what a missing modality means. See the module docstring.
    """
    if alpha is None:
        from config import FUSION_ALPHA
        alpha = FUSION_ALPHA

    if p_visual is not None and p_text is not None:
        return alpha * p_visual + (1 - alpha) * p_text, ["visual", "text"]
    if p_visual is not None:
        return p_visual, ["visual"]
    if p_text is not None:
        return p_text, ["text"]
    return None, []


def run_fusion(clip_heuristics=None, text: Optional[str] = None, alpha: Optional[float] = None) -> FusionResult:
    """
    Top-level §3.11 entry point: assesses both modalities independently, then
    fuses them. This is what main_pipeline.py calls per sample.

    Args:
        clip_heuristics: watchdog.cv_engine.ClipHeuristics or None (text-only
                          sample, e.g. TEXTUAL_CORPUS — no video at all).
        text:             Raw text or None (video-only sample).
        alpha:            Overrides config.FUSION_ALPHA — mainly for tests.
    """
    if alpha is None:
        from config import FUSION_ALPHA
        alpha = FUSION_ALPHA

    visual = combine_cv_heuristics(clip_heuristics)
    text_assessment = assess_text(text)
    psynth, modalities_used = compute_psynth(visual.p_visual, text_assessment.p_text, alpha=alpha)

    abstain_reason = None
    if psynth is None:
        reasons = [r for r in (visual.abstain_reason, text_assessment.abstain_reason) if r]
        abstain_reason = "both modalities abstained/missing — " + "; ".join(reasons) if reasons else \
            "both modalities abstained/missing"

    return FusionResult(
        psynth=psynth,
        p_visual=visual.p_visual,
        p_text=text_assessment.p_text,
        modalities_used=modalities_used,
        alpha=alpha,
        visual=visual,
        text=text_assessment,
        abstain_reason=abstain_reason,
    )
