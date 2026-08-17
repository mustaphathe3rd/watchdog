# =============================================================================
# watchdog/rules_engine.py  —  Day 16: Rules-as-Code Statutory Mapping (§3.13)
# =============================================================================
# PURPOSE:
#   Maps a fused Psynth decision (watchdog.fusion.FusionResult) plus evidence
#   context (media type, dataset category, IP-similarity match) onto exactly
#   one of the three violation types this project models, and the specific
#   Nigerian statute each one cites (config.STATUTORY_MAP) — the corrected
#   citations, not the write-up's original wrong ones (see
#   docs/PROJECT_CONTEXT.md's 2026-08-04 legal-correction entry):
#     - Video, POLITICAL_MEDIA category  -> DEEPFAKE_IMPERSONATION, NDPA 2023 s.37
#     - Video, other (ad/commercial)     -> UNDISCLOSED_SYNTHETIC_AVATAR,
#                                            ARCON guidelines + NDPA 2023 s.37
#     - Text, IP-similarity match        -> IP_SCRAPING_PLAGIARISM,
#                                            Copyright Act 2022 s.36(2)
#                                            (infringement) / s.37(2) (remedies)
#   Section 108 of the Copyright Act is deliberately never cited here as the
#   infringement basis — it is the Interpretation section (definitions only,
#   no cause of action; still correctly citable narrowly for its definition
#   of "copy" as including digital reproductions, which complaint_generator.py
#   may reference, but never as *why* something is infringing).
#
# WHY A SINGLE Psynth GATE FOR ALL THREE TYPES, INCLUDING PLAGIARISM:
#   config.PSYNTH_THRESHOLD's own comment is explicit: Psynth "must EXCEED
#   this value to trigger the Compliance Mapping and Auto-Complaint modules"
#   — no carve-out for the IP pathway. config.IP_SIMILARITY_THRESHOLD (0.82)
#   answers a DIFFERENT question — "does this text substantially reproduce a
#   SPECIFIC protected work" (Copyright Act 2022 s.36(2)) — from the one
#   Psynth answers — "is this media AI-generated at all" (§3.11). This
#   project's own scope (CLAUDE.md: "AI-plagiarised IP", not just
#   "plagiarised IP") is specifically about AI-DRIVEN plagiarism, so both
#   gates apply together: a human plagiarising another human (ip_result
#   flagged, but p_text/Psynth low because the classifier doesn't think the
#   text is AI-written) is correctly OUT of scope for this system and is not
#   escalated — that is a real, deliberate scope boundary, not an oversight.
#
# WHAT HAPPENS WHEN Psynth CROSSES THE THRESHOLD BUT NOTHING FITS:
#   Text content that scores as AI-generated (high p_text) but matches no
#   protected work has no modelled cause of action here at all — writing
#   text with an LLM is not itself illegal; only reproducing a specific
#   protected work is (Copyright Act) and that's the only text-side statute
#   this system maps. evaluate() returns None in that case rather than
#   guessing a violation_type db_connector.Violation.violation_type doesn't
#   even have a slot for. This is a genuine §1.7 scope limitation, stated
#   plainly (logged as a WARNING with the reasoning) rather than solved by
#   silently forcing the sample into one of the three known buckets.
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

VIOLATION_TYPE_DEEPFAKE_IMPERSONATION = "DEEPFAKE_IMPERSONATION"
VIOLATION_TYPE_UNDISCLOSED_SYNTHETIC_AVATAR = "UNDISCLOSED_SYNTHETIC_AVATAR"
VIOLATION_TYPE_IP_SCRAPING_PLAGIARISM = "IP_SCRAPING_PLAGIARISM"


@dataclass
class RuleDecision:
    violation_type: str
    statute: str
    statute_summary: str
    regulator: str
    action: str
    psynth: float
    basis: str   # human-readable explanation of WHY this violation_type was chosen — goes into the complaint


def determine_violation_type(*, media_type: str, category: Optional[str] = None, ip_result=None):
    """
    Classifies which of the three modelled violation types applies, given
    evidence context. Does NOT check Psynth itself — evaluate() below is the
    only function that gates on the threshold; this one answers "if this IS
    a violation, which kind" so the two decisions (whether / which) stay
    separately testable.

    Returns (violation_type_or_None, basis_string). violation_type is None
    exactly when the evidence doesn't fit any of the three modelled types
    (see module docstring's "text with no IP match" case) — basis_string
    still explains why, for logging/the complaint's own text.
    """
    if ip_result is not None and getattr(ip_result, "flagged", False):
        top = ip_result.matches[0]
        return VIOLATION_TYPE_IP_SCRAPING_PLAGIARISM, (
            f"IP similarity check matched protected work {top.sample_id} "
            f"({top.publisher_name}) at cosine similarity {top.similarity:.4f}, "
            f">= config.IP_SIMILARITY_THRESHOLD"
        )

    if media_type == "video":
        from watchdog.dataset_manager import CATEGORY_POLITICAL_MEDIA

        if category == CATEGORY_POLITICAL_MEDIA:
            return VIOLATION_TYPE_DEEPFAKE_IMPERSONATION, (
                "video in the POLITICAL_MEDIA category crossed the Psynth threshold — "
                "modelled as impersonation of a real public figure (§3.9 Table 3.4)"
            )
        return VIOLATION_TYPE_UNDISCLOSED_SYNTHETIC_AVATAR, (
            "video crossed the Psynth threshold with no confirmed IP match — "
            "modelled as an undisclosed synthetic avatar in an advert/commercial context "
            "(§3.9 Table 3.4)"
        )

    return None, (
        "text crossed the Psynth threshold (high AI-generation probability) but matched no "
        "protected work in the IP-similarity check — this system's modelled statutory scope "
        "covers AI-generated text only when it also reproduces a specific protected work "
        "(Copyright Act 2022 s.36(2)); undisclosed AI-authorship alone has no cause of action "
        "modelled here. Not escalated — see docs/PROJECT_CONTEXT.md / §1.7."
    )


def map_to_statute(violation_type: str) -> dict:
    """
    Fail-loud lookup into config.STATUTORY_MAP. Raises rather than fabricate
    a citation for a violation_type nobody has added a mapping for yet —
    a wrong or invented statute in a legal complaint is far worse than a
    crashed pipeline run.
    """
    from config import STATUTORY_MAP

    if violation_type not in STATUTORY_MAP:
        raise KeyError(
            f"No config.STATUTORY_MAP entry for violation_type={violation_type!r}. "
            f"Known types: {sorted(STATUTORY_MAP)}. Add the mapping before wiring a new "
            f"violation_type through the pipeline — this module never guesses a citation."
        )
    return STATUTORY_MAP[violation_type]


def evaluate(*, media_type: str, psynth: Optional[float], category: Optional[str] = None,
             ip_result=None, threshold: Optional[float] = None) -> Optional[RuleDecision]:
    """
    The single Psynth gate for all three violation types (see module
    docstring). Returns None (no violation, nothing written to `violations`,
    no complaint generated) when:
      - psynth is None (full abstain — both modalities missing), or
      - psynth is below threshold, or
      - determine_violation_type() can't classify the evidence into one of
        the three modelled types.

    Args:
        media_type: 'video' | 'text' (from the MediaLog entry / sample).
        psynth:     watchdog.fusion.FusionResult.psynth.
        category:   dataset category (watchdog.dataset_manager constants) —
                    optional; a live scrape outside the training categories
                    can still pass None and falls through to the
                    UNDISCLOSED_SYNTHETIC_AVATAR default for video.
        ip_result:  watchdog.ip_similarity.IPSimilarityResult, or None for
                    video content (no IP check applies to video).
        threshold:  overrides config.PSYNTH_THRESHOLD — mainly for tests.
    """
    if threshold is None:
        from config import PSYNTH_THRESHOLD
        threshold = PSYNTH_THRESHOLD

    if psynth is None:
        logger.info("rules_engine.evaluate: psynth is None (both modalities abstained) — no violation raised.")
        return None
    if psynth < threshold:
        logger.debug(f"rules_engine.evaluate: psynth={psynth:.4f} below threshold={threshold} — no violation raised.")
        return None

    violation_type, basis = determine_violation_type(media_type=media_type, category=category, ip_result=ip_result)
    if violation_type is None:
        logger.warning(
            f"rules_engine.evaluate: psynth={psynth:.4f} >= threshold={threshold} but no statutorily-mapped "
            f"violation_type applies. Reason: {basis}"
        )
        return None

    mapping = map_to_statute(violation_type)
    logger.info(
        f"rules_engine.evaluate: violation_type={violation_type}, psynth={psynth:.4f}, "
        f"statute='{mapping['statute']}'"
    )
    return RuleDecision(
        violation_type=violation_type,
        statute=mapping["statute"],
        statute_summary=mapping["summary"],
        regulator=mapping["regulator"],
        action=mapping["action"],
        psynth=psynth,
        basis=basis,
    )
