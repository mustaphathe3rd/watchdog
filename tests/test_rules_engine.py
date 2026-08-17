# =============================================================================
# tests/test_rules_engine.py  —  §3.13 Rules-as-Code Test Suite (Day 16)
# =============================================================================
# Covers:
#   - determine_violation_type()'s three routing branches, incl. the
#     deliberate "no statute applies" text-without-IP-match case
#   - map_to_statute()'s fail-loud behaviour on an unmapped type
#   - evaluate()'s single Psynth gate (threshold, None-abstain, and the
#     "crossed threshold but no applicable type" non-escalation case)
# =============================================================================

import pytest

from watchdog import rules_engine
from watchdog.dataset_manager import CATEGORY_POLITICAL_MEDIA, CATEGORY_TELECOM_FMCG_ADS
from watchdog.ip_similarity import IPSimilarityResult, SimilarityMatch
from config import STATUTORY_MAP, PSYNTH_THRESHOLD


def _ip_match(similarity=0.9, flagged=True):
    match = SimilarityMatch(
        sample_id="abc123", publisher_name="TheCable", source_url="https://thecable.ng/x",
        similarity=similarity, matched_passage="Some quoted sentence.",
    )
    return IPSimilarityResult(matches=[match], flagged=flagged)


# =============================================================================
# determine_violation_type()
# =============================================================================

class TestDetermineViolationType:
    def test_ip_match_wins_regardless_of_media_type(self):
        vtype, basis = rules_engine.determine_violation_type(
            media_type="text", category=None, ip_result=_ip_match()
        )
        assert vtype == rules_engine.VIOLATION_TYPE_IP_SCRAPING_PLAGIARISM
        assert "0.9" in basis

    def test_ip_result_present_but_not_flagged_is_ignored(self):
        vtype, _ = rules_engine.determine_violation_type(
            media_type="text", category=None, ip_result=_ip_match(similarity=0.5, flagged=False)
        )
        assert vtype is None

    def test_video_political_media_is_deepfake_impersonation(self):
        vtype, basis = rules_engine.determine_violation_type(
            media_type="video", category=CATEGORY_POLITICAL_MEDIA, ip_result=None
        )
        assert vtype == rules_engine.VIOLATION_TYPE_DEEPFAKE_IMPERSONATION
        assert "POLITICAL_MEDIA" in basis

    def test_video_ads_category_is_undisclosed_synthetic_avatar(self):
        vtype, _ = rules_engine.determine_violation_type(
            media_type="video", category=CATEGORY_TELECOM_FMCG_ADS, ip_result=None
        )
        assert vtype == rules_engine.VIOLATION_TYPE_UNDISCLOSED_SYNTHETIC_AVATAR

    def test_video_unknown_category_defaults_to_undisclosed_synthetic_avatar(self):
        vtype, _ = rules_engine.determine_violation_type(media_type="video", category=None, ip_result=None)
        assert vtype == rules_engine.VIOLATION_TYPE_UNDISCLOSED_SYNTHETIC_AVATAR

    def test_text_with_no_ip_match_has_no_applicable_statute(self):
        """
        The deliberate scope boundary: AI-generated text that reproduces no
        protected work has no modelled cause of action, even if it's
        obviously AI-written. Must return None, not force it into a bucket.
        """
        vtype, basis = rules_engine.determine_violation_type(media_type="text", category=None, ip_result=None)
        assert vtype is None
        assert "no cause of action" in basis.lower() or "not escalated" in basis.lower()


# =============================================================================
# map_to_statute()
# =============================================================================

class TestMapToStatute:
    @pytest.mark.parametrize("vtype", list(STATUTORY_MAP.keys()))
    def test_known_types_return_the_config_mapping_verbatim(self, vtype):
        mapping = rules_engine.map_to_statute(vtype)
        assert mapping == STATUTORY_MAP[vtype]

    def test_unknown_type_raises_rather_than_fabricating(self):
        with pytest.raises(KeyError):
            rules_engine.map_to_statute("SOME_MADE_UP_VIOLATION_TYPE")


# =============================================================================
# evaluate()  — the single Psynth gate
# =============================================================================

class TestEvaluate:
    def test_none_psynth_never_raises_a_violation(self):
        assert rules_engine.evaluate(media_type="video", psynth=None, category=CATEGORY_POLITICAL_MEDIA) is None

    def test_below_threshold_never_raises_a_violation(self):
        decision = rules_engine.evaluate(
            media_type="video", psynth=PSYNTH_THRESHOLD - 0.01, category=CATEGORY_POLITICAL_MEDIA
        )
        assert decision is None

    def test_at_or_above_threshold_video_raises_deepfake_impersonation(self):
        decision = rules_engine.evaluate(
            media_type="video", psynth=PSYNTH_THRESHOLD, category=CATEGORY_POLITICAL_MEDIA
        )
        assert decision is not None
        assert decision.violation_type == "DEEPFAKE_IMPERSONATION"
        assert decision.statute == STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["statute"]
        assert decision.psynth == pytest.approx(PSYNTH_THRESHOLD)

    def test_text_above_threshold_with_ip_match_raises_ip_scraping_plagiarism(self):
        decision = rules_engine.evaluate(
            media_type="text", psynth=0.95, category=None, ip_result=_ip_match(similarity=0.9),
        )
        assert decision is not None
        assert decision.violation_type == "IP_SCRAPING_PLAGIARISM"
        assert decision.statute == STATUTORY_MAP["IP_SCRAPING_PLAGIARISM"]["statute"]

    def test_text_above_threshold_without_ip_match_does_not_escalate(self):
        """
        The Psynth gate alone is not sufficient for text — this is the
        deliberate non-escalation case, not a bug. No RuleDecision, hence no
        Violation row and no complaint would be generated downstream.
        """
        decision = rules_engine.evaluate(media_type="text", psynth=0.99, category=None, ip_result=None)
        assert decision is None

    def test_custom_threshold_override(self):
        decision = rules_engine.evaluate(
            media_type="video", psynth=0.5, category=CATEGORY_TELECOM_FMCG_ADS, threshold=0.4,
        )
        assert decision is not None
        assert decision.violation_type == "UNDISCLOSED_SYNTHETIC_AVATAR"
