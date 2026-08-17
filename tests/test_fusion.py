# =============================================================================
# tests/test_fusion.py  —  §3.11 Weighted Fusion Test Suite (Day 16)
# =============================================================================
# THE ONE PROPERTY EVERY TEST GROUP BELOW IS ULTIMATELY CHECKING:
#   An abstained heuristic/modality is EXCLUDED from the arithmetic, never
#   substituted with 0 (or any other number). See watchdog/fusion.py's
#   module docstring for why this matters on the real corpus — blink-rate
#   abstention correlates near-perfectly with label, so treating "abstained"
#   as evidence would launder a genre confound into Psynth.
# =============================================================================

from unittest.mock import patch

import pytest

from watchdog import fusion
from watchdog.cv_engine import ClipHeuristics, HeuristicResult


def _heuristic(score, abstain_reason=None):
    return HeuristicResult(score=score, raw={}, abstain_reason=abstain_reason, time_s=0.01)


def _clip_heuristics(blink=None, rigidity=None, viseme=None, blending=None):
    return ClipHeuristics(
        sample_path="fake.mp4",
        blink_rate=blink or _heuristic(None, "abstained"),
        emotional_rigidity=rigidity or _heuristic(None, "abstained"),
        viseme_mismatch=viseme or _heuristic(None, "abstained"),
        blending_artifacts=blending or _heuristic(None, "abstained"),
        tracking_time_s=1.0,
    )


# =============================================================================
# combine_cv_heuristics()
# =============================================================================

class TestCombineCVHeuristics:
    def test_none_clip_heuristics_abstains(self):
        result = fusion.combine_cv_heuristics(None)
        assert result.p_visual is None
        assert "no video" in result.abstain_reason.lower()

    def test_all_four_abstained_abstains(self):
        clip = _clip_heuristics()  # all default to abstained
        result = fusion.combine_cv_heuristics(clip)
        assert result.p_visual is None
        assert result.scored_heuristics == []
        assert len(result.abstained_heuristics) == 4

    def test_single_scored_heuristic_is_not_pulled_toward_zero_by_abstentions(self):
        """
        THE key abstention-propagation test: blink_rate, viseme_mismatch,
        and blending_artifacts all abstain; only emotional_rigidity scores
        0.9. p_visual must be exactly 0.9 — not averaged with three implicit
        zeros (which would give 0.225).
        """
        clip = _clip_heuristics(rigidity=_heuristic(0.9))
        result = fusion.combine_cv_heuristics(clip)
        assert result.p_visual == pytest.approx(0.9)
        assert result.scored_heuristics == ["emotional_rigidity"]
        assert set(result.abstained_heuristics) == {"blink_rate", "viseme_mismatch", "blending_artifacts"}

    def test_all_four_scored_averages_correctly(self):
        clip = _clip_heuristics(
            blink=_heuristic(0.2), rigidity=_heuristic(0.4),
            viseme=_heuristic(0.6), blending=_heuristic(0.8),
        )
        result = fusion.combine_cv_heuristics(clip)
        assert result.p_visual == pytest.approx(0.5)
        assert result.abstain_reason is None
        assert len(result.scored_heuristics) == 4


# =============================================================================
# assess_text()
# =============================================================================

class TestAssessText:
    def test_empty_text_abstains_without_calling_classifier(self):
        result = fusion.assess_text("")
        assert result.p_text is None
        assert "no text" in result.abstain_reason.lower()

    def test_none_text_abstains(self):
        result = fusion.assess_text(None)
        assert result.p_text is None

    @patch("watchdog.nlp_engine.analyze_text")
    @patch("watchdog.nlp_engine.classify_text")
    def test_classifier_abstain_propagates_as_missing(self, mock_classify, mock_analyze):
        from watchdog.nlp_engine import TextClassification

        mock_classify.return_value = TextClassification(
            p_synthetic=None, predicted_label=None, abstain_reason="no trained classifier checkpoint"
        )
        mock_analyze.return_value = {"heuristic_vote": "UNCERTAIN"}

        result = fusion.assess_text("some real text here")
        assert result.p_text is None
        assert result.abstain_reason == "no trained classifier checkpoint"

    @patch("watchdog.nlp_engine.analyze_text")
    @patch("watchdog.nlp_engine.classify_text")
    def test_classifier_score_used_verbatim_as_p_text(self, mock_classify, mock_analyze):
        from watchdog.nlp_engine import TextClassification

        mock_classify.return_value = TextClassification(p_synthetic=0.93, predicted_label="SYNTHETIC")
        mock_analyze.return_value = {"heuristic_vote": "HUMAN"}  # deliberately disagrees with the classifier

        result = fusion.assess_text("some real text here")
        assert result.p_text == pytest.approx(0.93)
        assert result.source == "distilbert_classifier"
        # The heuristic vote is attached for diagnostic/logging purposes only —
        # it must NOT override or blend into p_text, even when it disagrees.
        assert result.heuristic_diagnostic == {"heuristic_vote": "HUMAN"}

    @patch("watchdog.nlp_engine.analyze_text", side_effect=RuntimeError("distilgpt2 not downloaded"))
    @patch("watchdog.nlp_engine.classify_text")
    def test_heuristic_diagnostic_failure_never_blocks_the_real_decision(self, mock_classify, mock_analyze):
        from watchdog.nlp_engine import TextClassification

        mock_classify.return_value = TextClassification(p_synthetic=0.7, predicted_label="SYNTHETIC")

        result = fusion.assess_text("some real text here")
        assert result.p_text == pytest.approx(0.7)
        assert result.heuristic_diagnostic is None  # failed, logged, not fatal


# =============================================================================
# compute_psynth()  — the missing-modality fallback contract
# =============================================================================

class TestComputePsynth:
    def test_both_present_uses_weighted_fusion(self):
        psynth, modalities = fusion.compute_psynth(p_visual=0.8, p_text=0.4, alpha=0.6)
        assert psynth == pytest.approx(0.6 * 0.8 + 0.4 * 0.4)
        assert modalities == ["visual", "text"]

    def test_visual_only_is_unweighted_not_multiplied_by_alpha(self):
        """
        A text-only or visual-only Psynth must equal that modality's own
        score exactly — NOT alpha*p_visual (which would silently treat the
        missing modality as counter-evidence of 0).
        """
        psynth, modalities = fusion.compute_psynth(p_visual=0.8, p_text=None, alpha=0.6)
        assert psynth == pytest.approx(0.8)
        assert modalities == ["visual"]

    def test_text_only_is_unweighted(self):
        psynth, modalities = fusion.compute_psynth(p_visual=None, p_text=0.4, alpha=0.6)
        assert psynth == pytest.approx(0.4)
        assert modalities == ["text"]

    def test_neither_present_abstains(self):
        psynth, modalities = fusion.compute_psynth(p_visual=None, p_text=None, alpha=0.6)
        assert psynth is None
        assert modalities == []

    def test_defaults_to_config_fusion_alpha(self):
        from config import FUSION_ALPHA

        psynth, _ = fusion.compute_psynth(p_visual=1.0, p_text=0.0)
        assert psynth == pytest.approx(FUSION_ALPHA)


# =============================================================================
# run_fusion()  — end-to-end orchestration
# =============================================================================

class TestRunFusion:
    @patch("watchdog.nlp_engine.analyze_text")
    @patch("watchdog.nlp_engine.classify_text")
    def test_text_only_sample_produces_text_only_decision(self, mock_classify, mock_analyze):
        """A text-only sample (clip_heuristics=None) must be a valid, non-degraded decision on its own."""
        from watchdog.nlp_engine import TextClassification

        mock_classify.return_value = TextClassification(p_synthetic=0.9, predicted_label="SYNTHETIC")
        mock_analyze.return_value = {}

        result = fusion.run_fusion(clip_heuristics=None, text="some article text")
        assert result.psynth == pytest.approx(0.9)
        assert result.p_visual is None
        assert result.modalities_used == ["text"]
        assert result.abstain_reason is None

    def test_no_video_no_text_fully_abstains(self):
        result = fusion.run_fusion(clip_heuristics=None, text=None)
        assert result.psynth is None
        assert result.abstain_reason is not None

    @patch("watchdog.nlp_engine.analyze_text")
    @patch("watchdog.nlp_engine.classify_text")
    def test_video_with_partial_heuristic_abstention_and_text_fuses_both(self, mock_classify, mock_analyze):
        from watchdog.nlp_engine import TextClassification

        mock_classify.return_value = TextClassification(p_synthetic=0.4, predicted_label="AUTHENTIC")
        mock_analyze.return_value = {}

        clip = _clip_heuristics(rigidity=_heuristic(0.8))  # only one heuristic scores
        result = fusion.run_fusion(clip_heuristics=clip, text="some article text", alpha=0.6)

        assert result.p_visual == pytest.approx(0.8)
        assert result.p_text == pytest.approx(0.4)
        assert result.psynth == pytest.approx(0.6 * 0.8 + 0.4 * 0.4)
        assert result.modalities_used == ["visual", "text"]
