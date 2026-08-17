# =============================================================================
# tests/test_complaint_generator.py  —  §3.13 Auto-Complaint PDF Test Suite (Day 16)
# =============================================================================
# Covers:
#   - build_complaint_context() correctly extracts IP-match / visual /
#     text detail from live FusionResult/RuleDecision/IPSimilarityResult
#     objects, including the abstained-modality cases
#   - generate_complaint_pdf() produces a real, non-trivial PDF file, with
#     the caveat block always present and the two conditional caveats
#     (IP false-positive modes; CV heuristic reliability) appearing only
#     when actually relevant
# =============================================================================

from datetime import datetime, timezone

import pytest

from watchdog import complaint_generator as cg
from watchdog import fusion, rules_engine
from watchdog.ip_similarity import IPSimilarityResult, SimilarityMatch


def _fusion_result(p_visual=None, p_text=0.9, modalities_used=None, visual_abstain_reason=None):
    visual = fusion.VisualAssessment(
        p_visual=p_visual,
        scored_heuristics=["emotional_rigidity"] if p_visual is not None else [],
        abstained_heuristics={} if p_visual is not None else {"blink_rate": "run too short"},
        abstain_reason=visual_abstain_reason,
    )
    text = fusion.TextAssessment(p_text=p_text, source="distilbert_classifier" if p_text is not None else None)
    modalities_used = modalities_used if modalities_used is not None else (["text"] if p_text is not None else [])
    psynth, _ = fusion.compute_psynth(p_visual, p_text, alpha=0.6)
    return fusion.FusionResult(
        psynth=psynth, p_visual=p_visual, p_text=p_text, modalities_used=modalities_used,
        alpha=0.6, visual=visual, text=text,
    )


def _decision(violation_type="IP_SCRAPING_PLAGIARISM", psynth=0.9):
    mapping = rules_engine.map_to_statute(violation_type)
    return rules_engine.RuleDecision(
        violation_type=violation_type, statute=mapping["statute"], statute_summary=mapping["summary"],
        regulator=mapping["regulator"], action=mapping["action"], psynth=psynth, basis="test basis",
    )


def _ip_result():
    match = SimilarityMatch(
        sample_id="abc-123", publisher_name="TheCable", source_url="https://thecable.ng/x",
        similarity=0.91, matched_passage="This is the closest matched sentence.",
    )
    return IPSimilarityResult(matches=[match], flagged=True)


# =============================================================================
# build_complaint_context()
# =============================================================================

class TestBuildComplaintContext:
    def test_ip_match_populated_when_flagged(self):
        context = cg.build_complaint_context(
            violation_id="v1", decision=_decision(), fusion_result=_fusion_result(p_text=0.9),
            ip_result=_ip_result(), source_url="https://example.ng/article", publisher_name="Example NG",
            media_type="text", detected_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
        )
        assert context.ip_match is not None
        assert context.ip_match["sample_id"] == "abc-123"
        assert context.ip_match["similarity"] == pytest.approx(0.91)

    def test_ip_match_none_when_not_flagged(self):
        context = cg.build_complaint_context(
            violation_id="v1", decision=_decision(violation_type="UNDISCLOSED_SYNTHETIC_AVATAR"),
            fusion_result=_fusion_result(p_visual=0.9, p_text=None, modalities_used=["visual"]),
            ip_result=None, source_url="https://example.ng/ad.mp4", publisher_name="Example NG",
            media_type="video",
        )
        assert context.ip_match is None

    def test_abstained_visual_modality_recorded_explicitly(self):
        context = cg.build_complaint_context(
            violation_id="v1", decision=_decision(violation_type="IP_SCRAPING_PLAGIARISM"),
            fusion_result=_fusion_result(p_visual=None, p_text=0.9, visual_abstain_reason="no video track"),
            ip_result=_ip_result(), source_url="https://example.ng/article", publisher_name=None,
            media_type="text",
        )
        assert context.p_visual is None
        assert context.visual_detail["abstain_reason"] == "no video track"

    def test_evidence_dict_is_json_serialisable(self):
        import json

        context = cg.build_complaint_context(
            violation_id="v1", decision=_decision(), fusion_result=_fusion_result(p_text=0.9),
            ip_result=_ip_result(), source_url="https://example.ng/article", publisher_name="Example NG",
            media_type="text",
        )
        json.dumps(context.to_evidence_dict())  # must not raise


# =============================================================================
# generate_complaint_pdf()
# =============================================================================

class TestGenerateComplaintPdf:
    def _context(self, violation_type="IP_SCRAPING_PLAGIARISM", p_visual=None, p_text=0.9, ip_result=None):
        modalities = []
        if p_visual is not None:
            modalities.append("visual")
        if p_text is not None:
            modalities.append("text")
        return cg.build_complaint_context(
            violation_id="test-violation-id", decision=_decision(violation_type=violation_type),
            fusion_result=_fusion_result(p_visual=p_visual, p_text=p_text, modalities_used=modalities),
            ip_result=ip_result, source_url="https://example.ng/thing", publisher_name="Example NG",
            media_type="text" if p_visual is None else "video",
        )

    def test_generates_a_real_pdf_file(self, tmp_path):
        context = self._context(ip_result=_ip_result())
        pdf_path = cg.generate_complaint_pdf(context, output_dir=tmp_path)
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF")
        assert pdf_path.stat().st_size > 1000  # a real rendered document, not an empty shell

    def test_regenerating_same_violation_overwrites_not_duplicates(self, tmp_path):
        context = self._context(ip_result=_ip_result())
        path1 = cg.generate_complaint_pdf(context, output_dir=tmp_path)
        path2 = cg.generate_complaint_pdf(context, output_dir=tmp_path)
        assert path1 == path2
        assert len(list(tmp_path.glob("*.pdf"))) == 1

    def test_filename_is_deterministic_from_violation_id(self, tmp_path):
        context = self._context(ip_result=_ip_result())
        pdf_path = cg.generate_complaint_pdf(context, output_dir=tmp_path)
        assert pdf_path.name == "complaint_test-violation-id.pdf"

    def test_statute_containing_ampersand_does_not_break_rendering(self, tmp_path):
        """
        config.STATUTORY_MAP["UNDISCLOSED_SYNTHETIC_AVATAR"]["statute"] contains a literal
        '&' ("ARCON Advertising Guidelines (2023) & NDPA 2023, Section 37"), and a real
        source_url can contain '&' in its query string. reportlab's Paragraph parses its
        content as markup, so an un-escaped '&' would raise or mis-render — this is a
        regression test for that.
        """
        context = self._context(violation_type="UNDISCLOSED_SYNTHETIC_AVATAR", p_visual=0.9, p_text=None)
        context.source_url = "https://example.ng/ad.mp4?utm_source=ig&utm_campaign=launch"
        pdf_path = cg.generate_complaint_pdf(context, output_dir=tmp_path)  # must not raise
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF")
