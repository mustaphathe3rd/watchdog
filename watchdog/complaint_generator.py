# =============================================================================
# watchdog/complaint_generator.py  —  Day 16-20: Auto-Complaint PDF (§3.13)
# =============================================================================
# PURPOSE:
#   Turns a watchdog.rules_engine.RuleDecision + watchdog.fusion.FusionResult
#   (+ an optional watchdog.ip_similarity.IPSimilarityResult) into a formal
#   PDF complaint via ReportLab — §3.13's "Document Generation" deliverable,
#   and the project's stated novel contribution (Objective 3).
#
# "STATE WHAT WAS ACTUALLY OBSERVED, NOT JUST A PROBABILITY":
#   A bare "Psynth = 0.91" is not something a regulator or a rights-holder
#   can act on. Every section below is built to cite specifics: the
#   classifier's own confidence and label (not just whether it "voted" AI),
#   the exact protected work + quoted passage an IP match points at, which
#   Stage 4 CV heuristics contributed a score vs. which abstained and why,
#   the forensic detection timestamp, and the source URL. Where a modality
#   abstained, that is stated explicitly in its own paragraph — silently
#   omitting a missing modality would read as "we checked and found
#   nothing", which is a different and misleading claim from "we couldn't
#   check". See watchdog/fusion.py's module docstring for why abstention is
#   never converted into evidence upstream of this module either.
#
# THE CAVEAT BLOCK (required, not optional):
#   §3.13 already requires a human officer to review every auto-drafted
#   complaint before dispatch (reports.status: PENDING_REVIEW -> DISPATCHED,
#   db_connector.py). This module puts WHY in the document itself, at the
#   very top, in a visually distinct block — not a legal boilerplate
#   footnote nobody reads:
#     - ALWAYS: this is an unreviewed machine draft, do not dispatch as-is.
#     - IP_SCRAPING_PLAGIARISM only: the two CONFIRMED false-positive modes
#       found in the 2026-08-04 threshold validation (same-event convergent
#       journalism quoting an official statement verbatim; same-publisher
#       beat clustering) — see validate_ip_similarity_threshold.py and
#       docs/PROJECT_CONTEXT.md §2's threshold-validation entry.
#     - Whenever visual evidence contributed to Psynth: the CV heuristics'
#       own known lack of clean discrimination (docs/PROJECT_CONTEXT.md §4's
#       full-corpus evaluation) — the reviewer should treat it as weak
#       corroboration, not proof. This is not a gap awaiting the Days 10-15
#       classifier: that classifier was built and trained across three
#       schedules but never cleared its pre-set FF++ credibility gate, and
#       was closed 2026-08-12 as a documented negative result (see
#       docs/PROJECT_CONTEXT.md §4's "CV classifier training" entry) — the
#       heuristics below are this project's permanent visual-evidence source
#       for the rest of its timeline.
# =============================================================================

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape as _xml_escape

logger = logging.getLogger(__name__)


def _esc(value) -> str:
    """
    Escapes &, <, > before handing text to a reportlab Paragraph — Paragraph
    parses its content as a small XML/HTML-like markup language, so any
    dynamic value (a statute name, a source URL with '&param=' in its query
    string, a matched passage) that happens to contain one of those
    characters would otherwise either break the parser or render wrong.
    Every value plugged into a Paragraph below is either this project's own
    literal, hand-written text (never escaped — it may legitimately use
    "<b>") or comes from here.
    """
    return _xml_escape(str(value))


# =============================================================================
# Context — the flat, serialisable shape this module actually renders
# =============================================================================
# Deliberately not typed against watchdog.fusion.FusionResult /
# watchdog.rules_engine.RuleDecision directly as the render function's
# input: build_complaint_context() below is the one place that translates
# those (richer, nested) objects into this flat shape, so generate_complaint_pdf()
# and its tests can construct/assert against plain data without needing a
# live FusionResult/RuleDecision.

@dataclass
class ComplaintContext:
    violation_id: str
    violation_type: str
    statute: str
    statute_summary: str
    regulator: str
    action: str
    basis: str                       # why this violation_type was chosen (rules_engine)
    psynth: float
    p_visual: Optional[float]
    p_text: Optional[float]
    modalities_used: list
    fusion_alpha: float
    source_url: str
    publisher_name: Optional[str]
    media_type: str
    detected_at: datetime
    visual_detail: dict = field(default_factory=dict)   # {"scored_heuristics", "abstained_heuristics", "abstain_reason"}
    text_detail: dict = field(default_factory=dict)      # {"source", "predicted_label", "p_synthetic", "abstain_reason"}
    ip_match: Optional[dict] = None                       # {"sample_id","publisher_name","source_url","similarity","matched_passage"}

    def to_evidence_dict(self) -> dict:
        """JSON-serialisable evidence snapshot for db_connector.Violation.evidence_json (§3.13 audit trail)."""
        d = asdict(self)
        d["detected_at"] = self.detected_at.isoformat()
        return d


def build_complaint_context(*, violation_id: str, decision, fusion_result, ip_result=None,
                             source_url: str, publisher_name: Optional[str], media_type: str,
                             detected_at: Optional[datetime] = None) -> ComplaintContext:
    """
    Assembles a ComplaintContext from the live objects the pipeline actually
    produces: a watchdog.rules_engine.RuleDecision, a watchdog.fusion.FusionResult,
    and (for text) an optional watchdog.ip_similarity.IPSimilarityResult.
    """
    if detected_at is None:
        detected_at = datetime.now()

    visual_detail = {
        "scored_heuristics": list(fusion_result.visual.scored_heuristics),
        "abstained_heuristics": dict(fusion_result.visual.abstained_heuristics),
        "abstain_reason": fusion_result.visual.abstain_reason,
    }
    text_detail = {
        "source": fusion_result.text.source,
        "p_synthetic": fusion_result.text.p_text,
        "abstain_reason": fusion_result.text.abstain_reason,
    }

    ip_match = None
    if ip_result is not None and getattr(ip_result, "flagged", False) and ip_result.matches:
        top = ip_result.matches[0]
        ip_match = {
            "sample_id": top.sample_id,
            "publisher_name": top.publisher_name,
            "source_url": top.source_url,
            "similarity": top.similarity,
            "matched_passage": top.matched_passage,
        }

    return ComplaintContext(
        violation_id=violation_id,
        violation_type=decision.violation_type,
        statute=decision.statute,
        statute_summary=decision.statute_summary,
        regulator=decision.regulator,
        action=decision.action,
        basis=decision.basis,
        psynth=fusion_result.psynth,
        p_visual=fusion_result.p_visual,
        p_text=fusion_result.p_text,
        modalities_used=list(fusion_result.modalities_used),
        fusion_alpha=fusion_result.alpha,
        source_url=source_url,
        publisher_name=publisher_name,
        media_type=media_type,
        detected_at=detected_at,
        visual_detail=visual_detail,
        text_detail=text_detail,
        ip_match=ip_match,
    )


# =============================================================================
# Caveat text — see module docstring
# =============================================================================

CAVEAT_ALWAYS = (
    "This complaint was drafted automatically by the Naija AI Ethics Watchdog pipeline and has "
    "NOT been reviewed by a human. Per §3.13 of the underlying methodology, a qualified human "
    "reviewer MUST examine this document and the evidence summarised below before it is "
    "dispatched to the named regulator (reports.status: PENDING_REVIEW). Do not forward, "
    "publish, or act on this document until that review is complete."
)

CAVEAT_IP_FALSE_POSITIVE_MODES = (
    "This complaint alleges IP_SCRAPING_PLAGIARISM on a whole-document cosine-similarity match "
    "against a protected work. A validation study of this exact matcher "
    "(2026-08-04, validate_ip_similarity_threshold.py) found two CONFIRMED false-positive modes "
    "that can exceed the matching threshold with NO actual copying: (1) same-event convergent "
    "journalism, where two outlets independently quote the same official statement verbatim; "
    "and (2) same-publisher beat clustering, where several articles from one outlet/desk share "
    "topic, house style, and framing without either being derived from the other. Before "
    "treating the matched passage below as evidence of copying, the reviewer should confirm it "
    "reflects the respondent's own original wording, not a shared quotation or beat-level "
    "convergence."
)

CAVEAT_CV_HEURISTIC_RELIABILITY = (
    "This complaint's visual (CV) evidence score is an average of whichever Stage 4 heuristics "
    "produced a score on this clip (see ‘Visual Evidence’ below); heuristics that abstained "
    "are excluded from that average, not counted as zero. A full-corpus evaluation "
    "(docs/PROJECT_CONTEXT.md §4) found that none of the four heuristics cleanly discriminates "
    "authentic from synthetic video on this project's own matched-subset comparison — one "
    "produced no usable data at all, and the others showed heavy overlap or ran opposite to the "
    "assumption behind them. A trained CV classifier intended to replace these heuristics was "
    "built and evaluated but never cleared its own pre-set credibility gate (docs/"
    "PROJECT_CONTEXT.md §4); these heuristics remain this project's permanent visual-evidence "
    "source. Treat this score as weak, uncalibrated corroboration only — not standalone proof."
)


# =============================================================================
# PDF rendering
# =============================================================================

def generate_complaint_pdf(context: ComplaintContext, output_dir: Optional[Path] = None) -> Path:
    """
    Renders `context` to a PDF at `output_dir / complaint_<violation_id>.pdf`
    (config.REPORTS_DIR by default). Deterministic filename from
    violation_id — re-running for the same violation overwrites the same
    file rather than accumulating duplicates (CLAUDE.md rule 2); the
    violation itself (the immutable evidentiary record) never changes, only
    its rendering can be regenerated.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    if output_dir is None:
        from config import REPORTS_DIR
        output_dir = REPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"complaint_{context.violation_id}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ComplaintTitle", parent=styles["Title"], fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], leading=14)
    mono_small = ParagraphStyle("MonoSmall", parent=styles["BodyText"], fontName="Courier", fontSize=8, leading=10)
    caveat_style = ParagraphStyle("Caveat", parent=styles["BodyText"], leading=13, textColor=colors.HexColor("#5c1a00"))

    def caveat_block(paragraphs: list) -> Table:
        """A visually distinct, bordered/shaded block — the human-review caveat must not read as ordinary body text."""
        flowables = [Paragraph("⚠ HUMAN REVIEW REQUIRED BEFORE DISPATCH", ParagraphStyle(
            "CaveatHeading", parent=styles["Heading3"], textColor=colors.HexColor("#5c1a00"), spaceAfter=4))]
        for p in paragraphs:
            flowables.append(Paragraph(p, caveat_style))
            flowables.append(Spacer(1, 4))
        table = Table([[flowables]], colWidths=[17 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdecea")),
            ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#b3261e")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return table

    label_cell_style = ParagraphStyle("KVLabel", parent=body, fontName="Helvetica-Bold")

    def kv_table(rows: list) -> Table:
        """
        Wraps every cell in its own Paragraph rather than passing bare
        strings. A bare string in a reportlab Table cell doesn't reliably
        wrap to the column width — long values (a full statute name, the
        `basis` sentence, a long source URL) were overflowing past the
        column instead of wrapping, and long labels (e.g. "Detected at
        (forensic timestamp):") were overlapping the row below them because
        the row height was computed from the unwrapped label. Paragraph
        cells make reportlab compute wrapped height correctly for both.
        """
        wrapped = [[Paragraph(_esc(label), label_cell_style), Paragraph(_esc(value), body)]
                   for label, value in rows]
        table = Table(wrapped, colWidths=[5 * cm, 12 * cm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        return table

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("AI ETHICS WATCHDOG — AUTOMATED COMPLIANCE COMPLAINT (DRAFT)", title_style))
    story.append(Paragraph(f"Violation ID: {context.violation_id}", body))
    story.append(Spacer(1, 8))

    # ── Caveat block (always first, prominent) ──────────────────────────
    caveat_paragraphs = [CAVEAT_ALWAYS]
    if context.violation_type == "IP_SCRAPING_PLAGIARISM":
        caveat_paragraphs.append(CAVEAT_IP_FALSE_POSITIVE_MODES)
    if "visual" in context.modalities_used:
        caveat_paragraphs.append(CAVEAT_CV_HEURISTIC_RELIABILITY)
    story.append(caveat_block(caveat_paragraphs))
    story.append(Spacer(1, 12))

    # ── Regulator / statute ──────────────────────────────────────────────
    story.append(Paragraph("Statutory Basis and Routing", h2))
    story.append(kv_table([
        ["Violation type:", context.violation_type],
        ["Statute cited:", context.statute],
        ["Regulator:", context.regulator],
        ["Recommended action:", context.action],
        ["Basis:", context.basis],
    ]))
    story.append(Paragraph(_esc(context.statute_summary), body))

    # ── Detection summary ─────────────────────────────────────────────────
    story.append(Paragraph("Detection Summary", h2))
    p_visual_str = f"{context.p_visual:.4f}" if context.p_visual is not None else "N/A — no visual evidence (see below)"
    p_text_str = f"{context.p_text:.4f}" if context.p_text is not None else "N/A — no text evidence (see below)"
    story.append(kv_table([
        ["Psynth (fused score):", f"{context.psynth:.4f}"],
        ["Modalities used:", ", ".join(context.modalities_used) or "none"],
        ["Fusion alpha (§3.11):", f"{context.fusion_alpha:.2f}"],
        ["p_visual:", p_visual_str],
        ["p_text:", p_text_str],
        ["Source URL:", context.source_url],
        ["Publisher:", context.publisher_name or "Unknown"],
        ["Media type:", context.media_type],
        ["Detected at (forensic timestamp):", context.detected_at.isoformat()],
    ]))

    # ── Evidentiary detail: "what was actually observed" ────────────────
    story.append(Paragraph("Evidentiary Detail", h2))

    story.append(Paragraph("<b>Text evidence:</b>", body))
    if context.p_text is not None:
        pred = "SYNTHETIC" if context.p_text >= 0.5 else "AUTHENTIC"
        story.append(Paragraph(_esc(
            f"The fine-tuned classifier ({context.text_detail.get('source')}) scored this text "
            f"P(synthetic) = {context.p_text:.4f}, predicted label {pred}."), body))
    else:
        story.append(Paragraph(_esc(
            f"No text evidence was used — the text modality abstained. Reason: "
            f"{context.text_detail.get('abstain_reason') or 'not recorded'}."), body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Visual evidence:</b>", body))
    if context.p_visual is not None:
        scored = context.visual_detail.get("scored_heuristics", [])
        abstained = context.visual_detail.get("abstained_heuristics", {})
        story.append(Paragraph(_esc(
            f"Average heuristic score {context.p_visual:.4f}, from {len(scored)} of 4 Stage 4 "
            f"heuristic(s) that produced a score: {', '.join(scored) or 'none'}."), body))
        if abstained:
            abstained_lines = "; ".join(f"{name} ({reason})" for name, reason in abstained.items())
            story.append(Paragraph(_esc(f"Abstained (excluded from the average): {abstained_lines}."), body))
    else:
        story.append(Paragraph(_esc(
            f"No visual evidence was used — the visual modality abstained. Reason: "
            f"{context.visual_detail.get('abstain_reason') or 'not recorded'}."), body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Matched protected work (IP similarity):</b>", body))
    if context.ip_match:
        m = context.ip_match
        story.append(kv_table([
            ["Protected work ID:", m["sample_id"]],
            ["Publisher:", m["publisher_name"]],
            ["Source URL:", m["source_url"]],
            ["Cosine similarity:", f"{m['similarity']:.4f}"],
        ]))
        story.append(Paragraph("Closest matched passage in the protected work:", body))
        story.append(Paragraph(f"“{_esc(m['matched_passage'])}”", ParagraphStyle(
            "Quote", parent=body, leftIndent=14, textColor=colors.HexColor("#333333"))))
    else:
        story.append(Paragraph("Not applicable — this violation type is not IP_SCRAPING_PLAGIARISM, "
                                "or no IP-similarity check was run against this content.", body))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Generated by main_pipeline.py / watchdog.complaint_generator. Report status begins as "
        "PENDING_REVIEW (see watchdog.db_connector.Report) and only becomes DISPATCHED after "
        "human sign-off.", ParagraphStyle("Footer", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                             leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    doc.build(story)

    logger.info(f"Generated complaint PDF: {pdf_path}")
    return pdf_path
