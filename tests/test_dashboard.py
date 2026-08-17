# =============================================================================
# tests/test_dashboard.py  —  Days 21-22 Flask Admin Dashboard Test Suite
# =============================================================================
# Covers:
#   - / : empty-state honesty, and rendering a real violation row
#   - /violations/<id>: the detail view's two hard requirements —
#       (1) an abstained modality/heuristic is shown with its stored reason,
#           never as a blank cell or a 0.00 score
#       (2) the two known false-positive-mode caveats (CV heuristic
#           reliability; IP same-event/same-publisher confounds) render
#           VERBATIM from watchdog.complaint_generator's own constants, only
#           when actually relevant to the violation being viewed
#     ...plus 404 handling and graceful degradation on missing evidence_json.
#   - /reports/<id>: serves a real PDF, 404s when there is no report row or
#     the file is missing on disk.
#   - /upload: form rendering, both submission modes (mocked at the
#     main_pipeline/scraper boundary — this suite never runs the real CV/NLP
#     models), and that latency gets logged to an INJECTED results_dir, never
#     the project's real results/ directory (create_app(results_dir=...) —
#     see dashboard/app.py's docstring for why that matters).
#
# Uses the exact same in-memory-SQLite-per-test pattern as
# tests/conftest.py, and the same build_complaint_context()/RuleDecision
# construction pattern as tests/test_complaint_generator.py, so the
# evidence_json fixtures here are shaped exactly like what main_pipeline.py
# actually writes — not a hand-rolled approximation of it.
# =============================================================================

import json
import uuid
from pathlib import Path

import pytest

from watchdog.db_connector import (
    get_engine, get_session_factory, init_db,
    log_media_ingestion, record_violation, record_report,
)
from watchdog import fusion, rules_engine
from watchdog.complaint_generator import (
    build_complaint_context, generate_complaint_pdf,
    CAVEAT_CV_HEURISTIC_RELIABILITY, CAVEAT_IP_FALSE_POSITIVE_MODES,
)
from watchdog.ip_similarity import IPSimilarityResult, SimilarityMatch

from dashboard.app import create_app


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def engine():
    eng = get_engine("sqlite:///:memory:")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    Session = get_session_factory(engine)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture()
def app(engine, tmp_path):
    application = create_app(engine=engine, results_dir=tmp_path / "results")
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


# =============================================================================
# Fixture-building helpers — mirror tests/test_complaint_generator.py's
# _fusion_result()/_decision() so evidence_json here matches production shape.
# =============================================================================

def _fusion_result(p_visual=None, p_text=0.9, modalities_used=None,
                    scored_heuristics=None, abstained_heuristics=None, visual_abstain_reason=None):
    visual = fusion.VisualAssessment(
        p_visual=p_visual,
        scored_heuristics=scored_heuristics if scored_heuristics is not None else (
            ["emotional_rigidity", "blending_artifacts"] if p_visual is not None else []),
        abstained_heuristics=abstained_heuristics if abstained_heuristics is not None else (
            {"blink_rate": "longest contiguous face run (2.4s) is below the 10.0s minimum for blink-rate analysis"}
        ),
        abstain_reason=visual_abstain_reason,
    )
    text = fusion.TextAssessment(p_text=p_text, source="distilbert_classifier" if p_text is not None else None,
                                  abstain_reason=None if p_text is not None else "no text supplied")
    if modalities_used is None:
        modalities_used = []
        if p_visual is not None:
            modalities_used.append("visual")
        if p_text is not None:
            modalities_used.append("text")
    psynth, _ = fusion.compute_psynth(p_visual, p_text, alpha=0.6)
    return fusion.FusionResult(
        psynth=psynth, p_visual=p_visual, p_text=p_text, modalities_used=modalities_used,
        alpha=0.6, visual=visual, text=text,
    )


def _decision(violation_type="UNDISCLOSED_SYNTHETIC_AVATAR", psynth=0.9):
    mapping = rules_engine.map_to_statute(violation_type)
    return rules_engine.RuleDecision(
        violation_type=violation_type, statute=mapping["statute"], statute_summary=mapping["summary"],
        regulator=mapping["regulator"], action=mapping["action"], psynth=psynth, basis="test basis",
    )


def _ip_result():
    match = SimilarityMatch(
        sample_id="protected-abc123", publisher_name="TheCable", source_url="https://thecable.ng/some-article",
        similarity=0.9102, matched_passage="This is the single closest matched sentence in the protected work.",
    )
    return IPSimilarityResult(matches=[match], flagged=True)


def _make_violation(session, *, violation_type="UNDISCLOSED_SYNTHETIC_AVATAR", p_visual=None, p_text=0.9,
                     ip_result=None, with_evidence=True, with_report=True, pdf_dir=None,
                     media_type="video", publisher_name="Test Publisher Ltd",
                     source_url="https://example.com/ad.mp4"):
    """Inserts one Violation (+ optional Report/PDF) shaped exactly like main_pipeline.py's own output."""
    media_log = log_media_ingestion(session, source_url=source_url, media_type=media_type)
    fusion_result = _fusion_result(p_visual=p_visual, p_text=p_text)
    decision = _decision(violation_type=violation_type, psynth=fusion_result.psynth or 0.9)
    violation_id = uuid.uuid4()

    evidence_json = None
    if with_evidence:
        context = build_complaint_context(
            violation_id=str(violation_id), decision=decision, fusion_result=fusion_result,
            ip_result=ip_result, source_url=source_url, publisher_name=publisher_name, media_type=media_type,
        )
        evidence_json = json.dumps(context.to_evidence_dict())

    violation = record_violation(
        session=session, media_log_entry=media_log, psynth_score=decision.psynth,
        p_visual=p_visual, p_text=p_text, violation_type=violation_type,
        statute_cited=decision.statute, statute_summary=decision.statute_summary, regulator=decision.regulator,
        source_url=source_url, publisher_name=publisher_name, evidence_json=evidence_json,
        violation_id=violation_id,
    )

    if with_report:
        pdf_dir = pdf_dir or Path("/tmp")
        if with_evidence:
            pdf_path = generate_complaint_pdf(context, output_dir=pdf_dir)
        else:
            # No evidence_json fixture requested, but a report row is still
            # needed — write a trivial placeholder file so /reports/<id>
            # has something real to serve.
            pdf_path = Path(pdf_dir) / f"complaint_{violation_id}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4 test placeholder")
        record_report(session, violation, pdf_path)

    return violation


# =============================================================================
# / — violations list
# =============================================================================

class TestIndex:
    def test_empty_state_is_honest(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"No violations recorded yet" in resp.data
        # No fabricated scale anywhere in the empty state.
        assert b"124" not in resp.data

    def test_lists_a_real_violation(self, client, session):
        _make_violation(session, p_visual=0.8, p_text=None)
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"UNDISCLOSED_SYNTHETIC_AVATAR" in resp.data
        assert b"Test Publisher Ltd" in resp.data
        assert b"PENDING_REVIEW" in resp.data


# =============================================================================
# /violations/<id>
# =============================================================================

class TestViolationDetail:
    def test_404_for_unknown_id(self, client):
        resp = client.get(f"/violations/{uuid.uuid4()}")
        assert resp.status_code == 404
        assert b"NOT FOUND" in resp.data

    def test_404_for_malformed_id(self, client):
        # get_violation_by_id() returns None (not a raise) for a syntactically
        # invalid UUID (db_connector.py's own documented bug fix) — the route
        # must still 404 cleanly, not 500.
        resp = client.get("/violations/not-a-uuid")
        assert resp.status_code == 404

    def test_visual_abstention_shows_reason_not_blank_or_zero(self, client, session):
        """The one hard requirement the spec calls out by name."""
        v = _make_violation(session, p_visual=None, p_text=0.7)
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        assert resp.status_code == 200
        assert "ABSTAINED" in body
        # The per-heuristic reason string must appear verbatim, not be
        # summarised away or dropped.
        assert "longest contiguous face run (2.4s) is below the 10.0s minimum" in body
        # And it must never be misrepresented as a numeric zero score —
        # nowhere on the page should the abstained p_visual render as 0.0000.
        assert "visual" in body.lower()
        assert "0.0000" not in body

    def test_text_abstention_shows_reason(self, client, session):
        v = _make_violation(session, p_visual=0.8, p_text=None)
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        assert "ABSTAINED" in body
        assert "no text supplied" in body

    def test_scored_visual_shows_average_and_per_heuristic_breakdown(self, client, session):
        v = _make_violation(session, p_visual=0.77, p_text=0.6)
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        assert "0.7700" in body
        assert "emotional_rigidity" in body
        assert "blending_artifacts" in body
        assert "SCORED" in body
        # The abstained heuristic (blink_rate, in every fixture here) is
        # still shown alongside the scored ones, with its reason.
        assert "blink_rate" in body

    def test_cv_caveat_shown_when_visual_used(self, client, session):
        v = _make_violation(session, p_visual=0.8, p_text=None)
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        # Jinja2 autoescapes the constant's apostrophe ("complaint's") to
        # &#39; on render, so the assertion below picks an apostrophe-free
        # slice of the real constant rather than a byte-for-byte prefix.
        assert "visual (CV) evidence score is an average of whichever Stage 4 heuristics" in body

    def test_cv_caveat_not_shown_when_text_only(self, client, session):
        v = _make_violation(session, p_visual=None, p_text=0.9, violation_type="IP_SCRAPING_PLAGIARISM",
                             ip_result=_ip_result(), media_type="text")
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        # Same apostrophe-escaping note as test_cv_caveat_shown_when_visual_used
        # above — a raw-prefix check here would pass trivially either way.
        assert "visual (CV) evidence score is an average of whichever Stage 4 heuristics" not in body

    def test_ip_caveat_and_match_shown_for_ip_violation(self, client, session):
        v = _make_violation(session, p_visual=None, p_text=0.93, violation_type="IP_SCRAPING_PLAGIARISM",
                             ip_result=_ip_result(), media_type="text")
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        assert CAVEAT_IP_FALSE_POSITIVE_MODES[:60] in body
        assert "protected-abc123" in body
        assert "0.9102" in body
        assert "This is the single closest matched sentence" in body

    def test_ip_caveat_not_shown_for_non_ip_violation(self, client, session):
        v = _make_violation(session, p_visual=0.8, p_text=None, violation_type="DEEPFAKE_IMPERSONATION")
        resp = client.get(f"/violations/{v.id}")
        body = resp.data.decode()
        assert CAVEAT_IP_FALSE_POSITIVE_MODES[:60] not in body

    def test_missing_evidence_json_falls_back_gracefully(self, client, session):
        v = _make_violation(session, p_visual=0.8, p_text=None, with_evidence=False)
        resp = client.get(f"/violations/{v.id}")
        assert resp.status_code == 200
        body = resp.data.decode()
        # Top-level Violation columns still render even with no evidence_json.
        assert "UNDISCLOSED_SYNTHETIC_AVATAR" in body
        assert "No detailed evidence_json is stored" in body


# =============================================================================
# /reports/<id>
# =============================================================================

class TestServeReport:
    def test_serves_pdf(self, client, session, tmp_path):
        v = _make_violation(session, p_visual=0.8, p_text=None, pdf_dir=tmp_path)
        resp = client.get(f"/reports/{v.id}")
        assert resp.status_code == 200
        assert resp.mimetype == "application/pdf"
        assert resp.data.startswith(b"%PDF")

    def test_404_when_no_report_row(self, client, session):
        v = _make_violation(session, p_visual=0.8, p_text=None, with_report=False)
        resp = client.get(f"/reports/{v.id}")
        assert resp.status_code == 404

    def test_404_for_unknown_violation(self, client):
        resp = client.get(f"/reports/{uuid.uuid4()}")
        assert resp.status_code == 404


# =============================================================================
# /upload
# =============================================================================

class TestUpload:
    def test_get_renders_form(self, client):
        resp = client.get("/upload")
        assert resp.status_code == 200
        assert b"Submit Content" in resp.data
        assert b"sample_id" in resp.data

    def test_sample_id_not_found_flashes_error(self, client, monkeypatch):
        import pandas as pd
        import watchdog.dataset_manager as dataset_manager

        monkeypatch.setattr(dataset_manager, "load_manifest",
                             lambda: pd.DataFrame(columns=dataset_manager.MANIFEST_COLUMNS))

        resp = client.post("/upload", data={"mode": "sample_id", "sample_id": "does-not-exist"},
                            follow_redirects=True)
        assert resp.status_code == 200
        assert b"not found in data/manifest.csv" in resp.data

    def test_sample_id_success_no_violation_and_latency_logged(self, client, app, monkeypatch, tmp_path):
        import pandas as pd
        import watchdog.dataset_manager as dataset_manager
        import main_pipeline

        monkeypatch.setattr(dataset_manager, "load_manifest", lambda: pd.DataFrame([
            {"sample_id": "sample-1", "local_path": "", "source_url": "https://example.ng/a",
             "media_type": "text", "category": "TEXTUAL_CORPUS", "label": 0,
             "publisher_name": "Punch", "scraped_at": "", "notes": ""},
        ]))

        fusion_result = _fusion_result(p_visual=None, p_text=0.2)  # below threshold -> no violation

        def fake_process_manifest_sample(sample_id, session):
            assert sample_id == "sample-1"
            return main_pipeline.PipelineResult(
                media_log=None, fusion_result=fusion_result, decision=None, ip_result=None,
            )

        monkeypatch.setattr(main_pipeline, "process_manifest_sample", fake_process_manifest_sample)

        resp = client.post("/upload", data={"mode": "sample_id", "sample_id": "sample-1"})
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "No violation raised" in body
        assert "WITHIN BUDGET" in body or "OVER BUDGET" in body

        # Latency was logged to the INJECTED results_dir, not the real project results/.
        log_path = app.config["RESULTS_DIR"] / "dashboard_pipeline_latency.csv"
        assert log_path.exists()
        assert "manifest sample sample-1" in log_path.read_text()

    def test_sample_id_success_violation_links_to_detail(self, client, session, monkeypatch):
        import pandas as pd
        import watchdog.dataset_manager as dataset_manager
        import main_pipeline

        monkeypatch.setattr(dataset_manager, "load_manifest", lambda: pd.DataFrame([
            {"sample_id": "sample-2", "local_path": "", "source_url": "https://example.ng/b",
             "media_type": "text", "category": "TEXTUAL_CORPUS", "label": 1,
             "publisher_name": "Vanguard", "scraped_at": "", "notes": ""},
        ]))

        violation = _make_violation(session, p_visual=None, p_text=0.95, violation_type="IP_SCRAPING_PLAGIARISM",
                                     ip_result=_ip_result(), media_type="text", with_report=False)
        fusion_result = _fusion_result(p_visual=None, p_text=0.95)

        def fake_process_manifest_sample(sample_id, session_arg):
            return main_pipeline.PipelineResult(
                media_log=None, fusion_result=fusion_result,
                decision=_decision("IP_SCRAPING_PLAGIARISM", psynth=0.95),
                ip_result=_ip_result(), violation=violation,
            )

        monkeypatch.setattr(main_pipeline, "process_manifest_sample", fake_process_manifest_sample)

        resp = client.post("/upload", data={"mode": "sample_id", "sample_id": "sample-2"})
        assert resp.status_code == 200
        assert f"/violations/{violation.id}".encode() in resp.data

    def test_url_scrape_failure_flashes_error(self, client, monkeypatch):
        import watchdog.scraper as scraper

        monkeypatch.setattr(scraper, "scrape_url", lambda url: None)

        resp = client.post("/upload", data={"mode": "url", "url": "https://unreachable.example/x"})
        assert resp.status_code == 200
        assert b"Could not scrape" in resp.data

    def test_no_mode_selected_flashes_error(self, client):
        resp = client.post("/upload", data={})
        assert resp.status_code == 200
        assert b"Choose a submission mode" in resp.data
