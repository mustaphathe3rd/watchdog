# =============================================================================
# dashboard/app.py  —  Days 21-22: Flask Admin Dashboard (§3.13 Routing)
# =============================================================================
# PURPOSE:
#   §3.13 already requires a human regulatory officer to review every
#   auto-drafted complaint before dispatch (reports.status: PENDING_REVIEW ->
#   DISPATCHED). Until now that review had no interface at all — a
#   PENDING_REVIEW report was only reachable via the raw PDF file in
#   reports/ or a direct `psql` query. This module is that interface: it
#   reads the same three tables watchdog/db_connector.py already defines
#   and renders them for a human, it never runs detection logic itself and
#   never writes to `violations` (append-only, NFR-04) except via the
#   ordinary /upload -> main_pipeline.process_media() path every other
#   entry point already uses.
#
# ROUTES:
#   /                       — violations list, most recent first
#   /violations/<id>        — full evidentiary detail for one violation
#   /reports/<id>            — serve the generated PDF (keyed by violation_id,
#                              same as the PDF's own filename convention —
#                              reports.violation_id is 1:1 with violations.id)
#   /upload                  — submit a manifest sample_id or a live URL,
#                              run main_pipeline, show the result + NFR-02 timing
#
# WHAT THIS UI MUST NOT DO (the two things the spec calls out explicitly):
#   1. Never render an abstained modality/heuristic as a blank cell or a
#      0.00 score — watchdog/fusion.py's whole module docstring exists to
#      stop exactly that confusion happening in the arithmetic; the
#      dashboard's job is to not reintroduce it in the rendering. Every
#      abstention shown here carries its stored reason string, visually
#      distinct (amber "ABSTAINED" pill, not an empty table cell).
#   2. Never bury the known false-positive modes at review time. The two
#      caveat strings this project has already written and validated
#      (watchdog/complaint_generator.CAVEAT_IP_FALSE_POSITIVE_MODES,
#      CAVEAT_CV_HEURISTIC_RELIABILITY) are imported and rendered VERBATIM
#      here — not paraphrased into a second, driftable copy — so the PDF a
#      reviewer downloads and the web page they read it on never disagree.
#
# DEV vs. PRODUCTION DATABASE:
#   create_app(engine=None) takes an optional SQLAlchemy engine purely so
#   tests/test_dashboard.py can inject the same in-memory SQLite pattern
#   tests/conftest.py already uses for the rest of the suite (CLAUDE.md:
#   "PostgreSQL, not SQLite, in production ... SQLite is tests only").
#   Every real invocation (`python -m dashboard.app`) calls create_app()
#   with no argument, which falls back to watchdog.db_connector.get_engine()
#   and therefore config.DATABASE_URL — Postgres, per NFR-04.
# =============================================================================

import csv
import json
import logging
import time
from pathlib import Path

from flask import Flask, abort, flash, g, render_template, request, send_file

logger = logging.getLogger(__name__)


# =============================================================================
# NFR-02 latency log — see the /upload route below
# =============================================================================
# Appends one row per pipeline run submitted through this dashboard, so
# Days 23-25 has a real, accumulating sample of end-to-end latency
# (main_pipeline.process_media()/process_manifest_sample() — detection +
# fusion + rules + PDF generation, the full NFR-02 scope) instead of only
# the Stage 2 face-pipeline-only figure README.md currently quotes.
# Append-only, header written once — same "idempotent, never corrupt
# existing work" posture as every other results/*.csv writer in this
# project (CLAUDE.md rule 2), just applied to a log instead of a dataset.
# =============================================================================
_LATENCY_LOG_COLUMNS = ["timestamp", "label", "media_type", "elapsed_s", "budget_s", "within_budget"]


def _log_pipeline_latency(*, results_dir: Path, label: str, media_type: str,
                           elapsed_s: float, budget_s: float, within_budget: bool) -> None:
    from datetime import datetime, timezone

    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "dashboard_pipeline_latency.csv"
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LATENCY_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "media_type": media_type,
            "elapsed_s": f"{elapsed_s:.3f}",
            "budget_s": budget_s,
            "within_budget": within_budget,
        })
    logger.info(f"NFR-02: {media_type} pipeline run took {elapsed_s:.2f}s "
                f"(budget {budget_s}s, {'within' if within_budget else 'OVER'} budget) — logged to {log_path}")


# =============================================================================
# App factory
# =============================================================================

def create_app(engine=None, results_dir=None) -> Flask:
    """
    Args:
        engine:      Optional SQLAlchemy engine — see the module docstring's
                     "DEV vs. PRODUCTION DATABASE" note.
        results_dir: Where /upload's NFR-02 latency log gets written
                     (default: config.BASE_DIR / "results", the same
                     directory every evaluate_*.py script in this project
                     writes to). Overridable so tests/test_dashboard.py can
                     point it at a tmp_path instead — running the test suite
                     must never append rows to the real
                     results/dashboard_pipeline_latency.csv, the same
                     "don't corrupt real project state" principle
                     CLAUDE.md's manifest rules apply elsewhere.
    """
    import config
    from watchdog.db_connector import get_engine, get_session_factory, init_db

    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY
    app.config["RESULTS_DIR"] = Path(results_dir) if results_dir is not None else (config.BASE_DIR / "results")

    if engine is None:
        engine = get_engine()
    init_db(engine)  # idempotent — creates tables / applies the evidence_json migration if not already there
    session_factory = get_session_factory(engine)

    def get_session():
        if "db_session" not in g:
            g.db_session = session_factory()
        return g.db_session

    @app.teardown_appcontext
    def _close_session(exception=None):
        session = g.pop("db_session", None)
        if session is not None:
            session.close()

    # ── / — violations list ─────────────────────────────────────────────────
    @app.route("/")
    def index():
        from watchdog.db_connector import get_all_violations
        violations = get_all_violations(get_session())
        return render_template("index.html", violations=violations)

    # ── /violations/<id> — full evidentiary detail ──────────────────────────
    @app.route("/violations/<violation_id>")
    def violation_detail(violation_id):
        from watchdog.db_connector import get_violation_by_id
        violation = get_violation_by_id(get_session(), violation_id)
        if violation is None:
            abort(404)

        evidence = None
        if violation.evidence_json:
            try:
                evidence = json.loads(violation.evidence_json)
            except (json.JSONDecodeError, TypeError):
                # A malformed evidence_json must not take the whole review
                # page down — the top-level Violation columns (psynth_score,
                # p_visual, p_text, statute_*) are always present regardless
                # and are still rendered; only the rich per-modality detail
                # is unavailable, and the template says so explicitly rather
                # than silently rendering nothing.
                logger.warning(f"violation {violation_id}: evidence_json failed to parse")

        modalities_used = evidence.get("modalities_used", []) if evidence else []

        from watchdog.complaint_generator import (
            CAVEAT_ALWAYS, CAVEAT_CV_HEURISTIC_RELIABILITY, CAVEAT_IP_FALSE_POSITIVE_MODES,
        )

        return render_template(
            "violation_detail.html",
            violation=violation,
            evidence=evidence,
            modalities_used=modalities_used,
            show_ip_caveat=(violation.violation_type == "IP_SCRAPING_PLAGIARISM"),
            show_cv_caveat=("visual" in modalities_used),
            caveat_always=CAVEAT_ALWAYS,
            caveat_ip=CAVEAT_IP_FALSE_POSITIVE_MODES,
            caveat_cv=CAVEAT_CV_HEURISTIC_RELIABILITY,
        )

    # ── /reports/<id> — serve the generated PDF ─────────────────────────────
    # Keyed by violation_id, not reports.id — every other route in this app
    # is keyed by violation_id (the thing a reviewer actually clicks through
    # from), and reports.violation_id is a 1:1 relationship
    # (db_connector.Report, uselist=False) so there is never ambiguity about
    # which report a violation_id resolves to.
    @app.route("/reports/<violation_id>")
    def serve_report(violation_id):
        from watchdog.db_connector import get_violation_by_id
        violation = get_violation_by_id(get_session(), violation_id)
        if violation is None or violation.report is None:
            abort(404)
        pdf_path = Path(violation.report.pdf_path)
        if not pdf_path.exists():
            logger.warning(f"reports.pdf_path={pdf_path} recorded in DB but missing on disk")
            abort(404)
        return send_file(pdf_path, mimetype="application/pdf", download_name=pdf_path.name)

    # ── /upload — submit content, run main_pipeline, show the result ───────
    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html", result=None)

        mode = request.form.get("mode", "")
        session = get_session()

        result = None
        try:
            if mode == "sample_id":
                result = _run_sample_id(session, request.form.get("sample_id", "").strip())
            elif mode == "url":
                result = _run_url(session, request.form.get("url", "").strip())
            else:
                raise ValueError("Choose a submission mode (manifest sample_id or URL).")
        except Exception as e:
            logger.exception(f"/upload failed (mode={mode!r}): {e}")
            flash(str(e), "error")
            return render_template("upload.html", result=None)

        return render_template("upload.html", result=result)

    @app.errorhandler(404)
    def not_found(exception):
        return render_template("404.html"), 404

    return app


# =============================================================================
# /upload helpers — kept out of the route function so each submission mode
# is independently testable (tests/test_dashboard.py monkeypatches
# main_pipeline.process_manifest_sample / process_media / scraper.scrape_url
# rather than exercising the real CV/NLP models through a web request).
# =============================================================================

def _budget_for(media_type: str):
    from config import TEXT_LATENCY_BUDGET_S, VIDEO_LATENCY_BUDGET_S
    return VIDEO_LATENCY_BUDGET_S if media_type == "video" else TEXT_LATENCY_BUDGET_S


def _summarise(*, label: str, media_type: str, elapsed_s: float, pipeline_result) -> dict:
    from flask import current_app

    budget_s = _budget_for(media_type)
    within_budget = elapsed_s <= budget_s
    _log_pipeline_latency(
        results_dir=current_app.config["RESULTS_DIR"], label=label, media_type=media_type,
        elapsed_s=elapsed_s, budget_s=budget_s, within_budget=within_budget,
    )
    return {
        "label": label,
        "media_type": media_type,
        "elapsed_s": elapsed_s,
        "budget_s": budget_s,
        "within_budget": within_budget,
        "decision": pipeline_result.decision,
        "violation": pipeline_result.violation,
        "psynth": pipeline_result.fusion_result.psynth,
        "modalities_used": pipeline_result.fusion_result.modalities_used,
    }


def _run_sample_id(session, sample_id: str) -> dict:
    if not sample_id:
        raise ValueError("Provide a manifest sample_id.")

    # Looked up here (not just left to process_manifest_sample()) purely so
    # the NFR-02 budget/label can be known and the timer can be started
    # around the pipeline call only — a manifest lookup is a local CSV read,
    # not part of what NFR-02 is timing.
    from watchdog.dataset_manager import load_manifest
    df = load_manifest()
    matches = df[df["sample_id"] == sample_id]
    if matches.empty:
        raise ValueError(f"sample_id={sample_id!r} not found in data/manifest.csv")
    media_type = matches.iloc[0]["media_type"]

    from main_pipeline import process_manifest_sample
    t0 = time.perf_counter()
    pipeline_result = process_manifest_sample(sample_id, session)
    elapsed_s = time.perf_counter() - t0

    return _summarise(label=f"manifest sample {sample_id}", media_type=media_type,
                       elapsed_s=elapsed_s, pipeline_result=pipeline_result)


def _run_url(session, url: str) -> dict:
    if not url:
        raise ValueError("Provide a URL.")

    # scrape_url() decides video vs. text from the URL itself (file
    # extension / known video-hosting domains — watchdog/scraper.py) and
    # does the actual acquisition (HTTP GET for text, yt-dlp/direct download
    # for video). That acquisition step is excluded from the NFR-02 timer
    # below deliberately: NFR-02 is a budget on the DETECTION pipeline
    # (analysis + fusion + rules + PDF generation), not on third-party
    # network/download latency, which this project doesn't control and
    # which the existing 30s Stage-2 figure never included either.
    from watchdog import scraper
    artifact = scraper.scrape_url(url)
    if artifact is None:
        raise ValueError(f"Could not scrape {url} — see server logs for details "
                          f"(unreachable, blocked, or an unsupported page/video format).")

    from main_pipeline import process_media
    t0 = time.perf_counter()
    if artifact.is_video():
        if artifact.local_path is None:
            raise ValueError(f"{url} was identified as video but no file was downloaded "
                              f"(see server logs — this project's own bug log has several "
                              f"real cases of this, e.g. Facebook Watch/Reels).")
        pipeline_result = process_media(
            session=session, source_url=artifact.source_url, media_type="video",
            local_path=str(artifact.local_path), publisher_name=artifact.publisher_name,
        )
    else:
        if not artifact.raw_text:
            raise ValueError(f"{url} was identified as a text page but no article text was extracted.")
        pipeline_result = process_media(
            session=session, source_url=artifact.source_url, media_type="text",
            text=artifact.raw_text, publisher_name=artifact.publisher_name,
        )
    elapsed_s = time.perf_counter() - t0

    return _summarise(label=url, media_type=artifact.media_type,
                       elapsed_s=elapsed_s, pipeline_result=pipeline_result)


if __name__ == "__main__":
    import config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    app = create_app()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
