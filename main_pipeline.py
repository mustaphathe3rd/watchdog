# =============================================================================
# main_pipeline.py  —  Day 16-20: End-to-End Orchestrator (§3.13)
# =============================================================================
# PURPOSE:
#   Wires the whole detection -> compliance chain together for one piece of
#   content at a time:
#
#     [scraper/manifest] -> preprocessor -> {cv_engine, ip_similarity}
#       -> fusion.run_fusion() -> rules_engine.evaluate()
#       -> db_connector (media_log, violations, reports)
#       -> complaint_generator.generate_complaint_pdf()
#
#   Every module in that chain already exists and is independently tested
#   (watchdog/cv_engine.py, watchdog/ip_similarity.py, watchdog/fusion.py,
#   watchdog/rules_engine.py, watchdog/complaint_generator.py) — this file's
#   only job is calling them in the right order with the right data, and
#   persisting the result. See README.md's Architecture diagram.
#
# WHY VIOLATION_ID IS GENERATED HERE, BEFORE THE DB INSERT:
#   generate_complaint_pdf() needs a violation_id to name the PDF file and
#   to embed in the evidence snapshot, but `violations` is append-only
#   (NFR-04) — there's no legitimate way to UPDATE a row afterwards to
#   attach evidence_json once it exists. So the UUID is generated up front
#   (uuid.uuid4()) and passed into record_violation()'s violation_id
#   parameter, making it a single INSERT with everything already in place —
#   see db_connector.record_violation()'s docstring for the same reasoning.
#
# USAGE:
#   python main_pipeline.py --sample-id <manifest sample_id>
#   python main_pipeline.py --text-file path/to.txt --source-url <url> [--publisher NAME]
# =============================================================================

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    media_log: object
    fusion_result: object
    decision: Optional[object]           # watchdog.rules_engine.RuleDecision, or None if no violation
    ip_result: Optional[object]          # watchdog.ip_similarity.IPSimilarityResult, or None for video
    violation: Optional[object] = None   # watchdog.db_connector.Violation, or None
    report: Optional[object] = None      # watchdog.db_connector.Report, or None
    pdf_path: Optional[Path] = None


def process_media(*, session, source_url: str, media_type: str, local_path: Optional[str] = None,
                   text: Optional[str] = None, category: Optional[str] = None,
                   publisher_name: Optional[str] = None) -> PipelineResult:
    """
    Runs the full pipeline against one piece of content and persists the
    result. Either `local_path` (video) or `text` (already-loaded text) must
    be supplied for the corresponding media_type — text may also be read
    from `local_path` if `text` isn't given directly.

    Args:
        session:        Active SQLAlchemy session (watchdog.db_connector).
        source_url:      Full URL of the content (written to `violations`
                         only if flagged — NFR-05).
        media_type:      'video' | 'text'.
        local_path:      Path to the video file (media_type='video'), or to
                         a text file to read (media_type='text', if `text`
                         isn't supplied directly).
        text:            Raw text (media_type='text') — takes precedence
                         over reading local_path if both are given.
        category:        Optional watchdog.dataset_manager category
                         constant — informs rules_engine's violation-type
                         choice for video (POLITICAL_MEDIA vs. other).
        publisher_name:  Optional publisher/brand name for the complaint.

    Returns:
        PipelineResult. `.decision` is None (and `.violation`/`.report`/
        `.pdf_path` stay None) when Psynth never crossed
        config.PSYNTH_THRESHOLD, or crossed it but no statutorily-mapped
        violation_type applies (see watchdog.rules_engine's module
        docstring) — that is a normal, expected outcome for most content,
        not a failure.
    """
    from watchdog.db_connector import log_media_ingestion, record_violation, record_report
    from watchdog import fusion, rules_engine
    from watchdog.complaint_generator import build_complaint_context, generate_complaint_pdf

    if media_type not in ("video", "text"):
        raise ValueError(f"media_type must be 'video' or 'text', got {media_type!r}")

    media_log = log_media_ingestion(session, source_url=source_url, media_type=media_type)

    clip_heuristics = None
    ip_result = None

    if media_type == "video":
        if not local_path:
            raise ValueError("media_type='video' requires local_path")
        from watchdog.cv_engine import run_heuristics
        logger.info(f"Running CV Stage 2+4 pipeline on {local_path} ...")
        clip_heuristics = run_heuristics(local_path)
        if clip_heuristics is None:
            logger.warning(f"cv_engine.run_heuristics() could not open/track {local_path} — visual modality will abstain.")

    else:  # text
        if text is None and local_path:
            text = Path(local_path).read_text(encoding="utf-8", errors="replace")
        if not text:
            raise ValueError("media_type='text' requires either text= or a readable local_path")

        from watchdog.ip_similarity import check_similarity
        logger.info("Running IP-similarity check ...")
        ip_result = check_similarity(text)

    logger.info("Running §3.11 fusion ...")
    fusion_result = fusion.run_fusion(clip_heuristics=clip_heuristics, text=text)
    logger.info(
        f"Psynth={fusion_result.psynth if fusion_result.psynth is None else round(fusion_result.psynth, 4)}, "
        f"modalities_used={fusion_result.modalities_used}"
    )

    decision = rules_engine.evaluate(
        media_type=media_type, psynth=fusion_result.psynth, category=category, ip_result=ip_result,
    )

    # Whether or not a violation resulted, the ingestion attempt is complete —
    # mark it processed. record_violation() below would also do this as a
    # side-effect when a violation IS raised; done explicitly here so the
    # no-violation path (the common case) doesn't leave media_log.processed
    # permanently False.
    media_log.processed = True
    session.commit()

    if decision is None:
        return PipelineResult(media_log=media_log, fusion_result=fusion_result, decision=None, ip_result=ip_result)

    violation_id = uuid.uuid4()
    context = build_complaint_context(
        violation_id=str(violation_id), decision=decision, fusion_result=fusion_result, ip_result=ip_result,
        source_url=source_url, publisher_name=publisher_name, media_type=media_type,
    )
    evidence_json = json.dumps(context.to_evidence_dict())

    violation = record_violation(
        session=session,
        media_log_entry=media_log,
        psynth_score=fusion_result.psynth,
        p_visual=fusion_result.p_visual,
        p_text=fusion_result.p_text,
        violation_type=decision.violation_type,
        statute_cited=decision.statute,
        statute_summary=decision.statute_summary,
        regulator=decision.regulator,
        source_url=source_url,
        publisher_name=publisher_name,
        evidence_json=evidence_json,
        violation_id=violation_id,
    )

    pdf_path = generate_complaint_pdf(context)
    report = record_report(session, violation, pdf_path)

    logger.info(
        f"VIOLATION RECORDED: id={violation.id}, type={violation.violation_type}, "
        f"psynth={violation.psynth_score:.4f} -> {pdf_path}"
    )

    return PipelineResult(
        media_log=media_log, fusion_result=fusion_result, decision=decision, ip_result=ip_result,
        violation=violation, report=report, pdf_path=pdf_path,
    )


def process_manifest_sample(sample_id: str, session) -> PipelineResult:
    """Looks up `sample_id` in data/manifest.csv and runs process_media() against it."""
    from watchdog.dataset_manager import load_manifest

    df = load_manifest()
    matches = df[df["sample_id"] == sample_id]
    if matches.empty:
        raise ValueError(f"sample_id={sample_id!r} not found in manifest")
    row = matches.iloc[0]

    return process_media(
        session=session,
        source_url=row["source_url"],
        media_type=row["media_type"],
        local_path=row["local_path"],
        category=row["category"] or None,
        publisher_name=row["publisher_name"] or None,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", help="Process one sample from data/manifest.csv by sample_id")
    parser.add_argument("--text-file", help="Process an ad-hoc text file (media_type=text)")
    parser.add_argument("--source-url", help="Source URL to record (required with --text-file)")
    parser.add_argument("--publisher", default=None, help="Publisher/brand name (optional)")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    args = _build_arg_parser().parse_args()

    from watchdog.db_connector import get_engine, get_session_factory, init_db

    engine = init_db(get_engine())
    Session = get_session_factory(engine)
    session = Session()

    try:
        if args.sample_id:
            result = process_manifest_sample(args.sample_id, session)
        elif args.text_file:
            if not args.source_url:
                raise SystemExit("--text-file requires --source-url")
            text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
            result = process_media(
                session=session, source_url=args.source_url, media_type="text",
                text=text, publisher_name=args.publisher,
            )
        else:
            raise SystemExit("Pass --sample-id or --text-file (see --help)")

        if result.decision is None:
            print(f"No violation raised. Psynth={result.fusion_result.psynth}, "
                  f"modalities_used={result.fusion_result.modalities_used}")
        else:
            print(f"VIOLATION: {result.violation.violation_type}  Psynth={result.violation.psynth_score:.4f}")
            print(f"Complaint PDF: {result.pdf_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
