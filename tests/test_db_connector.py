# =============================================================================
# tests/test_db_connector.py  —  Database Layer Test Suite
# =============================================================================
# Covers:
#   - NFR-05 (Data Privacy): MediaLog never stores the raw URL, only a hash
#   - NFR-04 (Auditability): violations table blocks UPDATE and DELETE,
#     both via raw SQL and via the ORM bulk-operation path
#   - The Day 2 admin bypass: legitimate, logged mutation still possible
#   - Core relationships: MediaLog -> Violation -> Report
#   - Query helpers: get_all_violations, get_violation_by_id
#
# RUN:
#   pytest tests/test_db_connector.py -v
# =============================================================================

import hashlib
import json
import uuid

import pytest
from sqlalchemy import text

from watchdog.db_connector import (
    MediaLog, Violation, Report,
    log_media_ingestion, record_violation, record_report,
    get_all_violations, get_violation_by_id,
    allow_violation_mutation,
)
from config import PSYNTH_THRESHOLD, STATUTORY_MAP


# =============================================================================
# NFR-05 — Data Privacy / NDPA Data Minimisation (§3.16)
# =============================================================================

class TestDataMinimisation:
    """
    §3.16 Ethical Considerations:
    'identifying metadata was only permanently stored if an 85% violation
     threshold was definitively crossed.'
    """

    def test_media_log_stores_hash_not_url(self, session):
        source_url = "https://example-news.ng/articles/some-unique-slug-12345"
        entry = log_media_ingestion(session, source_url, media_type="text")

        expected_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
        assert entry.url_hash == expected_hash

        # The raw URL string must NEVER appear on the MediaLog row.
        # We check every string-typed column value on the instance.
        for column in MediaLog.__table__.columns:
            value = getattr(entry, column.name)
            if isinstance(value, str):
                assert source_url not in value, (
                    f"NDPA VIOLATION: raw URL leaked into MediaLog.{column.name}"
                )

    def test_media_log_hash_is_deterministic(self, session):
        """Same URL must always hash the same way (needed for de-duplication)."""
        url = "https://example.com/campaign/ad.mp4"
        entry1 = log_media_ingestion(session, url, media_type="video")
        entry2 = log_media_ingestion(session, url, media_type="video")
        assert entry1.url_hash == entry2.url_hash

    def test_source_url_only_persisted_on_confirmed_violation(
        self, session, sample_video_media_log
    ):
        """
        The full source_url should only ever be written to the `violations`
        table — and only for entries that crossed PSYNTH_THRESHOLD (0.85).
        This test documents that contract explicitly.
        """
        url = "https://example.com/campaign/mtn-avatar-ad.mp4"
        assert sample_video_media_log.url_hash == hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()

        # Below-threshold content should simply never reach record_violation() —
        # that call site decision lives in fusion.py (Day 4+), not here. This
        # test just confirms record_violation() is the only function that
        # accepts/writes a raw source_url.
        violation = record_violation(
            session=session,
            media_log_entry=sample_video_media_log,
            psynth_score=0.91,   # >= PSYNTH_THRESHOLD
            p_visual=0.94, p_text=0.0,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["statute"],
            statute_summary=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["summary"],
            regulator=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["regulator"],
            source_url=url,
            publisher_name="MTN Nigeria (test fixture)",
        )
        assert violation.source_url == url
        assert violation.psynth_score >= PSYNTH_THRESHOLD


# =============================================================================
# NFR-04 — Auditability / Immutable Chain-of-Custody
# =============================================================================

class TestImmutabilityGuard:
    """
    NFR-04: 'All detected anomalies and generated reports must be stored
    immutably in a PostgreSQL database to maintain legal chain-of-custody.'
    """

    def test_raw_sql_update_is_blocked(self, session, engine, sample_violation):
        with pytest.raises(PermissionError, match="NFR-04"):
            with engine.connect() as conn:
                conn.execute(text(
                    f"UPDATE violations SET psynth_score = 0.0 "
                    f"WHERE id = '{sample_violation.id}'"
                ))

    def test_raw_sql_delete_is_blocked(self, session, engine, sample_violation):
        with pytest.raises(PermissionError, match="NFR-04"):
            with engine.connect() as conn:
                conn.execute(text(
                    f"DELETE FROM violations WHERE id = '{sample_violation.id}'"
                ))

    def test_orm_bulk_delete_is_blocked(self, session, sample_violation):
        """
        Regression test for a real Day 1 bug: Query.delete() issues a raw
        DELETE statement under the hood and WAS being silently swallowed by
        a broad except-clause in setup_db.py's cleanup step. It must still
        be blocked by default — this test proves the guard has no gaps for
        the ORM's bulk-operation path.
        """
        with pytest.raises(PermissionError, match="NFR-04"):
            session.query(Violation).filter(
                Violation.id == sample_violation.id
            ).delete(synchronize_session=False)

    def test_violation_survives_after_blocked_mutation_attempt(
        self, session, engine, sample_violation
    ):
        """The row must remain completely untouched after a rejected UPDATE."""
        original_score = sample_violation.psynth_score
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    f"UPDATE violations SET psynth_score = 0.0 "
                    f"WHERE id = '{sample_violation.id}'"
                ))
        except PermissionError:
            pass

        session.refresh(sample_violation)
        assert sample_violation.psynth_score == original_score

    def test_other_tables_are_not_restricted(self, session, sample_media_log):
        """
        Sanity check: the guard is scoped ONLY to `violations`. MediaLog rows
        (pre-violation, no legal significance yet) must remain freely editable.
        """
        sample_media_log.processed = True
        session.commit()  # Should not raise
        session.refresh(sample_media_log)
        assert sample_media_log.processed is True


class TestAdminBypass:
    """
    Day 2 addition: a legitimate, explicit, logged escape hatch for
    administrative correction — without weakening the default-on guard.
    """

    def test_bypass_allows_delete(self, session, sample_violation):
        violation_id = sample_violation.id
        with allow_violation_mutation(reason="pytest: admin bypass coverage"):
            session.query(Violation).filter(
                Violation.id == violation_id
            ).delete(synchronize_session=False)
            session.commit()

        assert get_violation_by_id(session, str(violation_id)) is None

    def test_guard_reactivates_after_bypass_block_exits(
        self, session, sample_violation
    ):
        """The bypass must be scoped to its `with` block only — not sticky."""
        with allow_violation_mutation(reason="pytest: scoping check"):
            pass  # No mutation performed inside the block

        # Guard must be back ON immediately after the block exits
        with pytest.raises(PermissionError, match="NFR-04"):
            session.query(Violation).filter(
                Violation.id == sample_violation.id
            ).delete(synchronize_session=False)

    def test_bypass_logs_a_warning(self, session, sample_violation, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            with allow_violation_mutation(reason="pytest: audit log check"):
                session.query(Violation).filter(
                    Violation.id == sample_violation.id
                ).delete(synchronize_session=False)
                session.commit()

        assert any("ADMIN OVERRIDE" in record.message for record in caplog.records)
        assert any("audit log check" in record.message for record in caplog.records)


class TestSetupCleanupRegression:
    """
    Regression coverage for a real bug found running setup_db.py against a
    live PostgreSQL instance (not caught by the original 58-test suite,
    because SQLite doesn't enforce foreign keys by default — see the
    `PRAGMA foreign_keys=ON` fix in get_engine()).

    Bug: get_session_factory() uses autoflush=False. `session.delete(report)`
    only marks the row for deletion in the session's pending state — it does
    NOT hit the database immediately. If a bulk delete on the PARENT
    Violation row (`.delete(synchronize_session=False)`) runs before that
    pending delete is flushed, the Report row is still physically present in
    the database, and the FK constraint `reports_violation_id_fkey`
    correctly rejects the parent delete.

    Fix: call `session.flush()` between the two operations (see setup_db.py).
    These tests pin down both the failure mode and the correct pattern so it
    can't silently regress.
    """

    def test_deleting_violation_before_flushing_pending_report_delete_fails(
        self, session, sample_violation, tmp_path
    ):
        """
        Reproduces the exact production failure: session.delete(report)
        WITHOUT a subsequent flush(), followed immediately by a bulk delete
        of the parent Violation, must raise — proving the FK constraint is
        genuinely active (this is what SQLite was silently skipping before
        the PRAGMA fix).
        """
        pdf_path = tmp_path / f"{sample_violation.id}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")
        report = record_report(session, sample_violation, pdf_path)

        from sqlalchemy.exc import IntegrityError

        session.delete(report)
        # Deliberately NOT calling session.flush() here — this is the bug.
        with pytest.raises(IntegrityError):
            with allow_violation_mutation(reason="pytest: reproduce ordering bug"):
                session.query(Violation).filter(
                    Violation.id == sample_violation.id
                ).delete(synchronize_session=False)
                session.flush()

    def test_flush_before_parent_delete_succeeds(
        self, session, sample_violation, tmp_path
    ):
        """
        The correct pattern (as fixed in setup_db.py): flush the child
        Report's pending delete BEFORE bulk-deleting its parent Violation.
        This must succeed cleanly with no FK error.
        """
        pdf_path = tmp_path / f"{sample_violation.id}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")
        report = record_report(session, sample_violation, pdf_path)
        violation_id = sample_violation.id

        session.delete(report)
        session.flush()   # <-- the fix
        with allow_violation_mutation(reason="pytest: correct cleanup ordering"):
            session.query(Violation).filter(
                Violation.id == violation_id
            ).delete(synchronize_session=False)
        session.commit()

        assert get_violation_by_id(session, str(violation_id)) is None


# =============================================================================
# Relationships: MediaLog -> Violation -> Report
# =============================================================================

class TestRelationships:

    def test_violation_links_back_to_media_log(self, session, sample_violation):
        assert sample_violation.media_log_entry is not None
        assert sample_violation.media_log_entry.id == sample_violation.media_log_id

    def test_media_log_marked_processed_after_violation(
        self, session, sample_video_media_log, sample_violation
    ):
        session.refresh(sample_video_media_log)
        assert sample_video_media_log.processed is True

    def test_report_links_to_violation(self, session, sample_violation, tmp_path):
        pdf_path = tmp_path / f"{sample_violation.id}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        report = record_report(session, sample_violation, pdf_path)

        assert report.violation_id == sample_violation.id
        assert report.status == "PENDING_REVIEW"
        assert sample_violation.report.id == report.id


# =============================================================================
# Query Helpers
# =============================================================================

class TestQueryHelpers:

    def test_get_violation_by_id_found(self, session, sample_violation):
        found = get_violation_by_id(session, str(sample_violation.id))
        assert found is not None
        assert found.id == sample_violation.id

    def test_get_violation_by_id_not_found(self, session):
        random_id = str(uuid.uuid4())
        assert get_violation_by_id(session, random_id) is None

    def test_get_all_violations_orders_most_recent_first(
        self, session, sample_video_media_log
    ):
        v1 = record_violation(
            session=session, media_log_entry=sample_video_media_log,
            psynth_score=0.86, p_visual=0.86, p_text=0.0,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited="NDPA s.37", statute_summary="first", regulator="NDPC",
            source_url="https://example.com/a.mp4",
        )
        media2 = log_media_ingestion(session, "https://example.com/b.mp4", "video")
        v2 = record_violation(
            session=session, media_log_entry=media2,
            psynth_score=0.95, p_visual=0.95, p_text=0.0,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited="NDPA s.37", statute_summary="second", regulator="NDPC",
            source_url="https://example.com/b.mp4",
        )

        results = get_all_violations(session)
        ids_in_order = [r.id for r in results]
        assert ids_in_order.index(v2.id) < ids_in_order.index(v1.id)


# =============================================================================
# Statutory Map Integration (config.py <-> db_connector.py contract)
# =============================================================================

class TestStatutoryMapIntegration:
    """
    Ensures every category in config.STATUTORY_MAP can actually be persisted
    through record_violation() without truncation or type errors — this is
    what rules_engine.py (Day 5+) will rely on.
    """

    @pytest.mark.parametrize("violation_type", list(STATUTORY_MAP.keys()))
    def test_each_statutory_category_persists_correctly(
        self, session, sample_video_media_log, violation_type
    ):
        mapping = STATUTORY_MAP[violation_type]
        violation = record_violation(
            session=session,
            media_log_entry=sample_video_media_log,
            psynth_score=0.90, p_visual=0.90, p_text=0.90,
            violation_type=violation_type,
            statute_cited=mapping["statute"],
            statute_summary=mapping["summary"],
            regulator=mapping["regulator"],
            source_url="https://example.com/artifact",
            publisher_name="Test Publisher",
        )
        assert violation.violation_type == violation_type
        assert violation.statute_cited == mapping["statute"]
        assert violation.regulator == mapping["regulator"]


# =============================================================================
# evidence_json / pre-assigned violation_id (Day 16, §3.13 audit trail)
# =============================================================================

class TestEvidenceJsonColumn:
    """
    watchdog/complaint_generator.py and main_pipeline.py (Day 16) need the
    violation's full evidentiary detail to survive after the process that
    generated it exits, and need the row's own id BEFORE insert (the
    append-only guard means there's no legitimate UPDATE afterwards) — see
    db_connector.record_violation()'s docstring.
    """

    def test_evidence_json_round_trips(self, session, sample_video_media_log):
        payload = json.dumps({"p_visual": 0.9, "p_text": 0.4, "modalities_used": ["visual", "text"]})
        violation = record_violation(
            session=session, media_log_entry=sample_video_media_log,
            psynth_score=0.9, p_visual=0.9, p_text=0.4,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["statute"],
            statute_summary=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["summary"],
            regulator=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["regulator"],
            source_url="https://example.com/x", evidence_json=payload,
        )
        assert violation.evidence_json == payload
        assert json.loads(violation.evidence_json)["p_visual"] == 0.9

    def test_evidence_json_defaults_to_none(self, session, sample_video_media_log):
        violation = record_violation(
            session=session, media_log_entry=sample_video_media_log,
            psynth_score=0.9, p_visual=0.9, p_text=0.4,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["statute"],
            statute_summary=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["summary"],
            regulator=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["regulator"],
            source_url="https://example.com/x",
        )
        assert violation.evidence_json is None

    def test_pre_assigned_violation_id_is_used_verbatim(self, session, sample_video_media_log):
        preassigned = uuid.uuid4()
        violation = record_violation(
            session=session, media_log_entry=sample_video_media_log,
            psynth_score=0.9, p_visual=0.9, p_text=0.4,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["statute"],
            statute_summary=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["summary"],
            regulator=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["regulator"],
            source_url="https://example.com/x", violation_id=preassigned,
        )
        assert violation.id == preassigned

    def test_omitted_violation_id_still_generates_one(self, session, sample_video_media_log):
        violation = record_violation(
            session=session, media_log_entry=sample_video_media_log,
            psynth_score=0.9, p_visual=0.9, p_text=0.4,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["statute"],
            statute_summary=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["summary"],
            regulator=STATUTORY_MAP["DEEPFAKE_IMPERSONATION"]["regulator"],
            source_url="https://example.com/x",
        )
        assert violation.id is not None

    def test_ensure_evidence_json_column_is_idempotent_on_a_fresh_db(self, engine):
        """
        A fresh SQLite test DB (conftest's `engine` fixture) already has the
        column via create_all() — calling the migration helper again must
        be a clean no-op, not an error (e.g. 'duplicate column').
        """
        from watchdog.db_connector import _ensure_evidence_json_column

        _ensure_evidence_json_column(engine)  # first call already happened inside init_db(); must not raise again
        _ensure_evidence_json_column(engine)
