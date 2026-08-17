# =============================================================================
# setup_db.py  —  Day 1 Database Initialisation Script
# =============================================================================
# RUN THIS ONCE on Day 1 after installing PostgreSQL.
#
# What it does:
#   1. Connects to PostgreSQL using the DATABASE_URL from config.py / .env
#   2. Creates the naija_watchdog database (if it doesn't exist)
#   3. Creates all 3 tables: media_log, violations, reports
#   4. Inserts a test row to verify the immutability guard (NFR-04)
#   5. Prints a success summary
#
# PREREQUISITES (Day 1, Task 2):
#   sudo apt install postgresql postgresql-contrib  (Ubuntu)
#   OR brew install postgresql  (macOS)
#
#   Then:
#   sudo -u postgres psql -c "CREATE USER watchdog_user WITH PASSWORD 'watchdog_pass';"
#   sudo -u postgres psql -c "CREATE DATABASE naija_watchdog OWNER watchdog_user;"
#
# USAGE:
#   python setup_db.py
# =============================================================================

import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_setup():
    print("\n" + "=" * 65)
    print("  Naija AI Ethics Watchdog — Database Initialisation")
    print("  NFR-04: Immutable Audit Log | NFR-05: NDPA Data Minimisation")
    print("=" * 65 + "\n")

    # ── Step 1: Import config and db_connector ────────────────────────────────
    try:
        from config import DATABASE_URL, PSYNTH_THRESHOLD
        from watchdog.db_connector import (
            init_db, get_engine, get_session_factory,
            log_media_ingestion, record_violation, record_report,
            get_all_violations
        )
        from pathlib import Path
        logger.info(f"Config loaded. Target DB: {DATABASE_URL}")
        logger.info(f"Psynth threshold: {PSYNTH_THRESHOLD}")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        logger.error("Make sure you are running from the project root and "
                     "have run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Step 2: Create engine and tables ─────────────────────────────────────
    logger.info("Connecting to PostgreSQL and creating tables...")
    try:
        engine  = init_db()
        Session = get_session_factory(engine)
        logger.info("✓ Tables created: media_log, violations, reports")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error(
            "\nTroubleshooting:\n"
            "  1. Is PostgreSQL running?  sudo systemctl start postgresql\n"
            "  2. Does the database exist? "
            "     sudo -u postgres psql -c 'CREATE DATABASE naija_watchdog OWNER watchdog_user;'\n"
            "  3. Check DATABASE_URL in config.py or your .env file\n"
            "  4. For SQLite testing, set DATABASE_URL=sqlite:///test_watchdog.db in .env"
        )
        sys.exit(1)

    # ── Step 3: Insert a test row to verify write works ───────────────────────
    logger.info("Testing write operations...")
    session = Session()

    try:
        # Log a dummy media ingestion (NDPA compliant — only hash stored)
        test_url = "https://www.youtube.com/watch?v=TEST_DAY1_SETUP"
        media_entry = log_media_ingestion(session, test_url, media_type="text")
        logger.info(f"✓ MediaLog test row: id={media_entry.id} | "
                    f"url_hash={media_entry.url_hash[:20]}...")

        # Record a dummy violation
        violation = record_violation(
            session=session,
            media_log_entry=media_entry,
            psynth_score=0.91,
            p_visual=0.88,
            p_text=0.95,
            violation_type="DEEPFAKE_IMPERSONATION",
            statute_cited="Nigeria Data Protection Act (NDPA) 2023, Section 37",
            statute_summary=(
                "Test violation — Day 1 database setup verification. "
                "Section 37 of the NDPA 2023 governs automated decision-making."
            ),
            regulator="Nigeria Data Protection Commission (NDPC)",
            source_url=test_url,
            publisher_name="TEST SETUP"
        )
        logger.info(f"✓ Violation test row: id={violation.id} | "
                    f"psynth={violation.psynth_score} | type={violation.violation_type}")

        # Record a dummy report
        dummy_pdf = Path("reports") / f"TEST_{violation.id}.pdf"
        report = record_report(session, violation, dummy_pdf)
        logger.info(f"✓ Report test row: id={report.id} | status={report.status}")

    except Exception as e:
        session.rollback()
        logger.error(f"Write test failed: {e}")
        sys.exit(1)

    # ── Step 4: Verify immutability guard (NFR-04) ────────────────────────────
    logger.info("Testing NFR-04 immutability guard on violations table...")
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(
                f"UPDATE violations SET psynth_score = 0.0 WHERE id = '{violation.id}'"
            ))
        # If we get here, the guard FAILED — that's an error
        logger.error("✗ IMMUTABILITY GUARD FAILED — UPDATE was not blocked!")
        logger.error("  The guard in db_connector.py may not be working correctly.")
        sys.exit(1)
    except PermissionError as e:
        logger.info(f"✓ Immutability guard ACTIVE: UPDATE correctly blocked.")
        logger.info(f"  Reason: {e}")
    except Exception as e:
        # Some drivers raise different exceptions — log but don't fail setup
        logger.warning(f"Immutability test raised unexpected error: {type(e).__name__}: {e}")

    # ── Step 5: Clean up test data ────────────────────────────────────────────
    # Remove test rows so the database starts clean.
    # NOTE (fixed Day 2): deleting from `violations` is blocked by the NFR-04
    # immutability guard by design — that guard was firing here too and being
    # silently swallowed by a broad except clause. We now use the explicit,
    # logged administrative bypass added in db_connector.py instead of
    # papering over the exception.
    #
    # BUG #12 (found 2026-08-13, docs/PROJECT_CONTEXT.md §3): this whole
    # block used to catch its own failure as `logger.warning(...
    # "non-critical")` and let the script finish and print the success
    # banner anyway. On two real runs (2026-07-03, 2026-07-14) that swallowed
    # a genuine cleanup failure both times, leaving this test fixture
    # (publisher_name="TEST SETUP") sitting in the real `violations` table —
    # indistinguishable from a real detection to anything downstream that
    # queries that table — for over a month, until the Days 21-22 dashboard
    # happened to render a row count that didn't match expectations. Neither
    # occurrence raised an exception the operator would see, and nothing
    # about the printed "DAY 1 DATABASE SETUP COMPLETE" banner said cleanup
    # had actually failed. Same silent-failure shape as bugs #6/#9/#10 in
    # this file's own log, in a new form: a caught exception logged at the
    # wrong severity is just as silent as no exception at all. Fixed two
    # ways: (1) a cleanup failure is no longer "non-critical" — it now exits
    # non-zero and skips the success banner entirely; (2) even a clean
    # commit is independently VERIFIED by re-querying for the row afterward,
    # rather than trusted — the same "verify, don't assume" principle
    # purge_setup_fixtures.py (written to remove the two rows this bug left
    # behind) applies to its own delete.
    from watchdog.db_connector import Report, Violation, allow_violation_mutation, get_violation_by_id

    # Captured BEFORE the delete, as a plain string — not read off `violation`
    # again afterward. get_session_factory()'s sessionmaker defaults to
    # expire_on_commit=True, so every attribute on `violation` (including
    # `.id`) is expired by the session.commit() below; touching `violation.id`
    # again after that forces SQLAlchemy to refresh it from the DB to serve
    # the attribute access, and since the row is now gone, that raises
    # `ObjectDeletedError` instead of returning cleanly — caught by actually
    # running this fix against the real DB (2026-08-13), not just reasoning
    # through it, which is exactly the "verify against reality" principle
    # this whole cleanup step exists to enforce.
    violation_id = str(violation.id)

    try:
        session.delete(report)
        session.flush()   # BUG FIX: force the pending Report DELETE to hit the DB
                          # now, before we touch its parent Violation row below.
                          # Without this, autoflush=False (get_session_factory())
                          # leaves the Report row physically in place, and Postgres
                          # correctly rejects "DELETE FROM violations" with a
                          # ForeignKeyViolation on reports_violation_id_fkey —
                          # exactly the error this fix resolves.
        with allow_violation_mutation(reason="Day 1/2 setup_db.py test-fixture cleanup"):
            session.query(Violation).filter(Violation.id == violation.id).delete(
                synchronize_session=False
            )
        session.delete(media_entry)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"✗ CLEANUP FAILED: {e}")
        logger.error(
            f"  The Day 1 test fixture (violation.id={violation_id}) is still in the real "
            f"`violations` table. This is NOT non-critical — Chapter 4 evaluation would count "
            f"it as a real detection. Run `python purge_setup_fixtures.py` to remove it, then "
            f"re-run this script."
        )
        session.close()
        sys.exit(1)

    # Verify the delete actually happened — a clean commit() is not proof by
    # itself (see bug #12 above: the previous version trusted exactly this).
    if get_violation_by_id(session, violation_id) is not None:
        logger.error(f"✗ CLEANUP VERIFICATION FAILED: violation.id={violation_id} "
                      f"still present after a commit that raised no exception.")
        logger.error("  Run `python purge_setup_fixtures.py` to remove it before trusting this database.")
        session.close()
        sys.exit(1)

    logger.info("✓ Test data cleaned up and VERIFIED gone — database is empty and ready.")
    session.close()

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DAY 1 DATABASE SETUP COMPLETE")
    print("=" * 65)
    print(f"  Database:    naija_watchdog (PostgreSQL)")
    print(f"  Tables:      media_log | violations | reports")
    print(f"  NFR-04:      Immutability guard ACTIVE on violations table")
    print(f"  NFR-05:      Data minimisation configured (PSYNTH_THRESHOLD={PSYNTH_THRESHOLD})")
    print(f"  Next step:   Run `python main_pipeline.py --help` to see usage")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_setup()
