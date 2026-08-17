# =============================================================================
# purge_setup_fixtures.py  —  Admin cleanup: orphaned setup_db.py test fixtures
# =============================================================================
# WHY THIS EXISTS (bug #12, docs/PROJECT_CONTEXT.md §3):
#   setup_db.py's own Step 5 inserts one throwaway violation
#   (publisher_name="TEST SETUP", source_url=".../watch?v=TEST_DAY1_SETUP")
#   to exercise NFR-04's immutability guard, then tries to delete it via the
#   documented allow_violation_mutation() admin bypass. That cleanup was
#   wrapped in a bare `except Exception: logger.warning(...)  # non-critical`
#   — so across two real setup_db.py runs (2026-07-03, 2026-07-14), cleanup
#   silently failed both times and left two fixture rows sitting in the real
#   `violations` table indistinguishable from genuine detections unless you
#   already knew to look for publisher_name="TEST SETUP". Caught only when
#   the Days 21-22 dashboard rendered 3 violations where 1 was expected —
#   not by an exception, not by a test failure. setup_db.py's own cleanup
#   step no longer swallows this silently (see its Step 5) — this script is
#   the one-off/reusable remediation for rows that already leaked out under
#   the old, silent behaviour.
#
# IDEMPOTENT (CLAUDE.md rule 2): queries live, matching rows by
# publisher_name — safe to re-run. Zero matching rows is a normal, correct
# outcome (nothing to do), not an error.
#
# Uses the documented, logged allow_violation_mutation() bypass
# (db_connector.py) — never touches `violations` directly. Every use is
# already logged as a WARNING by that context manager; this script also
# verifies the delete actually happened afterward, rather than trusting a
# clean exit — the same "verify, don't assume" principle setup_db.py's own
# fix (Step 5) now applies to itself.
#
# USAGE:
#   python purge_setup_fixtures.py            # deletes and verifies
#   python purge_setup_fixtures.py --dry-run  # lists matches, deletes nothing
# =============================================================================

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FIXTURE_PUBLISHER_NAME = "TEST SETUP"  # matches setup_db.py's literal marker, exactly


def find_fixture_violations(session):
    from watchdog.db_connector import Violation
    return session.query(Violation).filter(Violation.publisher_name == FIXTURE_PUBLISHER_NAME).all()


def purge(session, dry_run: bool = False) -> int:
    from watchdog.db_connector import Violation, get_violation_by_id, allow_violation_mutation

    fixtures = find_fixture_violations(session)
    if not fixtures:
        logger.info(f"No rows with publisher_name={FIXTURE_PUBLISHER_NAME!r} found — nothing to purge.")
        return 0

    logger.info(f"Found {len(fixtures)} fixture row(s):")
    for v in fixtures:
        logger.info(f"  id={v.id}  type={v.violation_type}  source_url={v.source_url}  detected_at={v.detected_at}")

    if dry_run:
        logger.info("--dry-run: no rows deleted.")
        return len(fixtures)

    purged_ids = []
    for v in fixtures:
        violation_id = v.id
        media_log_entry = v.media_log_entry
        report = v.report

        if report is not None:
            session.delete(report)
            session.flush()  # same FK-ordering fix as setup_db.py's own cleanup (bug #1) —
                              # the Report row must be physically gone before the DELETE
                              # against its parent Violation, or Postgres raises
                              # ForeignKeyViolation on reports_violation_id_fkey.

        with allow_violation_mutation(reason=f"purge_setup_fixtures.py: orphaned setup_db.py "
                                              f"test fixture (bug #12), violation_id={violation_id}"):
            session.query(Violation).filter(Violation.id == violation_id).delete(synchronize_session=False)

        if media_log_entry is not None:
            session.delete(media_log_entry)

        session.commit()
        purged_ids.append(violation_id)

    # Verify — don't just trust a clean commit. This is the exact principle
    # setup_db.py's old Step 5 skipped, which is how these rows survived in
    # the first place.
    still_present = [vid for vid in purged_ids if get_violation_by_id(session, str(vid)) is not None]
    if still_present:
        logger.error(f"VERIFICATION FAILED — still present after delete: {still_present}")
        raise RuntimeError(f"purge_setup_fixtures.py: {len(still_present)} row(s) survived the delete attempt "
                            f"— do not trust this cleanup, investigate before re-running.")

    logger.info(f"✓ Verified: {len(purged_ids)} fixture row(s) deleted and confirmed gone: {purged_ids}")
    return len(purged_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List matching rows without deleting them")
    args = parser.parse_args()

    from watchdog.db_connector import get_engine, get_session_factory, init_db

    engine = init_db(get_engine())
    Session = get_session_factory(engine)
    session = Session()
    try:
        n = purge(session, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"\nDone — {n} fixture row(s) purged.")
    except Exception:
        logger.exception("purge_setup_fixtures.py failed")
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
