# =============================================================================
# tests/conftest.py  —  Shared Pytest Fixtures
# =============================================================================
# WHY SQLITE FOR TESTS, POSTGRESQL FOR PRODUCTION?
#   setup_db.py (Day 1) verifies the schema against a REAL PostgreSQL instance
#   — that is the correct check for production readiness (NFR-04 requires
#   PostgreSQL specifically). But requiring a live PostgreSQL server for every
#   test run is slow and adds friction during development.
#
#   These fixtures instead spin up a fresh **in-memory SQLite** database for
#   every single test function. §2.4.3 of the SQLAlchemy documentation
#   confirms the ORM layer (Column types, relationships, event hooks) behaves
#   identically across both backends for everything this project uses —
#   verified directly in this project on Day 2 (see the probe scripts referenced
#   in the Day 2 write-up). This gives fast, isolated, repeatable tests without
#   ever touching the real compliance database.
#
# Each test gets a BRAND NEW empty database (function scope) — no cleanup
# code is needed, and no test can leak state into another.
# =============================================================================

import sys
from pathlib import Path

# Ensure the project root is importable regardless of where pytest is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from watchdog.db_connector import (
    get_engine, get_session_factory, init_db,
    log_media_ingestion, record_violation, MediaLog, Violation
)


@pytest.fixture()
def engine():
    """Fresh in-memory SQLite engine, one per test function."""
    eng = get_engine("sqlite:///:memory:")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    """SQLAlchemy session bound to the fresh test engine."""
    Session = get_session_factory(engine)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture()
def sample_media_log(session) -> MediaLog:
    """A single MediaLog row — a 'text' artifact ingested from a test URL."""
    return log_media_ingestion(
        session,
        source_url="https://example-news.ng/articles/sample-ad-copy",
        media_type="text"
    )


@pytest.fixture()
def sample_video_media_log(session) -> MediaLog:
    """A single MediaLog row — a 'video' artifact, for CV-pathway tests."""
    return log_media_ingestion(
        session,
        source_url="https://example.com/campaign/mtn-avatar-ad.mp4",
        media_type="video"
    )


@pytest.fixture()
def sample_violation(session, sample_video_media_log) -> Violation:
    """
    A confirmed violation above the Psynth threshold (0.85), matching the
    DEEPFAKE_IMPERSONATION category and NDPA 2023 s.37 mapping from
    config.STATUTORY_MAP.
    """
    return record_violation(
        session=session,
        media_log_entry=sample_video_media_log,
        psynth_score=0.91,
        p_visual=0.94,
        p_text=0.0,
        violation_type="DEEPFAKE_IMPERSONATION",
        statute_cited="Nigeria Data Protection Act (NDPA) 2023, Section 37",
        statute_summary="Test fixture violation for pytest.",
        regulator="Nigeria Data Protection Commission (NDPC)",
        source_url="https://example.com/campaign/mtn-avatar-ad.mp4",
        publisher_name="Test Publisher Ltd",
    )
