# =============================================================================
# watchdog/db_connector.py  —  Database Layer
# =============================================================================
# PURPOSE:
#   Defines the PostgreSQL schema and all database operations for the Naija AI
#   Watchdog. Implements the immutable audit log required by NFR-04 and the
#   data-minimisation principle of the NDPA 2023 (NFR-05, §3.16).
#
# SCHEMA (3 tables):
#   1. media_log      — every artifact ingested (URL, type, timestamp)
#   2. violations     — flagged detections where Psynth >= 0.85 (APPEND-ONLY)
#   3. reports        — generated PDF complaint records
#
# NFR-04 (Auditability):
#   "All detected anomalies and generated reports must be stored immutably in
#    a PostgreSQL database to maintain legal chain-of-custody."
#   Implementation: NO UPDATE or DELETE is ever called on `violations`.
#   Rows are inserted once and never modified.
#
# NFR-05 (Data Privacy / NDPA data minimisation, §3.16):
#   "identifying metadata was only permanently stored if an 85% violation
#    threshold was definitively crossed."
#   Implementation: media_log stores only non-identifying hashes and media
#   type. Publisher name and URL are only written to `violations` when the
#   threshold is crossed.
# =============================================================================

import hashlib
import logging
import threading
import contextlib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column, String, Float, DateTime, Text, Boolean,
    ForeignKey, create_engine, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from config import DATABASE_URL, STORE_METADATA_ONLY_ON_VIOLATION

logger = logging.getLogger(__name__)

# ── ORM Base ──────────────────────────────────────────────────────────────────
Base = declarative_base()


# =============================================================================
# ADMINISTRATIVE MUTATION BYPASS  (added Day 2 — see tests/test_db_connector.py)
# =============================================================================
# DISCOVERED DURING TESTING (Day 2):
#   The NFR-04 immutability guard below blocks ANY UPDATE/DELETE against
#   `violations`, including legitimate ORM-level administrative operations
#   (e.g. removing a confirmed test row, or — in a real deployment — formally
#   superseding a violation that a human reviewer determined was a false
#   positive). The guard was blocking its OWN cleanup code in setup_db.py.
#
# DESIGN DECISION:
#   Rather than weaken the guard, we add an explicit, logged, opt-in bypass.
#   This preserves the chain-of-custody principle (NFR-04) — no mutation can
#   happen silently or by accident — while allowing a documented, auditable
#   administrative override when genuinely required. Every use of this bypass
#   writes a WARNING to the log, so it shows up in any audit trail review.
#
# This is intentionally NOT used anywhere in the live detection pipeline
# (main_pipeline.py never imports it). It exists only for setup/test/admin
# tooling, where cleaning up or correcting rows is a legitimate operation.
# =============================================================================
_mutation_bypass = threading.local()


@contextlib.contextmanager
def allow_violation_mutation(reason: str):
    """
    Administrative override for the violations-table immutability guard.

    Usage:
        with allow_violation_mutation("removing test fixture row"):
            session.query(Violation).filter(Violation.id == x).delete(
                synchronize_session=False
            )
            session.commit()

    Args:
        reason: Short human-readable justification. Written to the log for
                every use, so the override itself is auditable even though
                it is not stored in the database.
    """
    _mutation_bypass.active = True
    logger.warning(f"ADMIN OVERRIDE: violations-table mutation guard bypassed. Reason: {reason}")
    try:
        yield
    finally:
        _mutation_bypass.active = False


# =============================================================================
# TABLE 1: MediaLog
# =============================================================================
# Records every artifact the scraper ingests.
# Stores ONLY: a SHA-256 hash of the source URL (not the URL itself),
# the media type, and the ingestion timestamp.
# This satisfies NDPA data minimisation — no PII stored unless threshold met.
# =============================================================================
class MediaLog(Base):
    __tablename__ = "media_log"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url_hash    = Column(String(64), nullable=False, index=True,
                         comment="SHA-256 hash of source URL. Not the URL itself — NDPA §3.16.")
    media_type  = Column(String(10), nullable=False,
                         comment="'video' or 'text'")
    ingested_at = Column(DateTime(timezone=True), nullable=False,
                         default=lambda: datetime.now(timezone.utc))
    processed   = Column(Boolean, default=False,
                         comment="Set True once pipeline completes")

    # One-to-one: if a violation was raised, it links back here
    violation = relationship("Violation", back_populates="media_log_entry",
                             uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MediaLog id={self.id} type={self.media_type} hash={self.url_hash[:12]}...>"


# =============================================================================
# TABLE 2: Violation  (APPEND-ONLY — no UPDATE, no DELETE)
# =============================================================================
# Only created when Psynth >= PSYNTH_THRESHOLD (0.85).
# This is where publisher metadata is first written — consistent with
# NDPA data minimisation in §3.16.
# =============================================================================
class Violation(Base):
    __tablename__ = "violations"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_log_id    = Column(UUID(as_uuid=True), ForeignKey("media_log.id"), nullable=False)

    # Detection results
    psynth_score    = Column(Float, nullable=False,
                             comment="Final Psynth score from fusion module (§3.11)")
    p_visual        = Column(Float, nullable=True,
                             comment="Raw output of CV module (fCNN(Xvideo) → pvisual)")
    p_text          = Column(Float, nullable=True,
                             comment="Raw output of NLP module (fNLP(Xtext) → ptext)")
    violation_type  = Column(String(50), nullable=False,
                             comment="One of: DEEPFAKE_IMPERSONATION | "
                                     "UNDISCLOSED_SYNTHETIC_AVATAR | IP_SCRAPING_PLAGIARISM")

    # Statutory mapping (§3.13 Rules-as-Code)
    statute_cited   = Column(Text, nullable=False,
                             comment="e.g. 'NDPA 2023, Section 37'")
    statute_summary = Column(Text, nullable=False)
    regulator       = Column(String(100), nullable=False,
                             comment="Target regulatory body for the complaint")

    # Publisher metadata — ONLY stored here, ONLY when threshold crossed (NFR-05)
    source_url      = Column(Text, nullable=False,
                             comment="Full URL of offending content — written only on violation")
    publisher_name  = Column(String(200), nullable=True)
    detected_at     = Column(DateTime(timezone=True), nullable=False,
                             default=lambda: datetime.now(timezone.utc),
                             comment="Forensic timestamp for chain-of-custody (NFR-04)")

    # Full evidentiary detail (Day 16, §3.13) — added because p_visual/p_text
    # alone can't answer "what was actually observed": which CV heuristics
    # scored vs. abstained and why, the classifier's confidence, and (for
    # IP_SCRAPING_PLAGIARISM) which specific protected work and passage
    # matched. watchdog.complaint_generator.py needs this to state specifics
    # in the PDF rather than a bare probability, and NFR-04's chain-of-custody
    # requirement means that evidentiary basis must be durably recorded here
    # too, not just held in memory for the one process run that generated the
    # PDF. JSON-serialised (watchdog.fusion.FusionResult /
    # watchdog.ip_similarity.IPSimilarityResult are dataclasses, not ORM
    # types) rather than normalised into more columns — nullable, and this
    # table remains append-only (NFR-04) regardless of this addition.
    evidence_json   = Column(Text, nullable=True,
                             comment="JSON: fusion breakdown (per-heuristic/classifier detail) "
                                     "+ IP match detail backing this violation — §3.13 audit trail")

    # Relationships
    media_log_entry = relationship("MediaLog", back_populates="violation")
    report          = relationship("Report", back_populates="violation",
                                   uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return (f"<Violation id={self.id} type={self.violation_type} "
                f"psynth={self.psynth_score:.3f}>")


# =============================================================================
# TABLE 3: Report
# =============================================================================
# Records each auto-generated PDF complaint (§3.13 Document Generation).
# =============================================================================
class Report(Base):
    __tablename__ = "reports"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    violation_id = Column(UUID(as_uuid=True), ForeignKey("violations.id"), nullable=False)
    pdf_path     = Column(Text, nullable=False,
                          comment="Absolute path to generated PDF in reports/ directory")
    generated_at = Column(DateTime(timezone=True), nullable=False,
                          default=lambda: datetime.now(timezone.utc))
    status       = Column(String(20), default="PENDING_REVIEW",
                          comment="PENDING_REVIEW → DISPATCHED. Human officer reviews "
                                  "before dispatch (§3.13 Routing).")

    violation = relationship("Violation", back_populates="report")

    def __repr__(self):
        return f"<Report id={self.id} status={self.status}>"


# =============================================================================
# ENGINE + SESSION FACTORY
# =============================================================================

def get_engine(db_url: str = DATABASE_URL):
    """
    Creates and returns the SQLAlchemy engine.
    Uses check_same_thread=False only for SQLite (test environments).
    Production always uses PostgreSQL via DATABASE_URL.
    """
    connect_args = {}
    is_sqlite = db_url.startswith("sqlite")
    if is_sqlite:
        # SQLite is used only in unit tests (tests/conftest.py).
        # PostgreSQL is the production target (NFR-04).
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        echo=False,       # Set True to log all SQL (useful for debugging)
        pool_pre_ping=True  # Verify connection before using from pool
    )

    # ── ENFORCE FOREIGN KEYS on SQLite (parity with PostgreSQL) ──────────────
    # BUG FOUND IN PRODUCTION (Day 2 — real PostgreSQL run, not caught by the
    # SQLite-backed test suite): SQLite does NOT enforce foreign key
    # constraints unless explicitly told to via `PRAGMA foreign_keys=ON`.
    # PostgreSQL enforces them always. This meant a genuine ordering bug in
    # setup_db.py's cleanup routine (deleting a Violation while its child
    # Report row still physically existed) passed all 58 tests locally, then
    # threw psycopg2.errors.ForeignKeyViolation the first time it ran against
    # real PostgreSQL. Enabling this pragma closes that blind spot — SQLite
    # now enforces the same referential integrity rules production does, so
    # tests/test_db_connector.py can catch this whole category of bug again.
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    # ── WARN ON docker-compose.yml / actual-server VERSION DRIFT ─────────────
    # FOUND IN PRODUCTION (2026-09-01): docker-compose.yml starts
    # postgres:15-alpine, but a native Ubuntu install of PostgreSQL 14 was
    # already holding port 5432 on this machine, so the container's port
    # mapping silently never binds (`docker ps` still reports it "healthy" —
    # its healthcheck only probes from inside the container). The app has
    # been talking to the native 14.18 server the entire time; every real
    # violation row lives there, not in the container. Nothing failed, so
    # nothing surfaced this on its own — same silent-wrong-target shape as
    # bugs #6/#9/#10/#12 in docs/PROJECT_CONTEXT.md's bug log. This check
    # makes that drift visible the moment a connection is actually made,
    # rather than requiring someone to notice by chance (as happened here):
    # if the server this engine just connected to doesn't match what
    # docker-compose.yml declares, log it loudly. Non-fatal by design — a
    # version mismatch alone doesn't mean the wrong data is being used
    # (native 14 is, currently, the correct/intended target; see config.py's
    # DOCKER_COMPOSE_POSTGRES_VERSION comment) — but it is exactly the kind
    # of thing that must never fail silently a second time, since the
    # failure mode this guards is the live pipeline routing at a different,
    # possibly-empty database.
    if not is_sqlite:
        try:
            from config import DOCKER_COMPOSE_POSTGRES_VERSION
            with engine.connect() as _probe:
                _server_version = _probe.dialect.server_version_info
            if _server_version and _server_version[0] != DOCKER_COMPOSE_POSTGRES_VERSION:
                logger.warning(
                    f"DATABASE VERSION DRIFT: connected to PostgreSQL "
                    f"{'.'.join(str(p) for p in _server_version)}, but "
                    f"docker-compose.yml declares postgres:{DOCKER_COMPOSE_POSTGRES_VERSION}-alpine. "
                    f"This almost certainly means a native PostgreSQL install is holding port "
                    f"5432 and `docker compose up -d`'s container is not actually being used "
                    f"(its port mapping fails to bind silently when this happens — `docker ps` "
                    f"still reports it healthy). Confirm DATABASE_URL is pointing at the server "
                    f"that actually holds your data before trusting reads/writes."
                )
        except Exception as _probe_err:
            # Never let this diagnostic check block real startup — e.g. DB
            # genuinely unreachable, in which case the caller's own
            # connection attempt below will raise the real, actionable error.
            logger.debug(f"Skipped docker-compose version-drift check: {_probe_err}")

    # ── ENFORCE IMMUTABILITY on violations table (NFR-04) ─────────────────────
    # Attach a before_execute event listener that raises an error if any
    # UPDATE or DELETE is attempted on the violations table.
    # This is the technical enforcement of the chain-of-custody requirement.
    @event.listens_for(engine, "before_execute")
    def _block_violation_mutations(conn, clauseelement, multiparams, params, execution_options):
        # Administrative override active (see allow_violation_mutation() above) —
        # explicitly requested and already logged at the point of entry.
        if getattr(_mutation_bypass, "active", False):
            return

        sql_str = str(clauseelement).upper().strip()
        # Check if this is an UPDATE or DELETE targeting violations
        if any(sql_str.startswith(op) for op in ("UPDATE", "DELETE")):
            if "VIOLATIONS" in sql_str:
                raise PermissionError(
                    "NFR-04 VIOLATION: The `violations` table is immutable. "
                    "No UPDATE or DELETE operations are permitted to preserve "
                    "legal chain-of-custody. (Nigeria Data Protection Act 2023) "
                    "If this is a legitimate administrative operation, wrap it in "
                    "`with allow_violation_mutation(reason='...'): ...`"
                )

    return engine


def get_session_factory(engine=None):
    """Returns a SQLAlchemy sessionmaker bound to the engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _ensure_evidence_json_column(engine):
    """
    Additive-only migration for `violations.evidence_json` (added Day 16).
    Base.metadata.create_all() below only creates TABLES that don't exist
    yet — it never alters a table that's already there, so a database set
    up before this column was defined (this project's own dev Postgres
    included: it already had 2 Day-1 setup-verification rows from
    setup_db.py before evidence_json existed) would otherwise fail on the
    first real INSERT that tries to populate it.

    Only ever ADDS a nullable column — never drops or modifies an existing
    one, so this is consistent with NFR-04's append-only guarantee on
    `violations` (that guarantee is about row mutation, not schema
    evolution). No-ops if the table doesn't exist yet (create_all() just
    created it fresh, with the column already) or already has the column.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "violations" not in inspector.get_table_names():
        return
    existing_columns = {c["name"] for c in inspector.get_columns("violations")}
    if "evidence_json" in existing_columns:
        return

    logger.info("Migrating: adding violations.evidence_json column (Day 16, §3.13 audit trail)...")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE violations ADD COLUMN evidence_json TEXT"))


def init_db(engine=None):
    """
    Creates all tables in the database if they do not already exist, then
    applies any additive-only column migrations (see
    _ensure_evidence_json_column() above). Call this once during Day 1
    setup or on first run — and it's what main_pipeline.py calls too, so a
    schema added after Day 1 (like evidence_json) is picked up automatically
    on an existing database, not just a fresh one.

    Usage:
        from watchdog.db_connector import init_db
        init_db()
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    _ensure_evidence_json_column(engine)
    logger.info("Database tables created (or already exist): "
                "media_log, violations, reports")
    return engine


# =============================================================================
# DATA ACCESS FUNCTIONS
# =============================================================================

def log_media_ingestion(session, source_url: str, media_type: str) -> MediaLog:
    """
    Logs a new media artifact ingestion to media_log.

    NDPA data minimisation (NFR-05): Only the SHA-256 hash of the URL is stored,
    not the URL itself. The full URL is only written to `violations` if the
    Psynth threshold is crossed.

    Args:
        session:    Active SQLAlchemy session.
        source_url: The full URL of the scraped artifact.
        media_type: 'video' or 'text'.

    Returns:
        The newly created MediaLog ORM object.
    """
    url_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()

    entry = MediaLog(
        url_hash=url_hash,
        media_type=media_type,
        processed=False
    )
    session.add(entry)
    session.commit()
    logger.debug(f"MediaLog entry created: {entry}")
    return entry


def record_violation(
    session,
    media_log_entry: MediaLog,
    psynth_score: float,
    p_visual: float,
    p_text: float,
    violation_type: str,
    statute_cited: str,
    statute_summary: str,
    regulator: str,
    source_url: str,
    publisher_name: str = None,
    evidence_json: str = None,
    violation_id: uuid.UUID = None,
) -> Violation:
    """
    Records a confirmed violation (Psynth >= 0.85) to the violations table.

    This is the ONLY place where the source_url and publisher_name are written
    to permanent storage — consistent with NDPA data minimisation (NFR-05).

    Args:
        session:         Active SQLAlchemy session.
        media_log_entry: The corresponding MediaLog entry.
        psynth_score:    Final fused synthetic probability (§3.11).
        p_visual:        Raw CV module score.
        p_text:          Raw NLP module score.
        violation_type:  Key from STATUTORY_MAP in config.py.
        statute_cited:   Human-readable statute name.
        statute_summary: Plain-English description of the violation.
        regulator:       Target agency (NITDA / ARCON / NDPC / NCC).
        source_url:      Full URL of offending content (written here only).
        publisher_name:  Optional: name of the publisher/brand.
        evidence_json:   Optional: JSON string of the full evidentiary detail
                         behind this violation (watchdog.rules_engine /
                         watchdog.complaint_generator build this) — §3.13
                         audit trail, so the complaint's basis survives after
                         the process that generated it exits.
        violation_id:    Optional: pre-assign the row's UUID rather than
                         letting the Column default generate one at flush
                         time. main_pipeline.py needs the id BEFORE this
                         insert — it goes into the PDF/evidence_json's own
                         content — and the violations table's append-only
                         guard (NFR-04) means there is no legitimate way to
                         UPDATE it in afterwards once inserted without an id.

    Returns:
        The newly created Violation ORM object.
    """
    violation = Violation(
        media_log_id=media_log_entry.id,
        psynth_score=psynth_score,
        p_visual=p_visual,
        p_text=p_text,
        violation_type=violation_type,
        statute_cited=statute_cited,
        statute_summary=statute_summary,
        regulator=regulator,
        source_url=source_url,
        publisher_name=publisher_name,
        evidence_json=evidence_json,
    )
    if violation_id is not None:
        violation.id = violation_id
    session.add(violation)

    # Mark the media log entry as processed
    media_log_entry.processed = True
    session.commit()

    logger.info(
        f"Violation recorded: type={violation_type}, "
        f"psynth={psynth_score:.3f}, statute='{statute_cited}'"
    )
    return violation


def record_report(session, violation: Violation, pdf_path: Path) -> Report:
    """
    Records a generated PDF complaint in the reports table.

    Args:
        session:   Active SQLAlchemy session.
        violation: The parent Violation ORM object.
        pdf_path:  Path object pointing to the generated PDF file.

    Returns:
        The newly created Report ORM object.
    """
    report = Report(
        violation_id=violation.id,
        pdf_path=str(pdf_path.resolve()),
        status="PENDING_REVIEW"
    )
    session.add(report)
    session.commit()
    logger.info(f"Report record created: pdf={pdf_path.name}, status=PENDING_REVIEW")
    return report


def get_all_violations(session):
    """Returns all violations ordered by detection time (most recent first)."""
    return (
        session.query(Violation)
        .order_by(Violation.detected_at.desc())
        .all()
    )


def get_violation_by_id(session, violation_id: str) -> Violation | None:
    """
    Fetches a single violation by its UUID string.

    BUG FIX (discovered via tests/test_db_connector.py, Day 2):
    The `id` column is UUID(as_uuid=True), which binds parameters as real
    uuid.UUID objects, not plain strings. Callers of this function pass a
    string (e.g. from a Flask URL route: /reports/<violation_id>), so we
    must explicitly coerce it here — otherwise SQLAlchemy's bind processor
    raises AttributeError('str' object has no attribute 'hex') deep inside
    the query, which is a confusing failure mode for an API caller.

    Also now returns None (rather than raising) for a syntactically invalid
    UUID string, since that is effectively the same as "not found" from the
    caller's point of view (e.g. a malformed URL parameter in the dashboard).

    Args:
        session:      Active SQLAlchemy session.
        violation_id: UUID as a string (e.g. from a URL path parameter).

    Returns:
        The matching Violation, or None if not found / not a valid UUID.
    """
    try:
        normalised_id = uuid.UUID(str(violation_id))
    except (ValueError, AttributeError, TypeError):
        logger.warning(f"get_violation_by_id: '{violation_id}' is not a valid UUID")
        return None

    return session.query(Violation).filter(
        Violation.id == normalised_id
    ).first()
