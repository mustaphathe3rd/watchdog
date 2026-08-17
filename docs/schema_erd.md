# Database Schema — Entity-Relationship Reference

This documents the schema built on Day 2 (`watchdog/db_connector.py`). It will
become **Fig 4.X: E-R Diagram** in Chapter 4 §4.1 — draw this exact structure
in your diagramming tool of choice (draw.io, Lucidchart, or a hand-drawn +
scanned figure) using standard crow's-foot notation.

## Entities and Relationships

```
┌────────────────────────┐
│      media_log          │
├────────────────────────┤
│ PK id            (UUID) │
│    url_hash    (String) │◄── SHA-256 hash only (NFR-05 minimisation,
│    media_type  (String) │     never the raw URL — see §3.16)
│    ingested_at (DateTime)│
│    processed   (Boolean)│
└───────────┬─────────────┘
            │ 1
            │
            │ 0..1
┌───────────▼─────────────┐
│       violations         │   (APPEND-ONLY — NFR-04)
├──────────────────────────┤
│ PK id             (UUID) │
│ FK media_log_id   (UUID) │
│    psynth_score  (Float) │◄── §3.11: ffusion(pvisual, ptext) → ŷ
│    p_visual      (Float) │
│    p_text        (Float) │
│    violation_type(String)│◄── DEEPFAKE_IMPERSONATION |
│                          │    UNDISCLOSED_SYNTHETIC_AVATAR |
│                          │    IP_SCRAPING_PLAGIARISM
│    statute_cited  (Text) │◄── §3.13 Rules-as-Code mapping
│    statute_summary(Text) │
│    regulator     (String)│◄── NDPC | ARCON | NCC
│    source_url      (Text)│◄── Written ONLY here, ONLY on violation
│    publisher_name(String)│
│    detected_at (DateTime)│◄── Forensic timestamp, chain-of-custody
│    evidence_json   (Text)│◄── Day 16 (§3.13): JSON fusion + IP-match
│                          │    detail, nullable — see note 5 below
└───────────┬──────────────┘
            │ 1
            │
            │ 0..1
┌───────────▼─────────────┐
│         reports          │
├──────────────────────────┤
│ PK id             (UUID) │
│ FK violation_id   (UUID) │
│    pdf_path        (Text)│
│    generated_at(DateTime)│
│    status        (String)│◄── PENDING_REVIEW | DISPATCHED
└──────────────────────────┘
```

## Cardinality Summary

| Relationship | Cardinality | Rationale |
|---|---|---|
| `media_log` → `violations` | 1 : 0..1 | Most ingested media is authentic and never becomes a violation. Only content crossing `PSYNTH_THRESHOLD` (0.85) gets a row. |
| `violations` → `reports` | 1 : 0..1 | A violation always eventually gets exactly one PDF complaint (§3.13), but there's a window between detection and generation where it doesn't have one yet. |

## Design Decisions Worth Citing in Chapter 4

1. **UUID primary keys, not auto-increment integers** — prevents an external
   party from guessing sequential violation IDs (e.g. `/reports/1`,
   `/reports/2`) and enumerating the entire compliance log, which would be
   inappropriate for a system handling pre-publication legal complaints.

2. **`media_log.url_hash` instead of a raw URL column** — the schema
   *structurally* enforces NDPA data minimisation (NFR-05). It isn't
   possible to accidentally query out a raw URL for non-violating content,
   because the column simply doesn't exist on that table.

3. **`violations` is intentionally append-only** — enforced in code (a
   SQLAlchemy `before_execute` event listener rejects any `UPDATE`/`DELETE`
   targeting the table — see `db_connector.py`), not just by convention.
   This is the literal technical implementation of the "immutable...legal
   chain-of-custody" requirement in NFR-04.

4. **`reports.status` as a simple string state machine**
   (`PENDING_REVIEW` → `DISPATCHED`) reflects §3.13's requirement that a
   human regulatory officer reviews every auto-drafted complaint before it
   is sent — the system never dispatches directly and autonomously.

5. **`violations.evidence_json` (added Day 16, `watchdog/fusion.py` +
   `watchdog/complaint_generator.py`)** — `p_visual`/`p_text` alone can't
   answer "what was actually observed": which of the four Stage 4 CV
   heuristics scored vs. abstained and why, the text classifier's raw
   confidence, or — for `IP_SCRAPING_PLAGIARISM` — which specific protected
   work and passage matched. Rather than normalise all of that into more
   narrow columns, it's JSON-serialised into one nullable `Text` column:
   `watchdog.fusion.FusionResult` and `watchdog.ip_similarity.IPSimilarityResult`
   are plain dataclasses, not ORM types, and the shape of "evidence" differs
   meaningfully by violation type. This keeps the append-only guarantee
   (NFR-04) intact — it's an additional column on insert, not a mutation
   path — while making sure the PDF complaint's full evidentiary basis
   survives after the process that generated it exits, not just the file
   path to the PDF itself.
