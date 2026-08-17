# Naija AI Ethics Watchdog

B.Tech final year project, Arukwe Chibuikem David (20211285585), Dept. of Computer
Science, SCIT Owerri. Detects AI-generated synthetic media and AI-plagiarised IP in
Nigerian digital content, then auto-generates PDF legal complaints citing Nigerian
statutes.

**The project write-up is the specification.** Section references like §3.12 or
NFR-04 throughout the code point to it. When changing behaviour that a § reference
describes, say so explicitly — the code and the document must not drift apart.

## Hard constraints

- **No GPU. CPU-only inference.** Model *training* happens in Google Colab (free
  T4); checkpoints are downloaded into `models/`. Never add code assuming local CUDA.
  (The write-up's §3.15 claims an RTX 3080 — that is aspirational, not real. §1.7
  Limitations is the honest statement.)
- **PostgreSQL, not SQLite,** in production (NFR-04). SQLite is tests only.
- Running under WSL2. Paths, browser access, and filesystem quirks follow from that.

## Commands

```bash
source venv/bin/activate          # always first
docker compose up -d              # Postgres; before setup_db.py
python setup_db.py                # creates 3 tables, verifies NFR-04 guard
pytest tests/ -v                  # MUST pass before any task is "done"
python collect_dataset.py --stats # dataset progress vs Table 3.4
```

## Rules

1. **Run `pytest tests/` before declaring anything finished.** Not optional.
2. **Scripts must stay idempotent.** Re-running any collection or generation script
   must skip existing work, never duplicate or overwrite it. This has caused real
   data loss before (see bug log).
3. **`cookies.txt` holds live session credentials.** Already gitignored. Never
   commit it, never print its contents, never paste it into output.
4. **`data/manifest.csv` is the source of truth** for dataset state. Files on disk
   without manifest rows, or rows without files, are bugs.
5. **The `violations` table is append-only** (NFR-04, legal chain-of-custody). A
   SQLAlchemy event listener blocks UPDATE/DELETE. Use the
   `allow_violation_mutation()` context manager only for genuine admin operations,
   never in the live pipeline.
6. **Don't relabel dataset samples without asking.** Ground-truth labels were
   assigned by human review; changing them silently corrupts the experiment.
7. **Prefer fixing root causes over widening thresholds.** Several config values
   (`PSYNTH_THRESHOLD`, `BURSTINESS_THRESHOLD`) came from the write-up or from
   guesses made before real data existed. If one looks wrong, say so rather than
   quietly tuning it to make a number look better.

## Layout

```
watchdog/          core package — scraper, preprocessor, db_connector,
                   dataset_manager, nlp_engine
config.py          ALL thresholds and paths. No magic numbers elsewhere.
seed_sources.py    curated dataset URLs (human-verified, not crawled)
collect_dataset.py scrape → save → manifest → interactive labelling
tests/             pytest suite
docs/              PROJECT_CONTEXT.md, schema_erd.md
```

## Deeper context

`docs/PROJECT_CONTEXT.md` has the full history: design decisions and their
rationale, a log of every real bug found and fixed (read this before touching
the scraper or db_connector), known data-quality problems, and current status
against the 28-day roadmap. Read it before non-trivial work.
