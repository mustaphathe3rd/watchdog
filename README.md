# Naija AI Ethics Watchdog

**Development of an Automated AI Ethics Watchdog for Regulatory Compliance and Intellectual Property Protection in Nigeria**

> Arukwe Chibuikem David — 20211285585  
> Department of Computer Science, SCIT Owerri  
> B.Tech Final Year Project — May 2026  
> Supervisor: Dr. O.A. Njoku

---

## Project Summary

The Naija AI Watchdog is a Python-based automated detection and enforcement system designed to identify AI-generated synthetic media (deepfakes, undisclosed AI avatars) and AI-plagiarised intellectual property in the Nigerian digital space. When a violation is confirmed (Psynth ≥ 0.85), the system automatically generates a formal PDF legal complaint citing the specific Nigerian statute violated (NDPA 2023 / Copyright Act 2022 / ARCON guidelines) and routes it to the relevant regulatory body.

---

## Architecture

```
Scraper → Preprocessor → [CV Engine + NLP Engine] → Fusion → Rules Engine → Complaint Generator → DB
```

Core modules:
| Module | File | Role |
|---|---|---|
| NLP Detection | `watchdog/nlp_engine.py` | Perplexity + burstiness heuristics (Day 3), fine-tuned DistilBERT classifier — training (Day 8-9) and live inference (`classify_text()`, Day 16) |
| IP Similarity | `watchdog/ip_similarity.py` | Protected-work matching → Copyright Act 2022 s.36(2) evidence (Day 7) |
| CV Detection | `watchdog/cv_engine.py` | MTCNN/Face Mesh tracking + four Stage 4 heuristics (Day 3-4) |
| Fusion | `watchdog/fusion.py` | §3.11 weighted fusion → Psynth, with correct missing-modality handling (Day 16) |
| Rules Engine | `watchdog/rules_engine.py` | Statutory mapping — NDPA 2023 s.37, Copyright Act 2022 s.36(2)/s.37(2), ARCON guidelines (Day 16) |
| Complaint Generator | `watchdog/complaint_generator.py` | §3.13 ReportLab PDF, with a mandatory human-review caveat block (Day 16) |

---

## Day 1 Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd naija_ai_watchdog
python3.10 -m venv venv
source venv/bin/activate      # Linux/macOS
# OR: venv\Scripts\activate   # Windows
```

### 2. Install dependencies (CPU-only, no GPU required)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL and FLASK_SECRET_KEY
```

### 4. Set up PostgreSQL

**Option A — Docker (recommended, works identically on any OS):**

```bash
docker compose up -d
```

**Option B — Native install:**

```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE USER watchdog_user WITH PASSWORD 'watchdog_pass';"
sudo -u postgres psql -c "CREATE DATABASE naija_watchdog OWNER watchdog_user;"
```

### 5. Initialise the database

```bash
python setup_db.py
```

Expected output: `DAY 1 DATABASE SETUP COMPLETE`

---

## Day 2 — Dataset Collection

Day 2 adds three modules on top of the Day 1 database layer:

| File | Purpose |
|---|---|
| `watchdog/dataset_manager.py` | Manages `data/manifest.csv` — the label index for Table 3.4's three dataset categories |
| `seed_sources.py` | Curated Nigerian domains + a queue of specific URLs to scrape (you populate this) |
| `collect_dataset.py` | CLI orchestrator: scrape → save → log → label |

### Workflow

**1. Populate seed sources.** Open `seed_sources.py` and add real, verified
URLs to the `*_SPECIFIC_URLS` lists (each category has usage guidance in
its docstring — e.g. how to source authentic vs. synthetic samples for the
`TELECOM_FMCG_ADS` category per §3.9).

**2. Preview before downloading anything:**

```bash
python collect_dataset.py --dry-run
```

**3. Run the actual collection:**

```bash
python collect_dataset.py
```

Each URL is scraped via `watchdog/scraper.py` (Day 1), saved under
`data/raw/{video,text}/`, and logged as an unlabelled row in
`data/manifest.csv`. Re-running is safe — already-collected URLs are
skipped automatically.

**4. Label what you collected:**

```bash
python collect_dataset.py --label
```

An interactive terminal session walks through every unlabelled sample —
`0` for authentic/human, `1` for synthetic/AI, matching the Table 3.4
label convention.

**5. Check progress at any time:**

```bash
python collect_dataset.py --stats
```

```
  DATASET MANIFEST — Current Status (cf. Table 3.4)
  Total samples:      147
  Labelled:           132
  Awaiting label:     15
  ------------------------------------------------------------
  TELECOM_FMCG_ADS         42/3000  ( 1.4%)  | authentic=20 synthetic=22
  POLITICAL_MEDIA          18/1000  ( 1.8%)  | authentic=9  synthetic=9
  TEXTUAL_CORPUS           87/6000  ( 1.5%)  | authentic=45 synthetic=42
```

### Downloading from Facebook / Instagram

Both platforms reject anonymous automated video downloads (`yt-dlp` will
fail with `login required` or `No video formats found`). Authenticate as
your own account using ONE of these:

**Option 1 — pull cookies live from a browser (easiest):**

```bash
# In .env:
YTDLP_COOKIES_FROM_BROWSER=chrome
```

Just needs that browser installed and logged into Facebook/Instagram on
this machine. No export step.

**Option 2 — manually export a cookies.txt:**

Install a browser extension like *"Get cookies.txt LOCALLY"* (Chrome/Firefox),
log into Facebook and Instagram, export cookies for both domains into one
file, save it as `cookies.txt` in the project root, then:

```bash
# In .env:
YTDLP_COOKIES_FILE=cookies.txt
```

`cookies.txt` is already in `.gitignore` — **never commit it**, it contains
live session authentication equivalent to your password.

**A caution worth taking seriously:** both platforms' Terms of Service
technically prohibit automated access even with your own cookies. This is
reasonable for the occasional, deliberate single-video pull this project
needs — not something to loop aggressively across hundreds of URLs, which
risks a temporary rate-limit or flag on your account.

### Running tests

```bash
pytest tests/ -v
```

58 tests cover the database layer (NFR-04 immutability, NFR-05 data
minimisation), the manifest manager, and the preprocessing pipeline. Video
tests generate tiny synthetic clips on the fly via `cv2.VideoWriter` — no
real footage is needed to run the suite.



## Compute Strategy

| Phase | Environment | Reason |
|---|---|---|
| Model Training (NLP + CV) | Google Colab (free T4 GPU) | §1.7 Limitations — no local GPU |
| Inference + Pipeline | Local CPU | All other modules run on CPU |
| Dashboard | Local Flask server | Admin review before complaint dispatch |

**Training workflow:**
1. Upload `data/` to Google Drive
2. Open `notebooks/02_nlp_training.ipynb` or `03_cv_training.ipynb` in Colab
3. Mount Drive, run all cells
4. Download `nlp_watchdog_v1.pt` and `cv_watchdog_v1.pt` to `models/`

---

## Tech Stack

| Library | Purpose |
|---|---|
| `opencv-python` | Frame extraction (`cv2.VideoCapture`) — §3.12 A |
| `mtcnn` | Facial detection and alignment — FR-02 |
| `transformers` | DistilBERT for NLP perplexity scoring — FR-04 |
| `sentence-transformers` | IP cosine similarity matrix — §3.12 B |
| `torch` (CPU build) | Model inference |
| `reportlab` | PDF auto-complaint generation — FR-06 |
| `flask` | Admin dashboard — §3.13 Routing |
| `postgresql` + `SQLAlchemy` | Immutable audit log — NFR-04 |
| `scikit-learn` | Evaluation metrics (F1, Precision, Recall) — §3.14 |

---

## Non-Functional Targets (§3.4)

**Full verification table with sources**: `results/nfr_verification_table.csv`,
generated (and cross-checked against the raw per-sample data, not just the
summary CSVs) by `generate_chapter4_evaluation.py` — the table below is a
condensed view of the same numbers.

| NFR | Target | Achieved |
|---|---|---|
| NFR-01 F1-Score — text pathway (NLP classifier, the only pathway with a reliable in-domain score) | ≥ 0.90 | **0.867 measured, target not met** — 5-fold group-CV on the real Nigerian textual corpus (`results/nlp_classifier_fold_metrics.csv` AGGREGATE row; precision 0.897, recall 0.839; cross-checked against the raw per-sample predictions in `results/nlp_classifier_cv_eval.csv` — see the confusion matrix, `results/confusion_matrix_nlp_classifier.png`). |
| NFR-01 F1-Score — video pathway (CV classifier) | ≥ 0.90 (gate) | **Gate not cleared — closed as a documented negative result (2026-08-12).** Best held-out video *accuracy* (not F1, and on FF++, not this project's own corpus) was 0.86 across three scheduled training runs (0.86 / 0.78 / 0.79) — see `docs/PROJECT_CONTEXT.md` §4. No checkpoint exists; no Nigerian-corpus CV number was ever produced. |
| NFR-01 F1-Score — fused, system-level | ≥ 0.90 | **Not computable, not just unmeasured.** No sample in this project's corpus has ever had both modalities scored together — `data/manifest.csv`'s `media_type` is video *or* text per row, never both — so `fusion.compute_psynth()`'s two-modality branch has never executed against real data. There is nothing to average into one fused number; see `generate_chapter4_evaluation.py`'s `build_nfr_table()` docstring for the full reasoning. |
| NFR-02 Text latency (end-to-end, `main_pipeline.process_media()` via `/upload`) | ≤ 5s | **Holds warm — 1.560s-2.625s** (dashboard, three consecutive submissions through one continuously-running process, 2026-08-13: 32.584s cold / 2.625s / 1.560s warm, confirming the 2026-08-13 profiling prediction of ~2.0-2.3s and closing out the earlier 16.882s reading as a post-restart cold classifier load, not a real per-request cost). **Cold fails (32.6s-36.5s)**, disclosed as one-time model loading (DistilBERT classifier + DistilGPT-2 + IP-similarity sentence-transformer), not a per-request cost — same posture as the video row below. A same-day fix to a `watchdog/preprocessor.py` caching bug (spaCy's pipeline was reloading on every call, unlike every other model loader in this codebase) took the warm figure well under 1.5s in isolated re-profiling; see `docs/PROJECT_CONTEXT.md` §4 for both the original 16.882s misread and the fix. |
| NFR-02 Video latency, full pipeline (`main_pipeline.process_media()` via `/upload`) | ≤ 30s | **Holds — cold 32.062s (one-time model-load cost, not a per-request cost), warm 4.886s** (dashboard, `results/dashboard_pipeline_latency.csv`). Supersedes the Stage-2-only figure below now that the full end-to-end path is timed. |
| NFR-02 Video latency (Stage 2 face pipeline only, superseded by the row above) | ≤ 30s | **Holds — 32/32 CV-eligible clips measured, all under 30s** (real `analyze_clip()` wall time, not projected; slowest is 23.8s). Real corpus spans 20.0s–128.4s after 2026-08-03 trimming (was up to 193.4s) — see `docs/PROJECT_CONTEXT.md` §4. One clip (`1aade253`) was excluded from CV analysis entirely as too choppily-edited to support continuous face-tracking, not because of latency. This covers only the sparse-MTCNN/dense-Face-Mesh stage in `watchdog/cv_engine.py`; classifier inference and audio viseme-sync aren't built yet and will add to this. |
| NFR-03 FPR on Nigerian English — Day 3 heuristic vote (perplexity+burstiness) | ≤ 5% | **38.2% measured, target not met** (13/34 authentic samples flagged AI — `results/nlp_heuristic_eval.csv`). DistilGPT-2's Western training distribution scores real Nigerian journalism as unexpectedly low-perplexity/in-distribution, which the conservative perplexity+burstiness vote reads as an "AI" signal — the register-bias finding, see `results/register_bias_perplexity_by_genre.png` and `docs/PROJECT_CONTEXT.md` §4. |
| NFR-03 FPR on Nigerian English — fine-tuned NLP classifier (the deployed `p_text` signal) | ≤ 5% | **8.8% measured (3/34), target still not met — but a 4.3x improvement over the heuristic vote** (`results/nlp_classifier_fold_metrics.csv` AGGREGATE row's `fpr_authentic`; cross-checked against `results/nlp_classifier_cv_eval.csv`). Report both figures honestly rather than only the better one: localisation (fine-tuning on real Nigerian-labelled samples) closed most of the gap the heuristic pair left open, without fully closing it. |
| NFR-04 Immutable audit log | PostgreSQL append-only | ✓ |
| NFR-05 NDPA data minimisation | No PII stored below threshold | ✓ |

---

## Legal Basis

| Violation Type | Nigerian Statute |
|---|---|
| Deepfake impersonation | NDPA 2023, Section 37 |
| Undisclosed synthetic avatar | ARCON Guidelines (2023) + NDPA s.37 |
| IP scraping / plagiarism | Copyright Act 2022, Section 36(2) (infringement) / Section 37(2) (remedies) |

---

## Project Structure

```
naija_ai_watchdog/
├── data/                     # Datasets (see §3.9)
│   ├── manifest.csv          # Label index — created by dataset_manager.py
│   ├── raw/{video,text}/     # Downloaded/scraped artifacts
│   └── protected_ip/         # Nigerian creative works for IP similarity check
├── docs/
│   └── schema_erd.md         # Entity-Relationship reference (→ Chapter 4 Fig 4.X)
├── models/                   # Trained .pt checkpoints (Colab → local)
├── watchdog/                 # Core Python package
│   ├── scraper.py            # Media acquisition (Day 1)
│   ├── preprocessor.py       # Frame extraction + tokenisation (Day 1)
│   ├── dataset_manager.py    # manifest.csv read/write/label (Day 2)
│   ├── nlp_engine.py         # Perplexity + burstiness (Day 3), DistilBERT train/infer (Day 8-9/16)
│   ├── ip_similarity.py      # Protected-work cosine-similarity matcher (Day 7)
│   ├── cv_engine.py          # MTCNN + Stage 4 heuristics (Day 3-4)
│   ├── fusion.py             # Weighted fusion → Psynth (Day 16)
│   ├── rules_engine.py       # Statutory mapping (Day 16)
│   ├── complaint_generator.py  # ReportLab PDF builder (Day 16)
│   └── db_connector.py       # PostgreSQL ORM (Day 1, extended Day 2 + Day 16)
├── dashboard/                # Flask admin UI
├── reports/                  # Generated PDF complaints
├── tests/                    # pytest test suites (58 tests)
│   ├── conftest.py           # Shared fixtures (in-memory SQLite per test)
│   ├── test_db_connector.py  # NFR-04 / NFR-05 coverage
│   ├── test_dataset_manager.py
│   └── test_preprocessor.py
├── notebooks/                # Jupyter training + evaluation
├── seed_sources.py           # Curated URL queue for dataset collection (Day 2)
├── collect_dataset.py        # Dataset collection CLI orchestrator (Day 2)
├── main_pipeline.py          # Orchestrator (Day 16)
├── setup_db.py               # Day 1 DB initialisation
├── docker-compose.yml        # Reproducible local PostgreSQL (Day 2)
├── config.py                 # Central configuration
└── requirements.txt          # All dependencies
```
