# Write-up Corrections

Consolidated list of every change the project write-up needs, derived from
`docs/PROJECT_CONTEXT.md`'s full build history (not from memory), cross-checked
against `config.py`, `README.md`, and the `results/*.csv` files those entries
cite. Compiled 2026-08-17, ahead of Days 26-28 (Write-up).

**How to read this list.** Every item is tagged:

- **⚠️ CORRECTION** — the document currently states something that is false, or
  presents an unmeasured/aspirational number as if it were a result. **Viva
  risk**: if asked to defend the number or claim as written, it will not hold up.
- **✚ ADDITION** — a real finding exists that the document doesn't mention yet.
  Nothing currently written is false, but the document is leaving results on
  the table — several of these are the project's strongest original
  contributions and are currently unclaimed.

Every item names the file and specific result it rests on. Where PROJECT_CONTEXT.md
is the primary source, it's cited by section/entry name rather than line number,
since the document is expected to keep growing.

---

## Priority summary

| # | Item | Chapter | Type |
|---|---|---|---|
| 1 | Abstract's 96.3% accuracy figure | Abstract | ⚠️ CORRECTION |
| 2 | §3.15 hardware — RTX 3080 | Ch.3 | ⚠️ CORRECTION |
| 3 | Table 3.4 — 10,000-sample target | Ch.3 | ⚠️ CORRECTION |
| 4 | Copyright Act citation — s.108 vs s.36 | Ch.3 | ⚠️ CORRECTION |
| 5 | SCOT theory misattributed | Ch.2 | ⚠️ CORRECTION |
| 6 | §3.12 A CV engine — MediaPipe API | Ch.3 | ⚠️ CORRECTION |
| 7 | NFR table figures unmeasured/stale | Ch.3/4 | ⚠️ CORRECTION |
| 8 | CV classifier — negative result | Ch.4/1/5 | ✚ ADDITION |
| 9 | Register-bias finding (DistilGPT-2) | Ch.4/1/2 | ✚ ADDITION |
| 10 | Fusion abstention contract | Ch.3 | ✚ ADDITION |
| 11 | IP similarity threshold validation | Ch.3/4 | ✚ ADDITION |
| 12 | Length-confound test | Ch.4 | ✚ ADDITION |
| 13 | Cross-LLM generalisation gap | Ch.1/5 | ✚ ADDITION |
| 14 | CV transfer-learning access, blocked | Ch.1/5 | ✚ ADDITION |
| 15 | Engineering bug findings | Ch.4 | ✚ ADDITION |
| 16 | Flask dashboard build | Ch.3/4 | ✚ ADDITION |
| 17 | Outstanding: Ch.4 screenshots | — | task, not a write-up error |
| 18 | "Sufficient effort" → "some effort" (Copyright Act 2022) | Ch.1/2 | ⚠️ CORRECTION |
| 19 | §1.7 mid-tier GPU / 4K video claims | Ch.1 | ⚠️ CORRECTION |
| 20 | SCOT misattribution also in Table 2.2 | Ch.2 | ⚠️ CORRECTION |
| 21 | §3.9 prose claims Tinubu clip collected | Ch.3 | ⚠️ CORRECTION |
| 22 | Table 3.1 "milliseconds per content frame" | Ch.3 | ⚠️ CORRECTION |
| 23 | §3.3.1 "effectively eliminate[d]" FP bias | Ch.3 | ⚠️ CORRECTION |
| 24 | Table 3.6's four targets all missed | Ch.3 | ✚ ADDITION |
| 25 | ToC/document missing Ch.4-5 | — | 🖼️ PRESENTATION |
| 26 | Figures 3.1-3.5 render as placeholder letters | Ch.3 | 🖼️ PRESENTATION |
| 27 | Ampifire/Amplifier; NITDA 2024/2026 | Ch.1/2 | 🖼️ PRESENTATION |
| 28 | Fadolapo misattributed as author, wrong year | Ch.1/2 | ⚠️ CORRECTION |
| 29 | 10 of 34 references never cited in text | — | 🖼️ PRESENTATION |
| 30 | System name drifts three ways | Title/Abstract/Ch.2 | ⚠️ CORRECTION |
| 31 | Table 3.8 — "PostgreSQL 15 via Docker Compose" vs. real native 14.18 | Ch.3 | ⚠️ CORRECTION |
| 32 | No idempotency check — double-submit produces duplicate legal complaints | Ch.1/4/5 | ✚ ADDITION |

Items 1–7 are the viva-risk group — a panel member who checks any of these
against the code or the data will find the document wrong. Fix these first.
Items 8–16 are upside: real, defensible findings this project produced that
aren't written up yet, several of which (9, 8) are stronger material than
what they'd replace. Items 18–30 (**Document-level corrections**, below) are
a second pass compiled by reading the write-up PDF directly rather than the
build history — items 18, 19, 21, 22, 23, 28, 30 are the same kind of
viva-risk as 1–7; 24, 27, 29 are internal-consistency gaps; 25 and 26 are
presentation/rendering defects a panel sees on the page without checking
anything against the code. Item 31 (found 2026-09-01, after this list was
first compiled) is viva-risk in the same class as 1–7 and 4 specifically —
a panel member who runs `docker ps` or `SELECT version()` against the
system, the way item 4's Section 108 error was only found by checking the
statute directly, will find Table 3.8 wrong.

---

## Abstract

### 1. ⚠️ CORRECTION — unmeasured 96.3% accuracy figure

**Document claims:** An overall accuracy figure of 96.3% (source of this
number is not traceable to any script, log, or results file in this repo).

**Actually happened:** No run in this project ever produced 96.3% at
anything. The real, measured, held-out numbers are per-pathway, because no
fused system-level number is computable (see item 7 below):

- Text (NLP) pathway: **F1 0.867**, precision 0.897, recall 0.839, pooled
  5-fold group-CV over all 65 labelled `TEXTUAL_CORPUS` samples.
  Evidence: `results/nlp_classifier_fold_metrics.csv` (AGGREGATE row),
  `results/nlp_classifier_cv_eval.csv` (raw per-sample), confusion matrix in
  `results/confusion_matrix_nlp_classifier.png` (31 TN / 3 FP / 5 FN / 26 TP,
  n=65).
- Video (CV) pathway: **no number is reported at all** — the 0.90 credibility
  gate was never cleared after three scheduled training runs (best 0.86).
  See item 8.

**Action:** Remove 96.3% and replace with the two real numbers above, stated
per-pathway with the "no fused figure exists" caveat from item 7. This is the
single highest-risk line in the whole document — an examiner asking "walk me
through how you got 96.3%" currently has no honest answer.

---

## Chapter 1 — Introduction

### 13. ✚ ADDITION — cross-LLM generalisation is untested, not a minor caveat

**Document currently says:** (implicitly, by omission) the classifier detects
AI-generated text.

**What actually happened:** All 31 synthetic `TEXTUAL_CORPUS` samples were
generated by one LLM family (Claude). No API key for any other provider
(GPT/Gemini/Llama) exists in this project's environment, and this was
checked directly rather than assumed — no substitute or same-LLM data was
used to fake the test. This is a genuine, currently-open limitation: nothing
rules out the classifier having learned Claude's house style specifically
rather than general LLM-authorship signal.

**Evidence:** `docs/PROJECT_CONTEXT.md` §4, "Cross-LLM generalisation —
UNTESTED, not just unresolved (2026-08-04)".

**Action:** §1.7 Limitations needs this stated explicitly, distinct from
the length-confound caveat (which *was* tested and cleared — see item 12).
Don't let the two get merged into one softer sentence; they're different
claims with different evidentiary status.

### 14. ✚ ADDITION — CV transfer-learning benchmark access is blocked, not skipped

**What actually happened:** The planned Days 10-15 FaceForensics++/Celeb-DF
transfer experiment required real, human-reviewed dataset access requests
(Google Form → manual TU Munich/Tencent review → signed ToU), not something
obtainable inside an automated session. A HuggingFace mirror of FF++ was
found and deliberately *not* used, since it's very likely an unauthorized
redistribution of a dataset whose own terms prohibit that — exactly the kind
of indefensible-in-a-viva shortcut this project's own scope argument (§1.6)
rules out.

**Evidence:** `docs/PROJECT_CONTEXT.md` §4, "CV transfer-learning
experiment — BLOCKED, deferred (2026-08-04)".

**Action:** §1.7 should state this as a scope limitation with the reason
(institutional-access gating on both standard benchmarks, not researcher
inaction), separate from item 8's negative training result — access being
blocked and the model underperforming once trained are two different facts,
both true, and both belong in the document.

### 32. ✚ ADDITION — no idempotency/deduplication check: a double-submit produces two separate, permanent legal complaints for the same content

**What actually happened:** `dashboard/app.py`'s `/upload` route (both the
`sample_id` and live-`url` submission modes) has no deduplication of any
kind against `media_log.url_hash` or an existing `violations` row before
running the full detection pipeline. `db_connector.log_media_ingestion()`
unconditionally inserts a fresh `MediaLog` row on every call, and
`main_pipeline.process_media()` never checks for a prior match before
computing a fresh Psynth score, filing a fresh `Violation` row, and
generating a fresh complaint PDF via `complaint_generator.generate_complaint_pdf()`.

This was confirmed against real data, not just read off the code: the live
`violations` table currently holds **two independent, correctly-computed
rows for the identical MTN-ad sample** (`b1ce1815-...`), `detected_at` only
29ms apart — `1b8d4940-9656-43a3-99d9-1cde4f38d78a` and
`b097703c-89fb-4328-9729-f505afcf4ff1`, each with its own genuinely
distinct complaint PDF on disk (`reports/complaint_1b8d4940-...pdf`,
`reports/complaint_b097703c-...pdf`), both `status=PENDING_REVIEW`. The
mechanism is a race, not a logic error in either individual run: Flask
3.0.3's `Flask.run()` sets `options.setdefault("threaded", True)`
(confirmed by reading `flask/app.py` — the Werkzeug dev server has defaulted
to threaded since Flask 1.0, independent of `config.FLASK_DEBUG`), so
`dashboard/app.py`'s own `app.run(...)` entry point serves requests on
concurrent threads by default. Two overlapping `POST /upload` requests for
the same content — a double-click, or two open browser tabs — are handled
concurrently, each running its own independent ~5-30s pipeline and each
seeing no prior record of the other, because nothing checks. Reconstructed
from `results/dashboard_pipeline_latency.csv`'s `elapsed_s` values: the two
runs started ~4.08s apart but finished within milliseconds of each other —
only possible if they genuinely overlapped, not if the server had handled
them one at a time.

**Why this is worse here than an ordinary duplicate-row bug:** NFR-04's
append-only constraint (`violations` blocks `UPDATE`/`DELETE` by design, to
preserve legal chain-of-custody) means there is no legitimate way to merge
or remove either row after the fact — the same design property that makes
the audit trail trustworthy makes this specific failure mode permanent. A
deploying agency would end up filing two separate statutory complaints
(to NDPC/ARCON/NCC, per `config.STATUTORY_MAP`) against the same publisher
for the same content, from one double-click.

**Evidence:** `docs/PROJECT_CONTEXT.md` §3, bug #13; `results/dashboard_pipeline_latency.csv`
(two `manifest sample b1ce1815-...` rows, 2026-08-31, `elapsed_s` 21.712 and
17.631); the live `violations`/`reports` tables (queried directly,
2026-09-01); `flask/app.py`'s `Flask.run()` (`options.setdefault("threaded", True)`).

**Action:** §1.7 (Limitations) should state this explicitly as a known
architectural gap, separate from items 13/14 above (which are about model
generalisation, not request handling) — see item 32's companion §5.3
recommendation below, and Chapter 4 should cite the two real duplicate rows
as the demonstrated evidence, in the same spirit as item 15's engineering
bug findings (this is effectively bug #13 in that same log, and arguably the
most consequential one, since #6/#9/#10/#12 were all fixed and this one was
deliberately left unfixed and undeleted so the real row count stays
reproducible for an examiner).

---

## Chapter 2 — Literature Review

### 5. ⚠️ CORRECTION — Social Construction of Technology (SCOT) misattributed

**Document currently states:** SCOT theory attributed to Levy and Munalim.

**Actually the case:** SCOT was developed by Trevor Pinch and Wiebe Bijker
(1984, "The Social Construction of Facts and Artefacts"). This is a
citation error, not a data/code issue — nothing in this repo produced or
depends on it, but it sits in the theoretical-framework chapter a panel is
likely to probe directly, and a misattributed foundational theory reads
badly regardless of everything else in the document being solid.

**Action:** Correct the citation to Pinch & Bijker (1984) wherever SCOT is
introduced (likely §2.2–§2.3, alongside the Value Sensitive Design framing
referenced in §2.2.3 — see item 9 below for where SCOT's actual thesis, that
technology's meaning is negotiated by its social context, connects directly
to this project's own register-bias finding).

### 9. ✚ ADDITION (partly) — register-bias finding is direct empirical support for the project's own thesis

**What actually happened:** DistilGPT-2 perplexity ran backwards on the Day
3 heuristic evaluation — synthetic (LLM-written, deliberately Nigerian-
register) text scored *higher* perplexity (mean 60.9) than authentic
Nigerian journalism (mean 30.3), the opposite of the assumption the
heuristic is built on. This was tested formally later
(`generate_chapter4_evaluation.py`'s register-bias figure): mean
DistilGPT-2 perplexity by genre shows a **9.39x gap** between Pidgin-register
synthetic samples (mean 218.2, n=4) and bureaucratic-register synthetic
samples (mean 23.2, n=4) — a within-synthetic-class comparison, so the
entire spread is attributable to how well DistilGPT-2's Western training
distribution recognises the register, not to authenticity (every sample
compared has the same true label).

**Evidence:** `results/nlp_heuristic_eval.csv` (Day 3 raw numbers),
`results/register_bias_perplexity_by_genre.png` (formal figure),
`docs/PROJECT_CONTEXT.md` §4, Day 3 findings + "Days 23-25" register-bias
entry.

**Action:** This is **empirical support for §1.1/§2.3.1's own claim that
Western AI tooling transfers poorly to Nigerian content** — it should be
written up as a result in Chapter 2's framing and Chapter 4's findings, not
buried as a Day-3 heuristic failure. It is currently the project's strongest
original NLP finding and has, until now, existed only as numbers in a CSV.

---

## Chapter 3 — Methodology / System Design

### 2. ⚠️ CORRECTION — §3.15 hardware: RTX 3080 claim

**Document claims:** Development/inference hardware includes an RTX 3080.

**Actually the case:** No local GPU exists or is assumed anywhere in this
project. Inference is CPU-only throughout; model *training* (NLP
classifier, attempted CV classifier) runs on Google Colab's free T4, not a
local 3080. `CLAUDE.md`'s hard constraints state this explicitly and forbid
any code assuming local CUDA.

**Evidence:** `CLAUDE.md` (Hard constraints section), `README.md`'s Compute
Strategy table, every CPU-only inference path in `watchdog/`.

**Action:** §3.15 needs the RTX 3080 claim replaced with the real
compute story (CPU-only local inference; Colab T4 for the two training
runs), and §1.7 already correctly states this as a limitation — §3.15
should not contradict it.

### 3. ⚠️ CORRECTION — Table 3.4's 10,000-sample target vs. the real 98

**Document claims:** Table 3.4 targets 3,000 (`TELECOM_FMCG_ADS`) / 1,000
(`POLITICAL_MEDIA`) / 6,000 (`TEXTUAL_CORPUS`) samples, 10,000 total.

**Actually happened:** `data/manifest.csv` holds **98 labelled samples**
(53 authentic / 45 synthetic) as of 2026-08-17: 65 in `TEXTUAL_CORPUS` (34
authentic / 31 synthetic), 33 in `TELECOM_FMCG_ADS`. **`POLITICAL_MEDIA` has
zero samples** — an entire category of Table 3.4 is unrepresented, not just
under target. This was expected from the start of the build
(`docs/PROJECT_CONTEXT.md` §1: "not achievable by one student doing
verified manual curation") and is the correct, defensible outcome of the
traceability-over-scale scope argument (§1.6) — but Table 3.4 as currently
written states the original targets as if they were met or still pending,
not as aspirational figures the project deliberately did not chase.

**Evidence:** `data/manifest.csv` (98 rows + header), `docs/PROJECT_CONTEXT.md`
§1 and §4 (dataset state).

**Action:** Table 3.4 needs the real N, broken down by category and label,
with the `POLITICAL_MEDIA` gap and the traceability rationale stated
explicitly — this is a "measure first, don't chase a stale target" finding
consistent with everything else this project does, and should read that
way rather than as an incomplete table.

### 4. ⚠️ CORRECTION — Copyright Act citation, Section 108 → Section 36

**Document claims (§3.12 B, §3.13):** The IP-scraping/plagiarism violation
is based on **Copyright Act 2022 Section 108**.

**Actually the case:** Section 108 is the Interpretation section (pure
definitions) — it has no cause of action and cannot itself be infringed.
The correct provision is **Section 36**, which enumerates infringing acts;
**s.36(2)** specifically covers reproducing "the whole or a substantial
part of the work... in any form recognisably derived from the original" —
exactly what the cosine-similarity threshold is a proxy for. **Section
37(2)** covers the civil remedies (damages, injunction, account of
profits) a complaint would actually seek. Section 108 remains correctly
citable, narrowly, for its definition of "copy" as including electronic
reproduction — just not as the infringement provision.

**Already fixed in code, 2026-08-04:** `config.STATUTORY_MAP["IP_SCRAPING_PLAGIARISM"]`
now cites s.36(2)/s.37(2) — see `config.py:372-390`. Every complaint PDF
generated by this system already cites the correct sections. **The document
is now the only place this error survives.**

**Evidence:** `config.py:372-390`, `docs/PROJECT_CONTEXT.md` §2, "FLAG FOR
CHAPTER 3 REWRITE" entry, `README.md`'s Legal Basis table.

**Action:** Correct §3.12 B and §3.13's citation from s.108 to s.36(2) /
s.37(2). This is the most concrete instance of code and document having
already diverged — the fix exists and is running in production; only the
prose is stale.

### 6. ⚠️ CORRECTION — §3.12 A CV engine: MediaPipe API / library description

**Document likely describes (per the original design):** `mediapipe.solutions.face_mesh`
(the legacy Solutions API).

**Actually the case:** The installed `mediapipe==1.0.0` removed the
`solutions` API entirely. `cv_engine.py`'s dense-landmark stage is wired
onto `mediapipe.tasks.python.vision.FaceLandmarker` (the Tasks API), which
requires an explicit downloaded model bundle
(`models/face_landmarker.task`, `config.FACE_LANDMARKER_MODEL_PATH`)
rather than one bundled in the package. Additionally, MTCNN's face
detection now uses `keep_all=True` plus IoU-based track continuity
(`config.MTCNN_TRACK_IOU_THRESHOLD = 0.3`, `detect_face_bbox()`) rather
than naive highest-confidence-box selection, added specifically to fix a
multi-face-scene misattribution bug found during calibration (see item 15,
bug pattern discussion).

**Evidence:** `docs/PROJECT_CONTEXT.md` §3, bug #10 ("`analyze_clip()`
raised `ModuleNotFoundError`..."); §2, the 2026-08-04 emotional-rigidity
follow-up entry (IoU continuity fix); `watchdog/cv_engine.py`.

**Action:** §3.12 A's description of the face-tracking pipeline (MTCNN +
Face Mesh) needs updating to name the actual API (`FaceLandmarker`/Tasks
API, not `solutions.face_mesh`) and the `keep_all=True` + IoU-continuity
design, both of which are real architectural decisions with a documented
reason, not implementation trivia to gloss over.

### 7. ⚠️ CORRECTION — NFR figures in §3.4's table are stale/unmeasured where the document currently asserts them

**Document claims:** (varies by what was written before real data existed —
likely NFR-01 F1 ≥0.90 stated as achieved, no NFR-02/03 real figures, no
explicit "not computable" framing for a fused score.)

**Actually measured (all real, held-out or end-to-end timed):**

| NFR | Target | Measured |
|---|---|---|
| NFR-01 F1 — text (NLP classifier) | ≥0.90 | **0.867** — target not met |
| NFR-01 F1 — video (CV classifier) | ≥0.90 (gate) | **Gate not cleared** — no number produced (item 8) |
| NFR-01 F1 — fused | ≥0.90 | **Not computable** — no sample in the corpus has ever had both modalities (item 7a below) |
| NFR-02 text latency | ≤5s | **Holds warm: 1.56–2.63s**; cold 32.6–36.5s (one-time model load, disclosed as such) |
| NFR-02 video latency | ≤30s | **Holds: cold 32.06s, warm 4.89s** (full pipeline); Stage-2-only sub-figure 20.0–128.4s range, slowest 23.8s |
| NFR-03 FPR — heuristic vote | ≤5% | **38.2%** — target not met |
| NFR-03 FPR — fine-tuned classifier | ≤5% | **8.8%** — target not met, but 4.3x improvement over the heuristic vote |
| NFR-04 immutable audit log | append-only Postgres | ✓ met — re-verified 2026-09-01 directly against the live PostgreSQL 14.18 instance holding the real data (item 31), not only SQLite test fixtures |
| NFR-05 data minimisation | no PII below threshold | ✓ met |

**Evidence:** `results/nfr_verification_table.csv` (generated by
`generate_chapter4_evaluation.py`, cross-checked against raw per-sample
data), `README.md`'s NFR table (already corrected there), `results/nlp_classifier_fold_metrics.csv`,
`results/dashboard_pipeline_latency.csv`, `results/nlp_heuristic_eval.csv`.

**Action:** §3.4's NFR table must match the numbers above exactly —
`README.md` already carries the corrected version; the write-up should be
brought in line with it, not the other way around. Report both the
heuristic-vote and classifier FPR figures for NFR-03 (not just the better
one) — the gap between them (38.2% → 8.8%) is itself evidence that
localisation (fine-tuning on real Nigerian-labelled data) is the right
fix, which is worth stating as a finding, not hidden by only quoting 8.8%.

#### 7a. ✚ ADDITION — why there is no fused system-level F1, and why that's a finding, not a gap

**What actually happened:** `data/manifest.csv`'s `media_type` is video
*or* text per row, never both, for every one of the 98 samples in this
corpus. `fusion.compute_psynth()`'s two-modality branch
(`alpha*p_visual + (1-alpha)*p_text`) has therefore **never executed
against real data** — there is no dual-modality item to average over, a
stronger and more specific reason than "the CV pathway is unreliable so we
can't average it in."

**Evidence:** `docs/PROJECT_CONTEXT.md` §4, "NFR-01 framing — no fused
system-level F1" entry; `generate_chapter4_evaluation.py`'s
`build_nfr_table()` docstring.

**Action:** State this explicitly in §3.4/§3.11 rather than leaving a
fused NFR-01 figure implied. This is a real methodological finding about
the corpus's structure, not an evasion.

### 10. ✚ ADDITION — fusion's abstention contract (§3.11) is undocumented design, not an implementation detail

Four decisions made building `fusion.py`/`rules_engine.py` each reverse a
plausible-but-wrong default, and none are currently described in §3.11:

1. **Abstention is excluded from arithmetic, never substituted with 0.**
   `fusion.combine_cv_heuristics()` drops an abstained heuristic from the
   average entirely — averaging 3 abstentions + one real 0.9 score
   produces 0.9, not 0.225. This matters because per-clip coverage
   diagnostics found blink-rate's abstention rate correlates with genre/
   edit-pace, not with the true label — treating an abstention as 0 would
   silently inject a genre confound into the score.
2. **A single-modality Psynth is that modality's own score, unweighted —
   never `alpha * missing_modality`.** Weighting only applies when both
   `p_visual` and `p_text` exist; otherwise the missing modality would be
   silently converted into active counter-evidence (a false "visual
   evidence of 0.0" claim).
3. **One Psynth gate for all three violation types, including
   `IP_SCRAPING_PLAGIARISM`** — no separate, lower bar for IP. Consequence:
   a human plagiarising another human (low `p_text`) is correctly out of
   scope, because this system detects **AI**-plagiarised IP specifically.
4. **Text with no IP match has no modelled cause of action, even at
   Psynth near 1.0.** `determine_violation_type()` returns `None` — writing
   text with an LLM isn't itself illegal in this system's scope; only
   reproducing a specific protected work is. Confirmed against real data:
   fires on all 31 real synthetic `TEXTUAL_CORPUS` samples (item 12/Chapter
   4 below).

**Evidence:** `docs/PROJECT_CONTEXT.md` §2, "Objective 3 build" entry
(2026-08-06); `tests/test_fusion.py`'s
`test_single_scored_heuristic_is_not_pulled_toward_zero_by_abstentions`.

**Action:** §3.11 (Mathematical Framework) should state all four rules
explicitly with their rationale — currently only the weighted-average
formula itself is likely documented, not the abstention/missing-modality
contract that governs when and how it actually applies.

### 11. ✚ ADDITION — IP similarity threshold (0.82) was validated against real data, not left as a guess

**What actually happened:** `IP_SIMILARITY_THRESHOLD = 0.82` was a
pre-data guess, like the Day 3 NLP thresholds. It was tested with two
direct controls against the 34-work protected corpus:

- **Negative control (leave-one-out):** raw result 5/34 unrelated pairs
  exceeded 0.82 (max 0.939), but hand-inspection found two distinct,
  legitimate structural causes — same-event convergent journalism (both
  outlets quoting an official statement) and same-publisher beat
  clustering — not the matcher confusing unrelated articles. Excluding
  those, the genuine ceiling among the remaining 29 is **0.7615**.
- **Positive control (substantial paraphrase):** four hand-paraphrased
  works (same facts, heavily reworded) all correctly attributed back to
  source, all scoring above 0.82 (floor 0.8676).
- **Verdict:** 0.82 sits inside a clean **[0.7615, 0.8676]** margin. Kept
  as measured, not retuned.

**Evidence:** `docs/PROJECT_CONTEXT.md` §2, "IP_SIMILARITY_THRESHOLD
validation" entry; `validate_ip_similarity_threshold.py`;
`results/ip_similarity_*_control.csv`; `config.py:133` comment.

**Action:** §3.12 B should cite this validation instead of presenting
0.82 as an assumed constant — it's a genuine methodological result
(including the honest disclosure that a raw threshold can't distinguish
"quoting the same press statement" from actual copying — a stated §1.7
limitation, not solved by picking a different number).

### 16. ✚ ADDITION — Flask admin dashboard (Days 21-22) is built and should be described in §3.13

**What actually happened:** `dashboard/app.py` implements the human-review
step §3.13 already requires (`PENDING_REVIEW → DISPATCHED`) with four
routes (violations list, detail view, PDF serve, `/upload` submission).
Notable design choices: the detail view imports the complaint PDF's
caveat text directly rather than re-writing it, so the PDF and the web
page structurally cannot drift apart; abstentions render as an explicit
amber "ABSTAINED" pill with the stored reason string, never a blank or a
0.00. 21 new tests (`tests/test_dashboard.py`), verified against the real
Postgres DB.

**Evidence:** `docs/PROJECT_CONTEXT.md` §4, "Days 21-22 — Flask admin
dashboard build" entry.

**Action:** §3.13 (Routing) should describe the actual four routes and
the two rendering guarantees above, not just the abstract requirement.

---

## Chapter 4 — Results and Discussion

### 8. ✚ ADDITION — CV classifier: a documented negative result, not missing work

**What actually happened:** Three training schedules for the
EfficientNet-B0 CV classifier were run against an 80-identity-group
FaceForensics++ subset, against a 0.90 held-out-accuracy credibility gate
agreed **before** any run:

| Run | Schedule | Held-out accuracy | Diagnosis |
|---|---|---|---|
| 1 | Single-phase, 6 epochs | 0.86 | Overfitting (train 0.978 vs test 0.86) |
| 2 | Two-phase, stronger aug | 0.78 | Overcorrected into underfitting |
| 3 | Two-phase, 20 fine-tune epochs | 0.79 | Converged correctly, caught by best-checkpoint selection |

The key finding is about the **split, not the schedule**: run 3's internal
validation accuracy (0.881) vs its held-out test accuracy (0.79) — a
~9-point gap between two disjoint splits of the *same* ~80-identity-group
pool, on the *same* trained weights — shows runs 1's 0.86 and 3's 0.79
plausibly sit inside the same split-luck noise band. The binding
constraint is identity-group count, not hyperparameters. Per an
agreement made in advance, the search was capped at three runs rather than
continued until one landed above 0.90 by the same noise that separates
0.86 from 0.79 (which would be indistinguishable from p-hacking the split).
**No CV transfer number is reported; no checkpoint exists.**

**Evidence:** `docs/PROJECT_CONTEXT.md` §4, "CV classifier training —
closed as a documented negative result" entry (2026-08-11/12);
`README.md`'s NFR-01 video row.

**Action:** This belongs in Chapter 4 as a full result, not a gap to
apologise for — a negative result with a clearly diagnosed cause
(dataset-scale-bound, not a fixable training defect) and a pre-registered
stopping rule is a legitimate, citable engineering finding. §1.7 and §5.5
both need the same framing (see item 14).

### 8a. ✚ ADDITION — the four hand-built CV heuristics also failed to discriminate, and that's why the classifier was tried

Before the classifier, the four Stage 4 heuristics (blink-rate, emotional
rigidity, viseme-phoneme mismatch, blending artifacts) were evaluated
across all 32 CV-eligible clips (primary comparison: `matched_subset`,
n=13, 4 authentic/9 synthetic):

- Blink rate: 100% abstention in **both** classes — no usable data at all.
- Emotional rigidity: authentic mean/std exactly 0.0/0.0 (n=3), synthetic
  mean 0.133 (n=7) — driven by outliers, not a clean shift.
- Viseme-phoneme mismatch: ran **backwards** — authentic mean 0.183 (n=3)
  vs synthetic 0.039 (n=7).
- Blending artifacts: heavy overlap (authentic 0.647 vs synthetic 0.524),
  also backwards.

**Evidence:** `results/cv_heuristic_eval*.csv`, `docs/PROJECT_CONTEXT.md`
§4, "Full 32-clip CV heuristic evaluation" entry.

**Action:** This is the direct, stated trigger for the Day 8-9 classifier
pivot and belongs in Chapter 4 as the reasoning chain, not just the
classifier result in isolation.

### 12. ✚ ADDITION — length-confound test on the NLP classifier (checked, not assumed)

**What actually happened:** The classifier's 0.088 FPR result carries an
obvious length-confound risk (authentic mean 865 words vs synthetic mean
303). This was tested directly, not left as a caveat:

- Per-length-band accuracy: 0.765–1.000 across four bands — did not
  collapse in the bands where both classes actually mix.
- Length-matched subset (200-850 words, n=40): accuracy 0.850, FPR 0.105 —
  within ~2 points of the full-corpus aggregate in both directions.
- Length-only baseline ceiling (best possible word-count-alone rule,
  optimistic in-sample search): 0.846 accuracy — the real classifier's
  honest out-of-fold accuracy (0.877) beats even this biased-in-its-favour
  ceiling.

**Verdict: length is not doing significant work.** The 0.088 FPR stands as
measured.

**Evidence:** `results/nlp_length_*.csv`, `analyze_length_confound.py`,
`docs/PROJECT_CONTEXT.md` §4, "Length-confound test" entry.

**Action:** Chapter 4 should present this as a check that was run and
passed, not omit it — a panel member's obvious first question about a
length-imbalanced corpus currently has no answer in the document even
though the project already answered it.

### 15. ✚ ADDITION — engineering bug findings worth citing as results, not just fixed silently

`docs/PROJECT_CONTEXT.md` §3's bug log flags several of its own entries as
worth citing in Chapter 4 as genuine findings, not just bugfixes:

- **Bug #6** — six Instagram video downloads silently overwrote each other
  (generic shared titles collided in the output filename template) with
  **no error and a "success" exit** — a lesson about metadata-dependent
  uniqueness guarantees.
- **Bug #9** — stdlib `html.parser`'s missing HTML5 implicit-`<p>`-close
  support caused cascading paragraph duplication on two Nigerian news
  sites (BusinessDay inflated 721→11,300 words, Punch 462→1,577), silently,
  while `pytest tests/` stayed green throughout. Root cause and fix
  (switch to lxml) directly produced item 3's corrected dataset numbers.
- **Bug #10** — `cv_engine.py`'s `analyze_clip()` was written against a
  MediaPipe API (`solutions.face_mesh`) that no longer exists in the
  installed version — imported cleanly, read correctly, passed code
  review, and failed only when actually run against real video (see item
  6).
- **Bug #12** — two database test fixtures survived undetected in the
  live, legally-relevant `violations` table for over a month because a
  cleanup failure was caught and logged as a non-critical warning instead
  of surfacing — found only by a real row-count mismatch, not by any test.

**Evidence:** `docs/PROJECT_CONTEXT.md` §3 (full bug log + the four
"is the important one" / "is the second/third/fourth instance" framing
paragraphs).

**Action:** Chapter 4 (or a Chapter 3 engineering-methodology subsection)
should cite at least these four as a demonstrated pattern — every one was
caught by inspecting real output against real data or a real database,
never by a clean test run or a clean diff, which is itself a stated,
repeated methodological finding of this project (see `docs/PROJECT_CONTEXT.md`
§6).

---

## Chapter 5 — Summary, Conclusions, Recommendations

### ✚ ADDITION — §5.3 Recommendations: deduplicate on `url_hash` before filing a complaint

Companion to item 32 above. §5.3 (Recommendations, addressed to a deploying
regulatory agency rather than to future student work — that's §5.5's scope)
currently has no recommendation covering request-level idempotency, and item
32's finding is exactly the kind of operational gap that belongs there
rather than only in §1.7: it's not a research limitation to study further,
it's an operational precondition a real deployment must satisfy before this
system's output (a filed statutory complaint) is used for anything.

**Recommended addition to §5.3:** *A production deployment must not file a
complaint for a `source_url` that already has a `violations` row for the
same `url_hash`, and must serialise or lock on that check* — i.e.
deduplication has to happen before the detection pipeline runs, not after,
since by the time a `Violation` row exists the append-only constraint
(NFR-04) that makes the audit trail trustworthy also makes a duplicate
un-mergeable. Concretely, this project's own prototype demonstrates why:
`data/manifest.csv`-driven or live-`url` submissions through
`dashboard/app.py`'s `/upload` route currently have no such check, and two
overlapping submissions of one artifact (Chapter 4/item 32's real,
reproducible example — `detected_at` 29ms apart, two distinct complaint
PDFs, `results/dashboard_pipeline_latency.csv`) produced two separate
complaints for the same content. §5.3 should name at least one concrete
mechanism a deploying agency could use — e.g. a `UNIQUE` constraint on
`media_log.url_hash` with `ON CONFLICT DO NOTHING`/an explicit
already-processed response, or a request-level idempotency key — rather
than leaving "add deduplication" unspecified.

### Restated from Chapter 1/3 for §5.5 Further Studies

- **CV domain-transfer claim (§2.3.4) remains untested**, not disproven —
  the in-domain baseline never reached a reliable enough number to make
  the FF++-vs-Nigerian-corpus comparison meaningful. §5.5 should
  recommend a larger identity-group count (full FF++ release, or the
  deferred Celeb-DF access request) or a lower-capacity/frozen-backbone
  approach, rather than more epochs on the current subset (item 8).
- **Cross-LLM generalisation** (item 13) needs either genuine access to a
  second model family or an explicit, acknowledged scope limitation —
  currently neither is stated.
- **CV heuristic reliability on multi-face scenes** — the emotional-
  rigidity IoU-continuity fix resolved one failure mode
  (`keep_all=False` misattribution) but not a second, structurally
  different one (a hard scene cut landing inside a single ~1s MTCNN
  sampling interval, unresolved, documented in `config.py`'s
  `EMOTIONAL_RIGIDITY_THRESHOLD` comment) — worth naming as concrete
  further-work, not folded into a generic "more data would help."
- **Fused (multimodal) evaluation** — item 7a's finding (no sample has
  ever had both modalities) means a genuine system-level F1 needs a
  corpus redesign (samples with both video and text for the same
  underlying content), not just more of the current one-modality-per-row
  corpus.

---

## Document-level corrections

Everything above was derived from `docs/PROJECT_CONTEXT.md` — the build history.
That source only records what happened during the build; it has no opinion on
errors that live purely in the document's own prose, tables, figures, and
reference list. This section was compiled the other way round: by reading
`FINAL_YEAR_PROJECT_1__8_.pdf` (54 pages, the file with the latest mtime among
the nine drafts in Downloads) directly, page by page, including rendering the
five Chapter 3 figures to images to check what they actually show. Every item
below is quoted or measured from the PDF itself, not inferred.

A third tag is needed here alongside the two above:

- **🖼️ PRESENTATION** — the document is not internally consistent, or a figure
  doesn't render as intended. Nothing here misstates a measured result the way
  a ⚠️ CORRECTION does; an examiner turning to the page sees the defect
  directly regardless of whether they check it against the code.

### Chapter 1 / 2

### 18. ⚠️ CORRECTION — "sufficient effort" is the repealed Act's standard; the Copyright Act 2022 says "some effort" (§1.1 and §2.3.3)

**Document states, twice:**
- §1.1 (p.13): *"Section 2(1) of the Copyright Act 2022 implicitly challenges the copyrightability of entirely AI-generated content by strictly requiring that 'sufficient effort' be expended by a human author..."*
- §2.3.3 (p.27): *"Section 2 of the Nigerian Copyright Act 2022 demands that 'sufficient effort' be expended by a human to make a work original..."*

**Verified against the statute (web search, not assumed):** Copyright Act
2022 §2(2) reads *"some effort has been expended on making the work, to give
it an original character."* "Sufficient effort" was the old (pre-2022, Cap
C28 LFN 2004) standard, deliberately loosened by the 2022 Act specifically
because "sufficient" had been used to deny protection to work judged "too
simple." Both write-up passages currently describe the repealed Act, not the
one actually cited.

**Action:** Replace "sufficient effort" with "some effort" in both §1.1 and
§2.3.3, and note the italicized reasoning quote above if a sentence explaining
*why* the standard changed is wanted (it strengthens, not weakens, the
document's own point about a low bar for human authorship).

**Sources:** [LawGlobal Hub, Section 2 Nigerian Copyright Act 2022](https://www.lawglobalhub.com/section-2-nigerian-copyright-act-2022/), [S.P.A. Ajibade & Co, A Review of the Nigerian Copyright Act 2022](https://spaajibade.com/a-review-of-the-nigerian-copyright-act-2022/)

### 19. ⚠️ CORRECTION — §1.7 still describes hardware and video quality that never existed

**Document states (p.17-18):** *"the reliance on mid-tier graphical
processing units (GPUs) restricted the processing of high-definition 4K
video feeds in absolute real-time, introducing minor latency during
frame-by-frame analysis under heavy data loads."*

**Actually the case:** No GPU of any kind — mid-tier or otherwise — exists or
is used locally anywhere in this project; `CLAUDE.md`'s hard constraints
state CPU-only inference explicitly. No 4K video was ever processed; the
video pathway operates on scraped social/ad clips, not real-time 4K feeds.
This is a second, separate location from item 2's §3.15 RTX 3080 claim
(§3.15 is Experimental Setup; §1.7 is Limitations) — both need the same real
compute story, not just the more prominent one.

**Action:** §1.7's first limitation should state the real constraint (no
local GPU; CPU-only inference; Colab T4 for the two training runs) instead
of describing hardware capability the project never had access to.

### 20. ⚠️ CORRECTION — SCOT misattribution also sits in Table 2.2's "Proponent(s)" column, not only §2.2.1's prose

Existing item 5 (above) documents SCOT's misattribution to "Levy (1998);
Munalim (2021)" in §2.2.1's prose. **Table 2.2** (p.25) repeats the identical
error in its own "Proponent(s)" column: *"Social Construction of Technology
(SCOT) | Levy (1998); Munalim (2021)."* Fixing only the prose and leaving the
table would leave a self-contradiction on the same two pages if only one is
corrected.

**Action:** Item 5's fix needs to explicitly include Table 2.2's row, not
just "§2.2–§2.3" prose — this is a distinct location, not covered by
correcting the sentence above it.

---

### Chapter 3 — overclaims

### 21. ⚠️ CORRECTION — §3.9 states the Tinubu ARCON deepfake was collected; `POLITICAL_MEDIA` holds zero samples

**Document states (§3.9, p.43-44):** *"The collected dataset included known,
verified AI-generated campaigns (such as the MTN AI avatars and Golden Terra
advertisements) as well as political deepfakes (including the unauthorized
Tinubu ARCON video)."*

**Actually the case:** Per existing item 3 (Table 3.4 correction, evidenced
against `data/manifest.csv`), `POLITICAL_MEDIA` has **zero** rows — the
Tinubu clip was never actually acquired and labelled into the dataset. §3.9's
prose asserts it as already collected, which is a second, separate place
(distinct from Table 3.4 itself) making the same false claim in narrative
form — a reader who only reads the prose and skips the table would come away
believing the political-deepfake category exists.

**Action:** §3.9's prose needs the same correction as Table 3.4 — state the
`POLITICAL_MEDIA` gap explicitly here too, not just in the table.

### 22. ⚠️ CORRECTION — Table 3.1 claims "milliseconds per content frame"; measured latency is single-digit seconds

**Document states (Table 3.1, p.36):** Detection Speed for the proposed
system: *"Near real-time inference (milliseconds per content frame)."*

**Actually measured** (`results/nfr_verification_table.csv`, NFR-02, already
the source for existing item 7): text pipeline warm **1.56s–2.63s**; video
pipeline warm **4.886s** (cold: 32–36s, disclosed separately as one-time
model load). Nothing in this system runs in milliseconds — the real figures
are 3-4 orders of magnitude slower than claimed, and NFR-02's own *target*
(≤5s text, ≤30s video) is stated in whole seconds a few pages later in the
same chapter (Table 3.3), directly contradicting Table 3.1's "milliseconds."

**One correction to your framing:** you cited "0.8-4.9s" as the measured
range. That doesn't match any single source cleanly — 4.886s is real (video
warm, `dashboard_pipeline_latency.csv`), but 0.8s traces to a *superseded*
internal profiling note in `docs/PROJECT_CONTEXT.md` (a pre-fix/post-fix
`spaCy.load()` caching comparison, "warm full-pipeline runs dropped to
0.78s-0.96s") that predates the final NFR-02 measurement run and was for text
only. The authoritative, already-validated figure to cite is **1.56s-4.89s
warm** (`nfr_verification_table.csv`), not 0.8s-4.9s.

**Action:** Replace "milliseconds per content frame" with the real warm
latency range, citing NFR-02's table — which already exists and is correct —
rather than introducing a second, slightly different number.

### 23. ⚠️ CORRECTION — §3.3.1 claims the FP bias was "effectively eliminate[d]"; measured FPR is 8.8%

**Document states (§3.3.1, p.35):** *"Localized NLP Pipeline: ... to
effectively eliminate the false-positive bias inherent in Western detection
tools."*

**Actually measured:** Per existing item 7 (NFR-03), the fine-tuned
classifier's FPR on authentic Nigerian English is **8.8%** (3/34) — an
improvement over the 38.2% heuristic-vote baseline, but neither figure meets
NFR-03's own ≤5% target two pages later in the same chapter (Table 3.3), let
alone "effectively eliminate[s]" the bias. "Effectively eliminated" and "the
target wasn't met, but localisation cut it 4.3x" are different claims — the
document should make the second one, which is both true and still a real
finding.

**Action:** Replace "effectively eliminate" with language matching the
measured 8.8%/38.2% figures already corrected in Table 3.3 by item 7 — the
two passages currently contradict each other within the same chapter.

### 24. ✚ ADDITION — Table 3.6's four evaluation targets (§3.14) are all missed; existing item 7 only covers Table 3.3

Existing item 7 corrects Table 3.3's NFR figures. It doesn't touch **Table
3.6** (§3.14, p.49), a separate table with its own four targets — Sensitivity
≥88%, Specificity ≥95%, Precision ≥92%, F1 ≥0.90 — that nothing in the
current corrections list addresses.

**Computed from the same confusion matrix item 1 already cites** (31 TN / 3
FP / 5 FN / 26 TP, n=65, `results/confusion_matrix_nlp_classifier.png`):

| Metric | Target | Measured | Result |
|---|---|---|---|
| Sensitivity (Recall) | ≥ 88% | 26/31 = **83.9%** | MISS |
| Specificity | ≥ 95% | 31/34 = **91.2%** | MISS |
| Precision | ≥ 92% | 26/29 = **89.7%** | MISS |
| F1-Score | ≥ 0.90 | **0.867** | MISS |

All four targets in Table 3.6 are missed, by margins of 2-11 points — none
by a large gap, but none met either. This is the same real 65-sample result
already reported elsewhere in the corrections list (item 1's F1/precision/
recall figures), just never checked against *this* table's specific targets.

**Action:** Table 3.6's target column should either be revised to match what
a first real fine-tuning run on 65 samples can defensibly promise, or the
table should carry the same "target not met, here's the real number" framing
item 7 already applies to Table 3.3 — this is a formatting/completeness gap
in the corrections list, not a new empirical finding.

---

### Presentation

### 25. 🖼️ PRESENTATION — ToC omits Chapters 4 and 5; the document itself currently ends at the References, not just the ToC entry

**Confirmed:** the Table of Contents (p.8-10) lists Chapter One through
Chapter Three, then jumps straight to "REFERENCES" — no Chapter 4 or Chapter
5 entries exist. This is not only a ToC gap: the PDF's actual body does the
same thing — §3.16 (Ethical Considerations) is immediately followed by the
References list starting on p.51; there is no Chapter 4 or Chapter 5 content
anywhere in this 54-page draft. That's consistent with existing item 17
("Chapter 4 dashboard screenshots are still outstanding") and the file's own
"Days 26-28 (Write-up)" framing — Chapters 4-5 haven't been drafted yet, so
the ToC is honestly reflecting the document as it currently stands, not lying
about it.

**Action:** No fix needed yet beyond finishing Chapters 4 and 5 (tracked by
item 17 and the Chapter 4/5 sections above) and then adding their ToC
entries at that point — flagging this mainly so it isn't mistaken for a
simple missed-ToC-line bug when Ch.4/5 do get written.

### 26. 🖼️ PRESENTATION — Figures 3.1-3.5 render as placeholder node labels, not the diagrams the surrounding text describes

Rendered each figure to an image directly from the PDF and inspected it:

- **Figure 3.1** (p.34, Existing System workflow): nodes read **A → B → C →
  D → (diamond: "Is Content Synthetically Generated?") → No→F / Yes→G → H →
  "Enforcement Action."** Only the decision diamond and final box have real
  labels; five of seven process boxes are bare letters.
- **Figure 3.2** (p.37, Proposed System architecture): **A → B → C, D →
  "Multimodal Fusion Module" → "Log as Authentic Content" → G, s3 → H → I.**
  This is worse than a labelling gap: as drawn, the fusion module's output
  flows *unconditionally* into a box reading **"Log as Authentic Content"** —
  exactly backwards from §3.6/§3.7's own prose ("If the combined synthetic
  probability score exceeds 85%, the system transitions..."), which requires
  a branch, not a single always-authentic path. If this figure is redrawn
  with real labels rather than just de-lettered, the branch itself needs
  fixing, not only the box text.
- **Figure 3.3** (p.40, Input structure): **A → "Watchdog Ingestion API" →
  D**, alongside two correctly-labelled boxes ("Video Artifacts," "Publisher
  Metadata"). Two of four nodes are placeholders.
- **Figure 3.4** (p.41, Output structure): **"Watchdog Processing Engine" →
  B, C, D, E** — all four outputs are bare letters; none show the four real
  output types (Binary Classification, Confidence Score, Statutory Mapping
  Tag, Auto-Complaint Document) listed in the paragraph immediately above the
  figure.
- **Figure 3.5** (p.42, "Operational Flowchart"): this one isn't a rendering
  bug — the caption promises a flowchart, but the content is a plain
  9-item numbered text list, not a diagram at all.

**Likely cause:** these all look like a diagramming tool's default/example
node labels (A, B, C, D... plus a stray autosaved id, "s3") shipped in place
of the edited version — a re-export from source rather than a PDF rendering
fault, since the text and table boxes on the same pages render perfectly.

**Action:** Figures 3.1-3.4 need to be regenerated with their actual labels
(the surrounding prose already states what each node should say). Figure
3.2's fusion→"Log as Authentic Content" edge is a logic error in the diagram
itself, not just a caption issue, and should be redrawn as a branch. Figure
3.5 needs an actual flowchart image, or its caption/expectation reworded to
match what it currently is.

### 27. 🖼️ PRESENTATION — reference inconsistencies: Ampifire/Amplifier; NITDA 2024/2026

- **Ampifire vs. Amplifier:** the reference list (`Ampifire. (2023). Is AI
  detection rigged?`) and the in-text citation in §2.3.4 (p.28, `(Ampifire,
  2023)`) agree — but **Table 2.3** (p.28-29) cites the same source as
  `Amplifier (2023)`. One source, two spellings, in the same chapter.
- **NITDA 2024 vs. 2026:** §1.1 (p.12) cites `(NITDA, 2024)`; §2.3.2 (p.27)
  cites `(NITDA, 2026)`; the reference list itself (`National Information
  Technology Development Agency (NITDA). (2026). Tech Regulator Issues
  Warning...`) dates it 2026. §1.1 is the outlier — two chapters and the
  reference list agree on 2026, only §1.1 says 2024.

**Action:** Standardize both to match the reference-list entries — Ampifire
(not Amplifier) in Table 2.3; NITDA, 2026 in §1.1.

### 28. ⚠️ CORRECTION — Fadolapo is ARCON's Director-General, quoted in a press statement — not the author of a 2023 work

**Document cites (reference list item 13, and in-text at §1.1/§2.3.2):**
`Fadolapo, O. (2023). ARCON threatens sanctions over 'unauthorised' AI advert
featuring Tinubu. The Cable.` — formatted and cited as if Fadolapo authored
this piece.

**Verified (web search):** Dr Olalekan Fadolapo is ARCON's Director-General.
The Tinubu-deepfake statement was issued **22 September 2025**, signed by
Fadolapo *as ARCON's DG* and reported by outlets including The Cable — he is
the official quoted in the story, not its author. Both the year (2023 → 2025)
and the citation form (treating a quoted official as the author of a news
piece) are wrong.

**Action:** Recite as an organisational/news source — e.g. `ARCON (2025)` or
`The Cable (2025), quoting ARCON Director-General Olalekan Fadolapo` — dated
September 2025, both in the reference list and at both in-text locations
(§1.1, §2.3.2).

**Sources:** [Brand Communicator, ARCON warns public, 23 Sept 2025](https://brandcom.ng/2025/09/23/arcon-warns-public-against-fake-ai-generated-advertisement-featuring-president-bola-tinubu/), [BrandCrunch, ARCON warns public, 24 Sept 2025](https://brandcrunch.com.ng/2025/09/24/arcon-warns-public-over-fake-ai-generated-ad-of-president-tinubu/)

### 29. 🖼️ PRESENTATION — exactly 10 of the 34 references are never cited in the body text

Checked every reference-list surname against the full extracted body text.
**Exactly ten** references appear only once in the whole document — their
own reference-list entry, with no in-text citation anywhere:

Adeleke (2021); Durodola (2025); Dwivedi et al. (2023); Ebosetale (2025);
Eke et al. (2023); Nasir (2025); Nwodu, Obiora & Agbachukwu (2025);
Sabaliauskaite, Paskauskas & Bružė (2026); Udo (2022); Zhang & Kreps (2020).

**Action:** Either cite each where it's actually relevant (several — Dwivedi
et al. on generative AI in creative industries; Zhang & Kreps on deepfake
detection — look like they belong somewhere in §2.3 or §2.4 and were simply
dropped when a paragraph was cut) or remove them from the reference list. An
examiner cross-checking references against text will notice a list this much
longer than what's actually cited.

### 30. ⚠️ CORRECTION — the system's name drifts three ways across title, Abstract, and §2.1

Three different names are in use, not two:

1. **Title page, Certification page, Fig. 2.1 caption, Fig. 3.5 caption:**
   *"[Automated] AI Ethics Watchdog"* — no "Naija."
2. **Abstract, §1.3, §2.1 body prose, §3.1:** *"Naija AI Watchdog"* — no
   "Ethics."
3. **The actual project name** (`CLAUDE.md`'s own header, the codebase this
   write-up is describing): *"Naija AI Ethics Watchdog"* — has both words.

The drift isn't confined to comparing chapters — it happens **within §2.1
itself**: the body text says *"the proposed Naija AI Watchdog"* while its
own **Figure 2.1**, two paragraphs later on the next page, is captioned
*"Conceptual Framework of **the AI Ethics Watchdog**."*

**Action:** Pick one name — `CLAUDE.md`'s "Naija AI Ethics Watchdog" is the
actual project name and the least-effort fix — and use it everywhere: title
page, Certification page, Abstract, every chapter, and every figure caption.

### 31. ⚠️ CORRECTION — Table 3.8 states "PostgreSQL 15 via Docker Compose"; the real server is native PostgreSQL 14.18

**Document states (Table 3.8):** The production database is PostgreSQL 15,
provisioned via Docker Compose.

**Actually the case (found and verified 2026-09-01):** A native Ubuntu
install of **PostgreSQL 14.18** holds port 5432 on the development machine.
`docker compose up -d` starts the `postgres:15-alpine` container declared in
`docker-compose.yml` and it reports "healthy" — but its `5432:5432` port
mapping never actually binds, because the native server already holds that
port at container start. `docker inspect`'s `NetworkSettings.Ports` confirms
this directly (`"5432/tcp":[]` — empty, no host-side mapping at all), and
connecting through `config.DATABASE_URL` and running `SELECT version()`
returns PostgreSQL 14.18, not 15. This is not a new problem introduced today —
it has been true since Day 1; nothing ever failed because both servers work
independently and the app was only ever pointed at one of them
(`localhost:5432`, which has always resolved to the native install).

**Why this matters beyond the version number:** every real row in
`violations` — the append-only legal audit table Table 4.9 and Chapter 4 both
evaluate — lives in the native Postgres 14 instance, not the Docker
container. The Docker container has been running this whole time with an
empty, unused database.

**Same pattern as item 4** (Copyright Act §108→§36 citation) and items 2, 6,
19 (RTX 3080, MediaPipe legacy API, "mid-tier GPU" claims): the document
describes the intended/designed setup, not what the running system actually
does, and nothing failed loudly enough to force a correction on its own.
Logged in `docs/PROJECT_CONTEXT.md` as the same recurring pattern.

**Action taken in code (2026-09-01), not just flagged in the document:**
`watchdog/db_connector.get_engine()` now logs a `WARNING` if the Postgres
server it actually connects to doesn't match the version
`docker-compose.yml` declares (`config.DOCKER_COMPOSE_POSTGRES_VERSION = 15`)
— so a future divergence (e.g. a reboot that happens to start Docker before
the native service, silently routing the live pipeline at an empty database)
surfaces immediately instead of requiring another accidental discovery.
`docker-compose.yml`'s own header comment now states the discrepancy
explicitly. The compose file itself was kept, not removed — it's a genuine,
working setup path for anyone provisioning this project on a machine that
doesn't already have something bound to 5432.

**Evidence:** `docker inspect naija_watchdog_db`, `SELECT version()` run
through `config.DATABASE_URL`, `docs/PROJECT_CONTEXT.md`'s "PostgreSQL 14
native vs Docker 15" entry, `watchdog/db_connector.py`'s version-drift check,
`docker-compose.yml`'s header comment.

**Action:** Table 3.8 should read "PostgreSQL 14.18 (native)," not
"PostgreSQL 15 via Docker Compose."

---

## Outstanding — not a write-up correction, but blocks Chapter 4 being complete

**17. Chapter 4 dashboard screenshots are still outstanding.** Every other
Days 23-25 artifact (confusion matrix, register-bias figure, NFR
verification table) is done as of 2026-08-13; the dashboard UI itself has
not yet been screenshotted against a live running session. This needs a
real browser session against the running Flask app before Chapter 4 can
be finished, separately from any of the corrections above.

**Evidence:** `docs/PROJECT_CONTEXT.md` §5, Days 23-25 roadmap entry
("Chapter 4 screenshots still outstanding").
