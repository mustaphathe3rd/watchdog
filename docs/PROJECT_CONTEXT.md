# Project Context

Everything not inferable from reading the code. Written as a handoff after Days 1–3
of a 28-day build. Read `CLAUDE.md` first for the operational rules; this file is
the *why*.

---

## 1. Academic framing

The deliverable is not just working software — it is a project that must be
defended. That changes the priorities in ways that are easy to miss:

- **Traceability beats scale.** Every dataset sample must be attributable to a
  specific, justifiable source. This is why `seed_sources.py` is a hand-curated
  list and not a crawler (see §1.6 scope argument in the write-up). An open crawler
  would collect faster and be indefensible in a viva.
- **Honest limitations are worth more than inflated numbers.** Where reality
  diverges from the write-up's claims (GPU, dataset size), the correct move is to
  document the divergence, not to hide it. §1.7 already establishes hardware
  constraints as a stated limitation.
- **The write-up's targets are aspirational in places.** Table 3.4 specifies
  3,000 / 1,000 / 6,000 samples across the three categories. That is not
  achievable by one student doing verified manual curation. Expect Chapter 4 to
  report the real, much smaller N with a justification. Do not treat those
  numbers as requirements to hit.

---

## 2. Design decisions and rationale

These were deliberate. Don't reverse them without understanding the reason.

**DistilGPT-2 for perplexity, DistilBERT reserved for classification.**
`config.NLP_BASE_MODEL` says `distilbert-base-uncased`, matching §3.12 B. But
DistilBERT is a masked/bidirectional model and has no well-defined autoregressive
perplexity — perplexity is `P(token_i | token_1..i-1)`, which only a causal model
computes. `nlp_engine.py` therefore loads DistilGPT-2 purely as a perplexity
feature extractor. DistilBERT remains for the fine-tuned classifier (Days 8–9).
This is a hybrid statistical-features + transformer design, and it should be
described that way in Chapter 3 rather than presented as a deviation.

**Burstiness uses Goh & Barabási (2008):** `B = (σ − μ) / (σ + μ)` over sentence
word-counts, bounded in [−1, 1]. A named, citable metric was chosen over an ad-hoc
variance measure specifically so the methodology section has a reference.

**Conservative voting.** `analyze_text()` returns `AI` only when *both* perplexity
and burstiness agree; disagreement returns `UNCERTAIN`, never a confident call.
This follows from NFR-01/NFR-03 (false-positive minimisation) and the Value
Sensitive Design framing in §2.2.3 — the system must not wrongly accuse human
creators. Do not "improve accuracy" by making this more aggressive.

**`PSYNTH_THRESHOLD = 0.85`** comes directly from §3.5.2 of the write-up. It is
not a tunable hyperparameter; changing it contradicts the document.

**Data minimisation is structural, not procedural (NFR-05, §3.16).** `media_log`
stores a SHA-256 hash of the source URL, never the URL itself. The raw URL is
written only to `violations`, and only when the 0.85 threshold is crossed. The
schema makes the privacy property impossible to violate by accident — there is no
column to leak.

**UUID primary keys, not auto-increment integers.** Prevents an external party
enumerating the compliance log via sequential IDs (`/reports/1`, `/reports/2`).

**Append-only `violations` table (NFR-04).** Enforced in code by a SQLAlchemy
`before_execute` listener, not by convention.

**FLAG FOR CHAPTER 3 REWRITE — wrong Copyright Act citation (found 2026-08-04).**
§3.12 B and §3.13 in the write-up cite **Copyright Act 2022 Section 108** as the
basis for the IP-scraping/plagiarism violation. This is wrong and needs correcting
in the document itself, not just in code: **Section 108 is the Interpretation
section — pure definitions, no cause of action, cannot itself be infringed.** The
correct provision is **Section 36**, which enumerates the acts that constitute
infringement; **s.36(2)** specifically covers reproducing "the whole or a
substantial part of the work either in its original form or in any form
recognisably derived from the original" — exactly what a cosine-similarity
threshold over embeddings is a proxy for. **Section 37(2)** covers the civil
remedies (damages, injunction, account of profits) a complaint would actually be
seeking. Section 108 remains correctly citable, narrowly, for its definition of
"copy" as including a reproduction in electronic/digital form — just not as the
infringement provision itself. `config.STATUTORY_MAP["IP_SCRAPING_PLAGIARISM"]`
was corrected to s.36(2)/s.37(2) on 2026-08-04 (see `watchdog/ip_similarity.py`'s
build entry below for the full IP-similarity-matrix work this correction landed
alongside); the write-up itself still needs the same correction in Chapter 3.

**`watchdog/ip_similarity.py` build (2026-08-04, Day 7, §3.12 B) — the last
unbuilt piece of Objective 2.** Embeds a "protected corpus" once via
sentence-transformers (`all-MiniLM-L6-v2`) and scores arbitrary input text
against it by cosine similarity, so a flagged match can cite Copyright Act
2022 s.36(2) against a *named* work, not just report a bare number. The
corpus reuses the 34 already-collected, human-reviewed authentic
`TEXTUAL_CORPUS` samples (Punch, Premium Times, Vanguard, BusinessDay,
TheCable, LEADERSHIP) rather than freshly scraping a separate set — real,
bylined, human-authored Nigerian journalism, which is exactly what
Objective 2 protects, and already verified. `build_protected_corpus_index()`
writes a lightweight, re-buildable index (sample_id + provenance) to
`data/protected_ip/protected_corpus_index.csv` rather than copying the raw
text files a second time, specifically to avoid the "files on disk without
manifest rows" bug pattern (CLAUDE.md rule 4). Scoring is whole-document
embedding cosine similarity (kept consistent with how it's validated below);
each protected work is additionally sentence-embedded at index-build time so
a matched result can surface the single closest sentence as a quotable
`matched_passage` for a complaint, not just a filename. **Stated limitation
(§1.7):** n=34, journalism-only — a real deployment protecting Nigerian
creators per §1.1 would also need Nollywood scripts, Afrobeats lyrics, and
other creative works, none of which this project has collected.

**IP_SIMILARITY_THRESHOLD validation (2026-08-04,
`validate_ip_similarity_threshold.py`, `results/ip_similarity_*_control.csv`).**
`config.IP_SIMILARITY_THRESHOLD = 0.82` was a pre-data guess (like the Day 3
NLP thresholds); this is the first time it was actually run against real
data. Two direct tests against the 34-work protected corpus, following the
same "measure first, don't quietly retune" rule as everywhere else in this
log:

- **Negative control (leave-one-out).** Each of the 34 works' best-match
  similarity against the other 33 was computed — all are distinct articles
  by different publishers, so none should score near 0.82. Raw result: 5/34
  exceeded 0.82 (max 0.9390). Inspecting those 5 by hand found two distinct,
  structural root causes rather than the matcher genuinely confusing
  unrelated articles: (1) **same-event convergent journalism** — TheCable
  and Punch both covered the identical ARCON/Tinubu fake-ad story and both
  quote director-general Fadolapo's press statement verbatim, which is
  normal, ethical journalism (quoting an official statement), not one outlet
  copying the other; (2) **same-publisher beat clustering** — three
  BusinessDay NG inflation/monetary-policy pieces share topic, house style,
  and framing without either being derived from the other. Both clusters are
  recorded in the script's `KNOWN_CONFOUND_SAMPLE_IDS` with the reasoning.
  Excluding those 5, the genuine ceiling among the remaining 29 is **0.7615**.
  A raw whole-document cosine-similarity threshold structurally cannot tell
  "quoting the same press statement" or "same reporting beat" apart from
  actual copying — that needs a different signal entirely (e.g. detecting
  attributed/quoted material, or an explicit same-live-event check), which
  is out of scope for this module. **Recorded as a genuine §1.7 limitation,
  not solved by picking a different threshold number.**
- **Positive control (substantial paraphrase).** Four of the 34 works were
  hand-paraphrased — same facts/entities, heavily reworded sentence-by-
  sentence — simulating the actual threat model this module exists for: an
  LLM asked to rewrite a Nigerian article to dodge a naive exact-match
  check. All 4 were correctly attributed back to their source article and
  all 4 scored above 0.82 (floor 0.8676, mean 0.8960).
- **Verdict.** Once the two identified structural confounds are excluded,
  0.82 sits inside a clean **[0.7615, 0.8676] margin** — it clears every
  genuinely-unrelated pair and catches every paraphrase tested. **0.82 is
  defensible as measured; not changed.** The raw, confound-blind numbers
  (5/34 negative-control pairs over threshold) would read as "threshold too
  low" in isolation, which is why the confound inspection step matters and
  is recorded here rather than left as a bare pass/fail — the same lesson as
  the Day 3 heuristic findings: a plain aggregate number can mislead without
  looking at *why* the outliers are outliers. Per CLAUDE.md rule 7 and this
  script's own design, the script does not write to `config.py`; the
  0.82-stands decision above was made separately with this evidence in hand,
  and `config.py`'s `IP_SIMILARITY_THRESHOLD` comment was updated to record
  it (2026-08-04).

**Objective 3 build — `fusion.py` / `rules_engine.py` / `complaint_generator.py`
/ `main_pipeline.py` (2026-08-06, Days 16-20).** Four design decisions worth
recording, since each reverses/qualifies a plausible-but-wrong default:

- **Abstention is excluded from arithmetic, never substituted with 0.**
  Every per-modality signal in this project abstains rather than guess when
  it can't say anything (cv_engine's four Stage 4 heuristics, `ip_similarity
  .check_similarity()`, and the new `nlp_engine.classify_text()`). On real
  data, abstention isn't random: the per-clip coverage diagnostic (§4)
  already found blink-rate's abstention correlates near-perfectly with
  label (genre/edit-pace, not a real signal). `fusion.combine_cv_heuristics()`
  therefore drops an abstained heuristic from the average entirely rather
  than counting it as 0 — averaging 3 abstentions + 1 real 0.9 score
  produces 0.9, not 0.225. Verified directly in `tests/test_fusion.py`
  (`test_single_scored_heuristic_is_not_pulled_toward_zero_by_abstentions`).
- **A single-modality Psynth is that modality's own score, unweighted —
  never `alpha * missing_modality`.** `fusion.compute_psynth()` only applies
  `alpha`/`(1-alpha)` weighting when BOTH `p_visual` and `p_text` exist.
  `alpha * (p_visual or 0)` would silently convert "no visual evidence" into
  "visual evidence of 0.0" — active counter-evidence — which is a false
  claim the system must never make. A text-only `Psynth` (the common case:
  no CV classifier exists yet, and most `TEXTUAL_CORPUS` samples have no
  video at all) is therefore a correct, non-degraded decision, not a
  fallback to apologise for.
- **One `Psynth` gate for all three violation types, including
  `IP_SCRAPING_PLAGIARISM`.** `config.PSYNTH_THRESHOLD`'s own comment says
  it "must EXCEED this value to trigger... Auto-Complaint" with no carve-out
  for the IP path — `rules_engine.evaluate()` follows that literally rather
  than gating IP violations on `IP_SIMILARITY_THRESHOLD` alone. Consequence,
  confirmed against the real corpus below: a human plagiarising another
  human (an IP match with low `p_text`) is correctly out of scope — this
  system detects "AI-generated synthetic media and **AI**-plagiarised IP"
  (per CLAUDE.md), not plagiarism in general.
- **Text with no IP match has no modelled cause of action, even at
  `Psynth` near 1.0.** `rules_engine.determine_violation_type()` returns
  `None` (logged, not escalated) for AI-generated text that reproduces no
  protected work — writing text with an LLM isn't itself illegal in this
  system's scope; only reproducing a specific protected work is. This is a
  deliberate §1.7 scope boundary, not a gap — see the real-data confirmation
  below, where it fires on all 31 real synthetic `TEXTUAL_CORPUS` samples.

`violations.evidence_json` (nullable `Text`, additive-only migration via
`db_connector._ensure_evidence_json_column()`) was added because `p_visual`/
`p_text` alone can't answer "what was actually observed" — which heuristics
scored vs. abstained and why, the classifier's confidence, the matched
protected work and passage. `main_pipeline.py` generates the row's UUID
itself (`uuid.uuid4()`) *before* the insert specifically so this can be
included in one INSERT — `violations` is append-only (NFR-04), so there is
no legitimate UPDATE afterwards to attach it once the row exists.

**Real-sample run (2026-08-06) — what actually happened when this was
pointed at real data, not just unit tests.** Two runs, both against the
actual corpus, both revealing the honest state of the two pathways rather
than a curated success story:

1. **All 31 real synthetic `TEXTUAL_CORPUS` samples scored `p_text` in
   0.92–0.96** via the live classifier (`nlp_engine.classify_text()`, first
   time this checkpoint has been loaded for inference rather than
   evaluation) — comfortably above `PSYNTH_THRESHOLD`. **Zero of them
   triggered a violation.** Every one also ran through
   `ip_similarity.check_similarity()` and matched no protected work above
   0.64 similarity (well below 0.82) — expected, since the synthetic corpus
   is original LLM-generated content, not a reproduction of a real article.
   This is the text-without-IP-match scope boundary above, confirmed against
   every real sample that could have exercised it, not just a unit test.
2. **One real video sample did cross the threshold**: `b1ce1815`
   (`TELECOM_FMCG_ADS`, label=SYNTHETIC, an MTN Instagram ad,
   `ai_avatar_ad` genre) produced `Psynth = 1.0` and a real
   `UNDISCLOSED_SYNTHETIC_AVATAR` complaint (ARCON + NDPA s.37,
   `reports/complaint_7d69408c-45c4-46e4-8157-5edda31a2eb9.pdf`) via
   `main_pipeline.py --sample-id`. **This must not be read as "the CV
   pathway works."** `Psynth = 1.0` here is a bare average of ONE
   heuristic — `blending_artifacts` scored 1.0; the other three
   (blink_rate, emotional_rigidity, viseme_mismatch) all abstained because
   the clip's longest contiguous face run (2.4s) is below their coverage
   gates. Cross-checking `results/cv_heuristic_eval.csv` (§4's full
   32-clip evaluation): only 3 of 32 clips ever hit `blending_artifacts
   >= 1.0` this way — two synthetic (`b1ce1815`, `f9bf1f34`) and **one
   authentic** (`ed3c9aad`) — consistent with §4's own finding that
   blending-artifacts shows heavy overlap between classes. The complaint
   PDF states this explicitly (which heuristics scored vs. abstained, and
   the CV-heuristic-reliability caveat) precisely so a human reviewer does
   not mistake a single-heuristic score for corroborated evidence. Treat
   this as "the pipeline is wired correctly end to end," not "the visual
   detector is validated" — that second claim would need the Days 10-15
   trained CV classifier this project doesn't have yet.

---

## 3. Bug log

Every one of these was found by running against real data or a real database, not
by reading code. They are recorded so the same class of mistake is caught faster,
and because several are worth citing in Chapter 4 as genuine engineering findings.

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | `ForeignKeyViolation` on `setup_db.py` cleanup | `autoflush=False` meant `session.delete(report)` stayed pending while a raw bulk DELETE hit the parent `violations` row | `session.flush()` between the two |
| 2 | Bug #1 passed 58/58 tests but failed on real Postgres | SQLite does not enforce foreign keys without `PRAGMA foreign_keys=ON` | Pragma enabled for SQLite test engine; regression tests added |
| 3 | `.env` silently ignored since Day 1 | `python-dotenv` was in `requirements.txt` but `load_dotenv()` was never called | `load_dotenv()` at top of `config.py`. Masked for weeks because `DATABASE_URL`'s hardcoded fallback happened to match the `.env` value |
| 4 | `cookiesfrombrowser=chrome` failed | WSL2 — Chrome is installed Windows-side, cookie DB is not in the Linux home directory | Use an exported `cookies.txt` (`YTDLP_COOKIES_FILE`) instead |
| 5 | Instagram downloads failed with "Requested format is not available" | `best[height<=720]` had no fallback; some posts expose no format matching that filter | `best[height<=720]/best` |
| 6 | **6 videos silently overwrote each other** | Output template used `%(title)s` only; Instagram returns a generic shared title (`Video by mtnng`) across different posts | `%(title)s_%(id)s.%(ext)s` plus `nooverwrites: True` |
| 7 | `cookies.txt` rejected as invalid Netscape format | One line had domain `facebook.com` (no leading dot) paired with subdomain flag `TRUE`; Python's parser requires agreement | Normalise to `.facebook.com` |
| 8 | `get_violation_by_id()` crashed on string input | UUID column compared against a raw string from a Flask route | Explicit UUID coercion |
| 9 | Two authentic `TEXTUAL_CORPUS` samples silently corrupted — BusinessDay inflated from ~721 to 11,300 words, Punch from ~462 to 1,577 | stdlib `html.parser` does not implement HTML5's implicit `<p>`-closing rule. Several Nigerian news sites (confirmed: BusinessDay, Punch) emit article HTML with unclosed `<p>` tags, so `html.parser` nests each paragraph inside the previous one instead of treating them as siblings. `find_all("p")` then `get_text()` recursed into every nested descendant, producing cascading duplication (para 1, then para 1+2, then para 1+2+3, ...) | Switched to lxml's HTML parser, which correctly auto-closes `<p>` per spec, restoring flat sibling paragraphs. Added a boilerplate-`<p>` filter alongside it (strips "Read also", newsletter/WhatsApp-channel CTAs, etc. that are real `<p>` tags but not article body text) |
| 10 | `watchdog/cv_engine.py`'s `analyze_clip()` raised `ModuleNotFoundError: No module named 'mediapipe.python'` the first time it was actually run against a real video | Written against MediaPipe's legacy `solutions.face_mesh` API (`from mediapipe.python.solutions import face_mesh`), which reads as correct and imports fine at the module level (the failing import is inside the function, lazy-loaded). The installed `mediapipe==1.0.0` removed the entire `solutions` API in favour of the Tasks API — the old call path no longer exists at all. No test existed for `cv_engine.py` at the time, so nothing exercised the function until it was actually called | Rewired the dense-landmark stage onto `mediapipe.tasks.python.vision.FaceLandmarker`, which requires an explicit downloaded model bundle (`models/face_landmarker.task`, `config.FACE_LANDMARKER_MODEL_PATH`) rather than one bundled in the package. Added `tests/test_cv_engine.py` so this path has coverage going forward |
| 11 | **Near-miss, not an actual failure.** `config.MANIFEST_PATH` (`data/manifest.csv`) resolved to nothing — the entire `data/` directory (manifest, strata.csv, raw + processed video/text) was found sitting at `dashboard/data/` instead of the project root, discovered 2026-08-04 while starting the full-corpus CV evaluation | Timestamps show `dashboard/data`'s contents predate `dashboard/` itself being touched — consistent with `data/` having been moved under `dashboard/` (likely while scaffolding the Flask dashboard directory layout) without updating `config.py`, and no dashboard app code exists yet that actually reads from there. A stray, fully-empty directory literally named `{data/{raw/{video,text},processed,protected_ip},models,watchdog,dashboard/{templates,static},reports,tests,notebooks}` was also found alongside it — an unrelated leftover from an earlier `mkdir -p` brace-expansion that didn't expand (e.g. run under `sh` instead of `bash`) | Moved `dashboard/data` back to `data/`, deleted the stray brace-expansion directory, and verified `data/strata.csv`'s `matched_subset` crosstab still reads 4 authentic / 9 synthetic post-move (i.e. no rows were lost or corrupted by the relocation). **This would have been a fourth instance of the silent-failure pattern behind bugs #6/#9/#10** — caught only because the manifest was inspected before any script ran, not because any script failed loudly. `watchdog/dataset_manager.require_manifest_exists()` was added afterward specifically to make this class of mistake fail loudly next time: evaluation entry-point scripts (`evaluate_nlp_heuristics.py`, `evaluate_cv_heuristics.py`) now call it at startup and refuse to run against a missing manifest, rather than silently evaluating zero samples. Deliberately not enforced inside `load_manifest()` itself or at `config.py` import time — a missing manifest is the normal, correct state on a first run before `collect_dataset.py` has ever executed, and library code returning an empty DataFrame there is intentional, not a bug |
| 12 | **Two `setup_db.py` test fixtures (`publisher_name="TEST SETUP"`, `DEEPFAKE_IMPERSONATION`) survived undetected in the real `violations` table for over a month** (inserted 2026-07-03 and 2026-07-14, found 2026-08-13) | `setup_db.py`'s Step 5 cleanup deletes its own throwaway violation via the `allow_violation_mutation()` admin bypass, but wrapped that delete in `except Exception: logger.warning(..., "non-critical")` — on both real runs, cleanup silently failed (or was interrupted) and the script still printed "DAY 1 DATABASE SETUP COMPLETE" and exited 0. Neither run raised anything an operator would see. Found only when the Days 21-22 dashboard rendered 3 violations against an expected 1 | `purge_setup_fixtures.py` (new, idempotent — queries live by `publisher_name`) removed both, independently re-verified by querying `violations` again afterward (count: 1, the real MTN-ad detection). `setup_db.py`'s Step 5 no longer swallows cleanup failure as "non-critical" — it now exits non-zero and skips the success banner, AND independently re-queries for the row after commit rather than trusting a clean commit as proof (a clean commit is exactly what this bug's two occurrences had). Re-run twice against the real DB after the fix: both times exited 0 with `VERIFIED gone`, and `violations` stayed at exactly 1 row throughout |

**Bug #6 is the important one.** It produced no error — the run reported complete
success while destroying five of six samples. Any operation writing files derived
from external metadata needs a uniqueness guarantee that doesn't depend on that
metadata being distinct.

**Bug #9 is the second instance of the same failure mode as bug #6.** It also
produced no error, and it also passed every test — `pytest tests/` was green the
entire time these two samples were quietly wrong. Both bugs were caught only by
inspecting real output against the real source (an 11,300-word BusinessDay article
is implausible on its face; so is a Punch report over 1,500 words). No test
fixture would have caught either, because both bugs are about *silent* data
corruption, not a thrown exception — a lesson worth stating plainly rather than
re-learning a third time: passing tests and a clean exit code are not evidence
that scraped or generated content is correct. Spot-check real output.

**Bug #10 is the third instance of the same pattern as #6 and #9, in a different
shape.** The first two were silent *data* corruption — wrong output, no exception,
passing tests. #10 was silent under code review instead — the code read correctly,
imported without error, and nothing in the test suite touched it (no
`tests/test_cv_engine.py` existed yet). It was caught only by actually running
`analyze_clip()` against a real video, at which point it failed loudly and
immediately. Same underlying lesson as #6/#9, restated once more because it keeps
recurring in a new form: neither a clean diff nor a green test run is evidence
that code path actually works — only running it against real input is.

**Bug #12 is the fourth instance of the same pattern, in its most dangerous
shape yet.** #6 and #9 were silent *data* corruption; #10 was silent under
code review but failed loudly the moment it actually ran; #12 combines the
worst of both — it failed loudly (an exception genuinely was raised, twice),
but the code caught that exception and *downgraded it to a warning*, so the
loud failure never reached anyone. A `try/except` that logs at the wrong
severity is functionally identical to no exception handling at all from the
operator's point of view, and worse than #6/#9 in one respect: those
corrupted a handful of dataset text files; this one corrupted the actual
legal-evidence table Chapter 4 evaluates and every complaint PDF cites as its
audit trail. Caught by the same discipline as the other three — inspecting
real output (the dashboard's row count) against what was actually
expected — not by any test (`pytest tests/` was green throughout; a
SQLite-backed test can't catch a bug that only manifests via cleanup
silently failing against a real, already-populated Postgres instance run
outside the test suite). The general lesson this project's bug log keeps
re-deriving in new forms: a caught exception is not a handled exception
unless something checks that the recovery actually worked — logging and
continuing is a choice to hide the failure, not fix it, and should be
reserved for cases where "continue anyway" is genuinely the correct
behaviour, which "a legal audit table still contains a labelled test
fixture" never is.

**Still unresolved:** Facebook video downloads fail with `No video formats found`
even with valid `c_user`/`xs` cookies loading correctly. Auth is confirmed working
(everything else in the same run succeeds). Most likely the content is unavailable
to the account, or a `yt-dlp` extractor limitation with Facebook Watch/Reels. Both
Facebook items (Golden Terra, iLOTBET) remain usable as *citations* even without
the raw files.

---

## 4. Dataset state and known problems

As of end of Day 3: 25 samples total. `TELECOM_FMCG_ADS` 11 (all synthetic),
`POLITICAL_MEDIA` 0, `TEXTUAL_CORPUS` 14 (3 authentic, 11 synthetic). Then +21
synthetic text samples added by the corpus expansion.

**Four known data-quality issues — #3 and #4 resolved, #1 and #2 outstanding:**

1. **Severe class imbalance.** Roughly 31 synthetic vs 3 authentic text samples.
   No accuracy figure computed on this is meaningful — a constant "synthetic"
   predictor scores ~91%. **The authentic side must be expanded before any
   evaluation is interpretable.** Add 20–30 real Nigerian news URLs to
   `TEXTUAL_CORPUS_SPECIFIC_URLS` and scrape them.

2. **Length confound — corrected numbers, still unresolved (worse than thought).**
   The previously-recorded authentic range (360–12,965 words) was not real data —
   the top end was bug #9 (see §3). After the lxml fix and re-scrape, the actual
   4-sample authentic range is **293–721 words** (TheCable 293, leadership.ng 363,
   Punch 462, BusinessDay 721). Current synthetic range is 57–217 words. That means
   **zero overlap** between the two classes right now — every authentic sample is
   longer than every synthetic sample, which is a worse confound than the old
   (partly fictitious) 9%-overlap figure suggested. The confound was always real;
   it was partly *manufactured* by the scraper bug, which inflated the authentic
   ceiling far past where real samples actually sit. Consequence for
   `expand_synthetic_corpus.py`: its header cites the old 360–12,965 range and
   targets 600–1,600-word synthetic samples to "overlap the authentic length
   distribution." That target range overshoots the real authentic ceiling (721)
   considerably — the gap it's closing is narrower than its own reasoning assumed.
   The genre-diversity rationale (Pidgin English, bureaucratic text — §3.9 gaps)
   is unaffected and still valid; only the length-matching rationale needs
   revisiting before that script is run.

3. **RESOLVED — the 12,965-word sample was a scraping artifact, not real data.**
   Root-caused and fixed as bug #9 (§3): stdlib `html.parser`'s missing HTML5
   `<p>`-auto-close support caused cascading paragraph duplication. Fixed by
   switching to lxml; the BusinessDay sample was re-scraped and corrected from
   11,300 to 721 words (Punch, hit by the same bug, corrected from 1,577 to 462).
   See bug #9 in §3 for the full root cause and fix.

4. **RESOLVED — the leadership.ng iLOTBET article is label=0 (AUTHENTIC).**
   Decision: this article is human commentary/journalism *about* an AI-generated
   ad, not AI-generated text itself. `TEXTUAL_CORPUS`'s label answers only "was
   this text written by a human or an LLM" — the opinionated, informal voice
   ("cheapest, ugliest, most robotic AI graphics imaginable") is clearly
   human-written, branded-content caveat notwithstanding. The synthetic artifact
   it describes (the iLOTBET ad itself) is separately and correctly catalogued
   under `TELECOM_FMCG_ADS` with label=1 — one real-world event, two dataset
   entries, two different labels, and that's correct because the two entries
   answer different questions. Recorded in `data/manifest.csv` notes and
   `seed_sources.py` for traceability. (Re-checked while investigating bug #9:
   the garbled mid-sentence text once visible in this sample — "forces a brutal
   que a company that claims" — is gone in the current fetch; it does not
   reproduce with the lxml-based scraper, so it does not need to be treated as a
   corrupted sample.)

**Day 3 evaluation findings** (`results/nlp_heuristic_eval.csv`):

- **Burstiness contributed nothing.** AUTHENTIC mean −0.420 vs SYNTHETIC −0.418.
  Worse: every sample scored below `BURSTINESS_THRESHOLD = 0.3`, so the flag fired
  100% of the time and the two-signal design collapsed into one signal. The
  threshold was set before any real data existed. Likely also entangled with the
  length confound — short texts have fewer sentences, mechanically reducing
  variance.
- **Perplexity ran backwards.** Synthetic mean 60.9 vs authentic 30.3 — LLM text
  scored *more* surprising, the opposite of the assumption. Plausible cause:
  DistilGPT-2 is trained on Western English, and the synthetic samples deliberately
  use Nigerian phrasing ("wahala", "Owambe", "Detty December") which is
  out-of-distribution regardless of author. Meanwhile real Nigerian tech journalism
  sits closer to the model's training distribution.

That second finding is **empirical support for the project's own thesis** (§1.1,
§2.3.1: Western AI tools transfer poorly to Nigerian content). It should be
written up as a result, not treated as a failure.

**Do not retune thresholds to fix these at current N.** That is fitting noise. The
Day 8–9 fine-tuned classifier should learn the boundary from data.

**Stage 2 video-corpus trimming (2026-08-03).** Three long-form video samples
(`33bc415e` 193.4s, `1aade253` 163.0s, `fc4b1a55` 140.2s) measured over NFR-02's
30s budget on the real Stage 2 face pipeline (`analyze_clip()`, sparse-MTCNN/
dense-Face-Mesh — see `config.py` and `watchdog/cv_engine.py`). Each was checked
individually via ffmpeg scene-cut detection (`select=gt(scene,0.15)`) plus
contact-sheet visual review for a continuous, no-cut, on-camera window, following
the same method used for the three earlier 55s trims (`83a9ffbb`, `b8cbac67`,
`5f52f83f`):

- **`fc4b1a55` (Dangote/"Partner for Growth" interview) — trimmed to 55s
  (35–90s).** Clean case: the original is a single continuous two-shot interview
  for its entire 0–134s body (only one detected cut, at 134.1s, into the closing
  card). Trim written to `data/processed/video/`, original untouched, manifest
  repointed.
- **`33bc415e` (healthcare-ceremony panel news package) — trimmed to 36s
  (133–169s), not 55s.** This one is a real multi-speaker panel edited with cuts
  every 30–45s; no window near 55s is actually unbroken. The longest genuine
  continuous window is 133–169s — one speaker the whole time (the on-screen name
  credit changes mid-window from Prof. Obinna Onwujekwe to Prof. Adetokunbo
  Fabamiwo, but the shot itself — same person, same background, one in-camera
  zoom — never cuts). The zoom was accepted as camera movement, not an edit cut.
  Shorter than the other four trims but the same *kind* of window (continuous,
  no cuts), and comfortably inside NFR-02.
- **`1aade253` (Arise News "Y'elloTide" vox-pop package) — excluded from the
  CV-eligible video set, not trimmed.** This is the choppiest of the three: a
  title card followed by several different named people (an MTN spokesperson,
  multiple voucher winners) on screen for 10–30s each between B-roll cutaways.
  The longest unbroken single-face window found is ~20s (Kolapo Olaleye,
  35.36–54.96s) — too short to support blink-rate or emotional-rigidity analysis,
  both of which need continuous tracking of one face for longer than that. The
  manifest row and the raw file are both kept (not deleted); `data/manifest.csv`
  marks it `EXCLUDED FROM CV FACE-ANALYSIS SCOPE` with the reasoning, so the
  decision is traceable and reversible if the heuristics later prove tolerant of
  shorter windows.

**This drops the CV-eligible authentic video count by one**: 19 authentic video
rows remain in `data/manifest.csv` (nothing was deleted), but only 18 are
currently eligible for face-based CV analysis, against 14 synthetic — worth
stating explicitly in Chapter 4 rather than quoting the raw manifest count.

**Stage 4 face-coverage diagnostic (2026-08-03), run before building any
heuristic.** `analyze_clip()` across all 32 CV-eligible clips, grouped by label:

| | n | mean face-present ratio | mean longest run | median longest run | max longest run |
|---|---|---|---|---|---|
| authentic | 18 | 49.7% | 15.9s | 4.0s | 68.8s |
| synthetic | 14 | 36.2% | 3.9s | 3.8s | 6.5s |

Mean face-present ratio is close between classes and not that informative on
its own — the real finding is at the tail. **At longest_run ≥ 8s, 6/32 clips
qualify and all six are authentic. Zero synthetic clips ever exceed 6.5s of
contiguous face tracking** — every `ai_avatar_ad` sample cuts too often for
MTCNN to hold a track that long. At ≥3s, coverage is much more balanced (12
authentic / 10 synthetic). This directly set `config.MIN_FACE_RUN_FOR_BLINK_S
= 10.0` (blink rate needs at least one plausible inter-blink interval to be
anything more than noise) and `MIN_FACE_RUN_FOR_RIGIDITY_S =
MIN_FACE_RUN_FOR_VISEME_S = 3.0`. **Consequence: the blink-rate heuristic will
structurally abstain (return `None`) on every synthetic clip in the current
corpus** — not a bug, a genre-driven coverage confound (short-cut ad footage
vs. longer-form interview/panel content) that must be stated plainly in
Chapter 4 rather than left to read as "no opinion" from a silent `None`.

**Stage 4 heuristic calibration (2026-08-03), 9 clips (4 authentic
interview/panel, 5 synthetic `ai_avatar_ad`).** Before wiring in
`config.EMOTIONAL_RIGIDITY_THRESHOLD`, `VISEME_CORRELATION_THRESHOLD`, and the
two `BLENDING_*` thresholds, each heuristic's *raw* measurement (not the
score) was checked against real clips, since the write-up's
`EMOTIONAL_RIGIDITY_THRESHOLD=3` turned out to be off by roughly four orders
of magnitude from the real measurement scale (normalised displacement
variance sits in the 1e-5–1e-3 range, never near 3):

- **Emotional rigidity.** Two of the 9 clips (one authentic — `fc4b1a55`, one
  synthetic — `6d95436c`) measured wildly higher variance (0.12, 0.03) than
  the rest (1.3e-5–3.3e-4). Both are scenes with a second face visible;
  the likely cause is facenet-pytorch MTCNN's `keep_all=False` picking
  whichever face has higher confidence at each sparse detection tick, so the
  "tracked" identity can flicker between two different people mid-run,
  producing a huge spurious displacement at the switch rather than a real
  micro-expression signal. This is a real limitation of the single-face
  tracker on multi-person scenes, not calibration noise — worth fixing
  (e.g. IoU-based track continuity against the previous box, not just
  highest confidence) before this heuristic is trusted on multi-face content.
  Excluding those two, the 6 clean values do **not separate cleanly by
  label** (authentic: 9.0e-5, 1.3e-4, 2.5e-4; synthetic: 1.3e-5, 7.3e-5,
  3.3e-4) — the lowest (most "rigid") value in the whole sample is actually a
  synthetic clip, but so is the highest.

  **2026-08-04 follow-up.** `keep_all=True` plus IoU-based track continuity
  (`config.MTCNN_TRACK_IOU_THRESHOLD`, `detect_face_bbox()` in
  `watchdog/cv_engine.py`) were implemented and re-run against both outlier
  clips. Only one resolved:
  - `fc4b1a55`: 0.12 -> 7.7e-5 — now inside the clean range. This one was
    genuinely the keep_all=False multi-face misattribution the fix targets.
  - `6d95436c`: 0.03 -> 0.027 — essentially unchanged. Frame-by-frame
    inspection (dumping the tracked crop at each Face-Mesh tick around the
    spike, t=9.0-9.3s) found a **different** root cause: a hard scene cut
    landing inside a single ~1s MTCNN sampling interval. The crop is
    identical, continuous close-up footage at t=9.0s and t=9.1s, then at
    t=9.2s — one dense tick later, same held bounding box — the ad has cut
    to a completely different shot (same actor, now holding a product box,
    much wider framing). MTCNN doesn't re-detect until its next ~1s tick, so
    Face Mesh spent that one tick reading landmarks out of a stale,
    now-wrong crop. There is no second same-tick candidate for IoU
    continuity to disambiguate against here — the content changed *between*
    MTCNN ticks, not the face count *within* one — so this is a structurally
    different failure mode that IoU continuity cannot fix. Not addressed;
    see `config.py`'s `EMOTIONAL_RIGIDITY_THRESHOLD` comment. Fast-cut
    `ai_avatar_ad` footage (6d95436c's genre) is exactly where a cut is
    likely to land inside a 1s MTCNN window, so this is plausibly a
    genre-correlated failure mode worth watching for in the full-corpus
    evaluation, not a one-off.
- **Viseme-phoneme mismatch.** Correlations clustered near zero for BOTH
  classes (authentic: −0.055 to 0.127; synthetic: −0.161 to 0.411) — the
  opposite of the "genuine speech shows strong positive correlation"
  assumption the heuristic is built on. Possibly a real limitation of coarse
  (50ms) RMS energy without band-pass filtering to speech frequencies, or of
  the simplified 2-point MAR. This is the same category of honest null
  result as the Day-3 NLP finding that burstiness "contributed nothing" —
  recorded here rather than tuned away.

  **Audio-decode cost fix (2026-08-04).** `librosa.load()` on an MP4 has no
  fast native path — PySoundFile can't read MP4 containers, so librosa
  silently falls back to `audioread`'s ffmpeg-subprocess backend on every
  `compute_viseme_mismatch()` call. Fixed by extracting audio to a WAV file
  once via a direct ffmpeg call in `_track_clip()`
  (`_extract_audio_wav()`, `config`-free — no new threshold, just a
  subprocess call), then loading that WAV instead — `librosa`'s fast native
  soundfile path handles a WAV directly. Isolated before/after on the raw
  decode primitive (excluding MTCNN/tracking): direct-`mp4`-via-audioread
  averaged ~0.32s/call warm (1.6s cold); ffmpeg-extract-once (~0.15s) +
  `wav`-load averaged ~0.02s/call after — roughly a 16x reduction on the
  decode itself, with identical output (`mouth_audio_correlation` unchanged
  before/after on `d5e94745`).
  **However**, re-measuring `d5e94745` (`run_heuristics()`, full pipeline)
  only dropped from ~23.8s to ~23.4s total, because `compute_viseme_mismatch()`
  is called once per clip, not repeatedly, so the one-call saving is small in
  absolute terms — and because tracking (`MTCNN`+`Face Mesh` across the whole
  clip) dominates this specific clip's cost at ~21s, not audio decode.
  `d5e94745` is 128.4s long — well outside the other Stage-2-trimmed clips'
  30-55s range and NFR-02's 30s budget — and was never triaged in the Stage 2
  trimming pass above. The fix is real and correctly implemented (verified:
  temp WAV cleaned up after use via `run_heuristics()`'s `finally` block, no
  leftover files), but it was not the dominant cost for this particular
  clip; `d5e94745`'s own untrimmed length is the bigger NFR-02 risk and is a
  separate, not-yet-made decision (same kind of scene-cut review the other
  long clips got) rather than something silently fixed here.
- **Blending artifacts.** Face/surround Laplacian-variance ratios overlapped
  heavily between classes (authentic 0.47–1.29, synthetic 0.61–0.92); edge
  discontinuity was small and mostly negative for both.

**None of the three thresholds calibrated this way should be read as
validated decision boundaries** — they're real-scale, real-data starting
points (replacing a guess that was wrong by orders of magnitude), set from 9
clips because that's what a quick calibration pass could afford, not a
systematic evaluation. A full run across all 32 CV-eligible clips, split by
`matched_subset`, is the next step before drawing any conclusion about
whether these heuristics discriminate at all. Do not retune these thresholds
to make a future evaluation number look better without new evidence — same
rule as the Day 3 NLP thresholds.

**Full 32-clip CV heuristic evaluation (2026-08-04, `evaluate_cv_heuristics.py`,
`results/cv_heuristic_eval*.csv`).** Ran all four heuristics across every
CV-eligible clip. Primary result is `matched_subset` (n=13: 4 authentic/9
synthetic) — the full 32-clip corpus is confounded by genre/edit pace and is
reported only as secondary context. On the primary comparison:
- **Blink rate produced zero usable data** — 100% abstention in BOTH classes
  within `matched_subset` (not just synthetic, as the 9-clip coverage
  diagnostic alone would have predicted — none of the 4 matched authentic
  clips happened to clear the 10s floor either). It contributed nothing to
  this comparison at all.
- **Emotional rigidity**: authentic mean/std exactly 0.0/0.0 (n=3 scored) —
  every non-abstained authentic clip's variance exceeded
  `EMOTIONAL_RIGIDITY_THRESHOLD`, clamping the score floor; synthetic mean
  0.133 (n=7), driven by a couple of high outliers, not a clean shift.
- **Viseme-phoneme mismatch**: ran BACKWARDS — authentic mean 0.183 (n=3) vs
  synthetic mean 0.039 (n=7), i.e. authentic clips scored as the worse
  lip-sync match. Same category of inverted-direction result as the Day 3
  NLP perplexity finding.
- **Blending artifacts**: heavy overlap, authentic mean 0.647 vs synthetic
  0.524 (n=4/9) — no separation, and also backwards in direction.

Net result: **all four CV heuristics fail to cleanly discriminate on the
matched-subset comparison** — one produces no data at all, the rest either
show no separation or point the wrong way. This meets or exceeds the
pessimistic expectation set going in and is the direct trigger for pivoting
straight to Day 8-9's trained classifier (see below) rather than building a
fusion layer on top of heuristics that don't individually work.

**Pivot to classifier training (2026-08-04).** Both heuristic pathways
(NLP: `results/nlp_heuristic_eval.csv`, 67.7% UNCERTAIN vote rate; CV: the
32-clip result just above) return too little usable signal to justify
building the §3.11 fusion layer on top of them first. Moved straight to
Day 8-9's fine-tuned DistilBERT classifier — `train_nlp_classifier.py` — to
establish whether a trained model does meaningfully better before any more
heuristic-layer work happens. CV's own trained classifier (EfficientNet-B0,
Days 10-15) is the equivalent next step on that side, not yet built.

**NLP classifier result (2026-08-04, `train_nlp_classifier.py`,
`results/nlp_classifier_*.csv`).** Fine-tuned `distilbert-base-uncased` via
5-fold `StratifiedGroupKFold` (not a single split — n=65 makes a single
13-sample test set too noisy to trust) on all 65 labelled `TEXTUAL_CORPUS`
samples (34 authentic, 31 synthetic). Groups: authentic by publisher (7
groups), synthetic by generation batch — the 10 original samples
(`generate_synthetic_corpus.py`, tagged `claude-sonnet-5` in `source_url`)
and 21 expansion samples (`expand_synthetic_corpus.py`, tagged `claude`)
split further by genre, 10 groups total, 17 groups overall. Ran on CPU in
631.6s (~10.5 min) total for 5-fold CV + one final fit on all data —
practical, no need for Colab.

Pooled out-of-fold results across all 65 samples: precision 0.897, recall
0.839, F1 0.867, **FPR on authentic (NFR-03's tracked metric) 0.088 (3/34)**
— down from the Day 3 heuristic baseline's 0.382, and unlike the heuristic
vote the classifier never abstains (baseline UNCERTAIN rate was 67.7%). Real
per-fold spread, not just the aggregate: FPR ranged 0.0-0.182 and recall
0.429-1.0 across the 5 folds — small, unevenly-sized groups mean per-fold
numbers are noisy; the pooled aggregate is the number to trust.

Two things checked and stated explicitly rather than left implicit:
- **Register effect does NOT appear to persist.** The 4 Pidgin-genre
  synthetic samples (`pidgin_ad` x2, `pidgin_social`, `pidgin_news`) scored
  mean P(synthetic)=0.915 and were 4/4 correctly classified — HIGHER
  confidence than the 27 non-Pidgin synthetic samples' mean 0.677, the
  opposite direction from Day 3's perplexity finding (which ran backwards on
  Nigerian-register text). Checked against a length-shortcut explanation:
  the 4 Pidgin samples are 167-292 words, unremarkable within the synthetic
  class's own 57-849 word range, so this isn't just "short text scores
  higher."
- **All 31 synthetic samples came from one LLM family (Claude)** —
  whatever the classifier learned is Claude's writing style specifically,
  not "AI-generated text" in general. Generalisation to GPT/Gemini/Llama
  output is untested and should not be assumed.

A third caveat not asked for but directly relevant to interpreting the
above: the **length confound flagged in this section for the Day 3 n=14 data
is still present at n=65** — authentic mean 865 words (range 70-1893) vs
synthetic mean 303 words (range 57-849). There's real overlap now (unlike
Day 3's zero-overlap case), but the means are still far apart, so some of
the classifier's performance could be a length shortcut rather than a
genuine authorship signal — this cannot be ruled out from the numbers above
alone and would need either a length-matched held-out set or per-length-band
breakdown to check directly, neither of which has been done yet.

Final model (trained on all 65 samples after the CV loop, standard
practice) saved to `config.NLP_MODEL_PATH`
(`models/nlp_watchdog_v1.pt`) — its own accuracy is not independently
measured; the 5-fold numbers above are the honest estimate for what this
training procedure produces, not a claim about that specific checkpoint.

**Length-confound test (2026-08-04, `analyze_length_confound.py`,
`results/nlp_length_*.csv`).** The classifier result above was NOT accepted
at face value — the length-confound caveat it raised was tested directly
against the real out-of-fold predictions in `results/nlp_classifier_cv_eval.csv`
before treating 0.088 as the real NFR-03 number:
- **Per-length-band breakdown**: accuracy was 0.765-1.000 across four word-
  count bands (0-200, 200-400, 400-800, 800+). It did NOT collapse in the
  200-400 and 400-800 bands where both classes are actually mixed — the
  lowest accuracy (0.765) was in the 200-400 band, still far above chance,
  though that band's authentic n=3 makes its own FPR figure (0.333, 1/3)
  too small to trust on its own.
- **Length-matched subset (200-850 words, n=40: 19 authentic/21 synthetic —
  above the ~20-sample floor for a real, not just indicative, result)**:
  accuracy 0.850, F1 0.850, FPR 0.105 (2/19) — within ~2 percentage points
  of the full-corpus aggregate (F1 0.867, FPR 0.088) in BOTH directions.
  Mean P(synthetic) stayed clearly separated even here (0.211 authentic vs
  0.701 synthetic) — a pure length shortcut would produce near-50/50 scores
  once length itself carries little information, which is not what happened.
- **Length-only baseline ceiling**: the best possible word-count-alone rule
  (optimistic, in-sample/resubstitution threshold search, not held-out) tops
  out at 0.846 accuracy / 0.118 FPR. The real classifier's honest
  out-of-fold accuracy (0.877) beats even this generous, biased-in-its-favour
  ceiling. Word count alone cannot explain the classifier's performance.

**Verdict: the length signal is not doing significant work.** NFR-03's 0.088
FPR figure stands as measured — this was checked, not assumed. Per the
instruction this analysis was run under: no threshold or training change was
made as a result, since none was needed. If a future, larger corpus shifts
this picture, re-run `analyze_length_confound.py` before trusting the
aggregate number again rather than assuming this result is permanent.

**Cross-LLM generalisation — UNTESTED, not just unresolved (2026-08-04).**
The classifier has only ever seen synthetic text from one LLM family
(Claude — see the single-LLM caveat above). Checked whether this project has
access to a second model family (GPT/Gemini/Llama) to generate a genuine
held-out generalisation test: no API key for any non-Anthropic LLM provider
exists in `.env` or is referenced in `.env.example`/`requirements.txt`, and
there is no tool available in this environment to call one directly. This
test was **not run** — it was not attempted with a substitute or
approximated with same-LLM data dressed up as different. **This is a real,
open limitation, not a checked-and-cleared one**: unlike the length
confound above, nothing here rules out the classifier having simply learned
Claude's house style rather than general LLM-authorship signal. Must be
stated as such in §1.7/§5.5 rather than implied to be covered by the
length-confound check, which tested a different hypothesis. Resolving this
needs either genuine access to a second model family, or an explicit,
acknowledged scope limitation in the write-up.

**CV transfer-learning experiment — BLOCKED, deferred (2026-08-04).** The
planned Days 10-15 replacement (fine-tune EfficientNet-B0 on FaceForensics++
or Celeb-DF, validate on the benchmark's own test split, then evaluate
zero-shot on the 32 CV-eligible Nigerian clips to measure the transfer-
degradation gap §2.3.4 predicts) was assessed for feasibility before any
build work started, per the same "verify before claiming" convention as
everything else in this log. Checked both benchmarks' actual current access
procedure directly (not from memory):
- **FaceForensics++**: Google Form -> manual human review by the TU Munich
  authors -> signed Terms of Use -> download link, ~1 week turnaround. No
  bypass.
- **Celeb-DF**: same pattern — Google/Tencent form -> manual approval ->
  signed terms.
- A HuggingFace re-upload of FaceForensics++ (`Ella4839/FaceForensics_plusplus`)
  was found and **deliberately not used** — it's very likely an unauthorized
  mirror of a dataset whose own Terms of Use require a signed agreement and
  prohibit redistribution. Using it would be exactly the kind of
  unverifiable, indefensible-in-a-viva shortcut §1.6's scope argument (and
  this file's own "traceability beats scale" framing, §1) rules out for this
  project — a data source that couldn't be cited honestly in Chapter 3.

Neither dataset is obtainable from within an automated session — both forms
require a real institutional identity and a human decision on the other
end. **Decision: the student is submitting both access requests directly**
(2026-08-04); this is not something that can be delegated. The transfer
experiment is deferred pending that approval, **with no guarantee it arrives
within the project's remaining timeline** — if approval doesn't come back in
time, this is a genuine §1.7 scope limitation (no accessible standard
deepfake benchmark for a supervised CV transfer-learning test within the
project window), not a failure to execute. Do not substitute an
unofficial/unverified copy of either dataset if the wait becomes
inconvenient — the same reasoning that ruled out the HuggingFace mirror
today applies regardless of deadline pressure later.

**CV transfer notebook — reviewed cell-by-cell before handoff to Colab, one
blocking bug fixed, one methodological caveat logged (2026-08-07).**
`notebooks/train_cv_classifier_colab.ipynb` (FF++ training) and
`evaluate_cv_transfer.py` (zero-shot Nigerian-corpus evaluation) exist and were
walked through in detail before running either. Checked and confirmed sound:
the identity-grouped train/test split (union-find spans Deepfakes and
NeuralTextures together, per the reciprocal-pair/overlapping-identity-pool
issue described above, plus an independent raw-identity-number leak
assertion that doesn't trust the grouping logic it's checking); the ≥90% FF++
accuracy gate (confirmed it `raise`s and halts a "Run All", not just warns —
a redundant assert was added to the checkpoint-save cell too, so it still
refuses to save if that cell is ever run out of order). One blocking bug was
found and fixed: the notebook's discovery globs assumed
`original_sequences/`/`manipulated_sequences/` sat directly under the
extracted zip root, but the actual `ff_subset.zip` wraps them in a top-level
`ff_data/` folder — fixed with a guarded
`if (FFPP_ROOT / "ff_data").exists(): FFPP_ROOT = FFPP_ROOT / "ff_data"`
rather than hardcoding the prefix into the globs, so it survives a future
re-export that doesn't have the wrapper. Also added: extraction (15-45 min of
I/O-bound work, previously living only on Colab's ephemeral `/content` disk)
now backs itself up to Drive as a zip and restores from that cache on a later
run instead of re-extracting, so a mid-training T4 reclaim no longer means
redoing it.

**One finding couldn't be fixed, only disclosed, and belongs in the write-up
alongside the transfer result, not just as a notebook comment: the notebook's
MTCNN runs on `device="cuda"` (training ~500 videos, GPU is the entire reason
this step is in Colab), while `watchdog/cv_engine._load_winning_mtcnn()` — the
detector `evaluate_cv_transfer.py` actually calls for the Nigerian-corpus side
of the comparison — hardcodes `device="cpu"` per this project's CPU-only-
outside-Colab constraint.** Every other piece of the two preprocessing
pipelines (letterbox, `_CROP_MARGIN_FRAC` crop padding, the frame-sampling
formula, highest-confidence box selection, ImageNet normalization) was
checked line-by-line and matches exactly. But `facenet-pytorch`'s GPU and CPU
code paths aren't guaranteed bit-identical — floating-point non-associativity
through PNet/RNet/ONet means a face sitting right at MTCNN's confidence
threshold could in principle be detected on one device and missed on the
other. Low-probability in practice, but it is the one non-identical piece in
an otherwise matched pipeline being used to measure a domain-transfer gap, so
§3/§4 must state it as an explicit caveat next to the FF++-vs-Nigerian number,
not leave it as an implicit assumption of parity.

**CV classifier training — closed as a documented negative result, gate not
cleared after three scheduled runs (2026-08-11/12).** Three training schedules
were run against the same FF++ subset (80 identity groups total, ~64 train /
16 test after the identity-grouped split), diagnosing each result before
changing the next, per this file's own working-style convention (§6) rather
than retuning blindly toward 0.90:

| Run | Schedule | Held-out video accuracy | Diagnosis |
|---|---|---|---|
| 1 | Single-phase, 6 epochs, lr=1e-4 | **0.86** | Overfitting: train_acc 0.978 vs 0.86, 12-point gap, loss still falling at epoch 6 — high/improving train accuracy alongside a wide test gap is a regularization problem, not too few epochs. `4000 crops` also overstated real diversity (~10 near-duplicate crops/video from ~64 train identity groups). |
| 2 | Two-phase (3 head-only + 9 fine-tune, backbone unfrozen at lr=1e-5), internal train-only val split added, stronger augmentation (`RandomResizedCrop`+`RandomRotation` alongside existing flip/jitter) | **0.78** | Overcorrected into underfitting: gap closed (+0.12 → +0.06) but val_video_acc was still climbing at epoch 9 (0.845 → 0.845 → 0.869) and train_acc only reached 0.869 (vs run 1's 0.978) — not converged, rather than converged badly. |
| 3 | Same as run 2 but `FINETUNE_EPOCHS` 9→20, fine-tune lr 1e-5→5e-5 | **0.79** | Converged correctly this time: internal-val peaked 0.881 at epoch 8-9, then declined while train_acc rose to 0.972 — classic overfitting onset, correctly caught by best-checkpoint-by-val-video-accuracy selection, which kept the epoch-8/9 weights rather than the final epoch's. |

**Class weighting was deliberately never added** across any of the three runs
— the classifier's job is to be a *representative* Western-trained detector so
the FF++→Nigerian-corpus transfer gap (§2.3.4) is measurable; biasing it
toward this project's own NFR-01/NFR-03 false-positive preferences (which
govern the *deployed* pipeline, not this baseline) would have made it less
representative of the thing being measured. Any specificity/recall asymmetry
in the per-run results is a property of the baseline, not something corrected
for in the loss.

**The key finding is about the split, not the schedule.** Run 3's internal-val
video accuracy peaked at 0.881 while its *held-out* FF++ test accuracy came in
at 0.79 — a ~9-point gap between two disjoint identity-grouped splits drawn
from the same 100-group pool (80 identity groups after dedup — see section 3's
printed count), evaluated on the *same* trained weights. A gap that size
between two splits of the same underlying pool means run 1's 0.86 and run 3's
0.79 plausibly sit inside the same split-luck noise band, not on a real
schedule-quality gradient — no combination of epochs/lr/freezing tried here
can be reliably distinguished from which 20% of ~80 identity groups happened
to land in the test fold. **The binding constraint is identity-group count,
not hyperparameters.** This is exactly why the search was capped at three
runs, agreed in advance, rather than continued until some schedule happened to
land above 0.90 by the same noise that separates 0.86 from 0.79 — that outcome
would have been indistinguishable from p-hacking the split.

**Gate verdict: not cleared. No fourth run.** Per the agreement made before
run 3, this stops here. **No CV transfer number is reported.**
`evaluate_cv_transfer.py`'s `load_cv_classifier()` requires
`config.CV_MODEL_PATH` to exist; section 9 of the training notebook (the
checkpoint save) never runs when section 8's gate raises, so no checkpoint was
ever produced or downloaded into `models/` — confirmed: no `cv_watchdog_v1.pt`
present locally. Running `evaluate_cv_transfer.py` as-is would `sys.exit(1)` at
the missing-checkpoint check. The script itself is built and was reviewed
cell-by-cell in the entry above (identity-grouped split logic, gate handling,
the CUDA/CPU MTCNN asymmetry) — it is **deliberately left unrun against a
baseline that isn't credible**, not abandoned mid-build.

**Write-up framing:**
- **§1.7 limitation**: 80 FaceForensics++ training identity groups is
  insufficient to reliably fine-tune EfficientNet-B0 end-to-end above the
  0.90 in-domain credibility gate set in advance of any run. The in-domain
  baseline peaked at 0.86 across three schedules (single-phase, two-phase
  underfit, two-phase converged), and the ~9-point gap between two
  same-pool splits on the best-converged run shows the shortfall is
  dataset-scale-bound rather than a fixable training-procedure defect.
- **§5.5 further studies**: a larger identity-group count (the full FF++
  release, not the ~80-group subset staged here, or a second benchmark like
  the deferred Celeb-DF access request above) — or alternatively a
  lower-capacity/frozen-backbone approach less prone to the identity-count
  sensitivity seen here — would be needed before a CV domain-transfer claim
  (§2.3.4) is defensible. Do not read the 0.86-0.79 spread as evidence
  either for or against the transfer-degradation hypothesis; the in-domain
  number itself was never established as reliable enough to make that
  comparison meaningful.

**Days 21-22 — Flask admin dashboard build (2026-08-13, `dashboard/app.py` +
`templates/`), §3.13 Routing.** Gives the human-review step §3.13 already
requires (`reports.status: PENDING_REVIEW -> DISPATCHED`) an actual
interface — before this, a PENDING_REVIEW report was only reachable via the
raw PDF file or a direct DB query. Four routes: `/` (violations list),
`/violations/<id>` (full evidentiary detail), `/reports/<id>` (serves the
PDF), `/upload` (submit a manifest sample_id or live URL, run
`main_pipeline`, show the result).

- **The detail view reuses `complaint_generator`'s caveat constants
  verbatim, not a second paraphrased copy.** `CAVEAT_CV_HEURISTIC_RELIABILITY`
  and `CAVEAT_IP_FALSE_POSITIVE_MODES` are imported directly from
  `watchdog.complaint_generator` and rendered as-is in an amber caveat
  panel — the PDF a reviewer downloads and the web page they read it on
  structurally cannot drift apart, since there is only one copy of the text.
- **Abstention is never rendered as a blank cell or a 0.00.** Every
  per-modality and per-heuristic abstain reason already stored in
  `violations.evidence_json` (fusion.py's `VisualAssessment`/
  `TextAssessment`) is displayed as an amber "ABSTAINED" pill plus its
  stored reason string, verbatim — confirmed against the one real
  production violation with visual evidence (the `b1ce1815` MTN ad,
  `complaint_7d69408c-...pdf` — see the 2026-08-06 real-sample entry above):
  `blink_rate`/`emotional_rigidity`/`viseme_mismatch` all render their real
  cv_engine.py coverage-gate reasons ("longest contiguous face run (2.4s) is
  below the 10.0s/3.0s minimum..."), not a dash or a zero.
- **`evidence_json` can be `None`** (nullable column; pre-Day-16 rows or a
  parse failure) — the detail view still renders every top-level
  `violations` column (psynth_score, p_visual, p_text, statute_*) in that
  case, with an explicit "no detailed evidence_json stored" note rather than
  a blank or broken page.
- **NFR-02 is now actually measurable end-to-end, not just at the Stage 2
  CV sub-pipeline.** `/upload` times the real `main_pipeline.process_media()`/
  `process_manifest_sample()` call (detection + fusion + rules + PDF
  generation — the full NFR-02 scope) and appends one row per submission to
  `results/dashboard_pipeline_latency.csv`, the same append-only,
  header-once convention every other `results/*.csv` writer in this project
  uses. Deliberately excludes content acquisition (HTTP/yt-dlp) from the
  timed window — that's third-party network latency this project doesn't
  control, not part of the detection budget. `config.TEXT_LATENCY_BUDGET_S`
  (5) was added alongside this — the ≤5s NFR-02 text target has been in
  README.md's NFR table since Day 1 but had no named `config.py` constant
  until something actually needed to compare against it (CLAUDE.md: "ALL
  thresholds ... in config.py"). **No real timing rows exist yet** — the
  mechanism was verified end-to-end against the real Postgres DB (all three
  real violations currently in `violations` render correctly, real PDFs
  serve correctly) but no one has submitted content through `/upload` yet;
  README's NFR-02 rows still correctly read "NOT YET MEASURED" until that
  happens.
- **`dashboard/app.py`'s `create_app(engine=None, results_dir=None)`**
  takes both as optional overrides purely so `tests/test_dashboard.py` can
  inject an in-memory SQLite engine (same pattern as `tests/conftest.py`)
  and a `tmp_path` results directory — the second one specifically so
  running the test suite can never append fake rows into the real
  `results/dashboard_pipeline_latency.csv` (confirmed: the file did not
  exist in `results/` before or after a full test run). 21 new tests in
  `tests/test_dashboard.py`, all against the real route logic (only
  `main_pipeline`'s own entry points and `scraper.scrape_url()` are
  monkeypatched, so the suite never has to run real CV/NLP models over a
  web request).
- **Real production data confirmed the dashboard hadn't been told about a
  fourth violation this session's earlier entries didn't record**: three
  rows in `violations` as of this build (the known `b1ce1815` MTN ad plus
  two `DEEPFAKE_IMPERSONATION` rows), not the "currently one violation"
  assumed when this task was scoped — rendered honestly (no fabricated
  scale either direction), and worth reconciling against `data/manifest.csv`
  / this file's dataset-state section in a future pass rather than silently
  assumed to be stale.
- **NFR-02 first real numbers (2026-08-13), and a profiling pass on why
  text's warm figure looked wrong.** Four `/upload` submissions produced
  `results/dashboard_pipeline_latency.csv`'s first rows: video cold
  32.062s / warm 4.886s (**passes NFR-02 warm, 30s budget**); text cold
  36.491s (fails, expected — first-load cost) / "warm" **16.882s**, still
  3.4x over the 5s budget as measured. That text-warm figure didn't match
  what the code should do once genuinely warm, so it was profiled rather
  than taken at face value:
  - **Two independent, faithful reproductions of the exact call
    (`main_pipeline.process_media()`, the same manifest sample,
    `60ee644d-...`, that produced the dashboard rows) both measured a
    genuinely-warm run at ~2.0–2.3s** — comfortably inside the 5s budget,
    not 16.9s. One via `cProfile` over three back-to-back in-process calls,
    one via plain `time.perf_counter()`; both used an in-memory SQLite
    session (`tests/conftest.py`'s pattern) since Postgres wasn't reachable
    in this environment (`docker` isn't installed in this WSL2 shell) —
    Postgres access doesn't affect NLP/CV inference timing, so this doesn't
    weaken the result.
  - **All three heavy models really do cache correctly at module level**,
    confirmed directly, not assumed: `nlp_engine.classify_text()`
    16.581s cold → 0.206s/0.196s warm; `nlp_engine.compute_perplexity()`
    7.135s cold → 0.523s/0.444s warm; `ip_similarity._get_embedder()`
    likewise near-instant on repeat. Grepped the whole codebase for
    anything that could reset these module-level caches (`importlib.reload`,
    a stray `_CLASSIFIER_MODEL = None`, etc.) — nothing found.
  - **Answering the two things worth checking**: (1) yes,
    `nlp_engine.analyze_text()` (Day 3 perplexity + burstiness) is still
    called on every live text request — `fusion.assess_text()` runs it
    purely to populate `TextAssessment.heuristic_diagnostic` for
    transparency/logging (its own docstring: "never feeds into p_text's
    value"). Measured cost: ~0.9–1.0s per warm request, roughly 40-50% of
    the whole warm pipeline, entirely for a signal §4's own earlier
    evaluation already found doesn't discriminate cleanly on this corpus.
    Real, quantified, avoidable cost — not fixed here (asked to measure
    first, not optimise), but worth an explicit call before Days 23-25
    treats it as free. (2) No — the three heavy models are *not*
    reloading per warm request (previous bullet). But a **different,
    smaller caching bug was found while checking**:
    `watchdog/preprocessor.py`'s `preprocess_text()` calls
    `spacy.load("en_core_web_sm")` fresh on every single call — no
    module-level cache, unlike every other model loader in this codebase
    (`nlp_engine._load_perplexity_model()`/`_load_classifier()`,
    `ip_similarity._get_embedder()`, all of which use the documented
    lazy-singleton pattern). Measured directly: ~0.35–0.76s per call,
    every call, not just the first. Real and easily fixed the same way
    the other three loaders already are, but too small (well under 1s) to
    be the source of the 16.9s figure on its own.
  - **So what actually explains the 16.882s dashboard row?** Most likely:
    that submission wasn't actually warm. 16.882s sits suspiciously close
    to a cold `classify_text()` load alone (16.581s, measured standalone
    above) — consistent with the classifier cache having been reset
    between the 13:51:53 cold submission and the 14:15:43 "warm" one (24
    minutes apart), most plausibly because the Flask dev server process
    was restarted in between. Nothing in the code itself would explain a
    cache reset (previous bullet's grep), and `config.FLASK_DEBUG=False`
    (`.env`) rules out Werkzeug's auto-reloader silently doing this.
    **Not confirmed** — this environment can't reach the real dashboard
    process to check its actual uptime/restart history, only reproduce
    its code path in isolation.
  - **Action before trusting either number**: resubmit the same sample
    twice in immediate succession through one continuously-running
    dashboard process (no restart between the two) and log both. If the
    second submission lands near ~2s as this profiling predicts, NFR-02
    text passes warm and `README.md`'s row above should be corrected
    accordingly; if it reproduces ~16s even with no restart, that's a real
    finding this entry's analysis missed and needs a fresh look — either
    way, `README.md`'s NFR-02 text row currently reflects this
    uncertainty rather than asserting either outcome as settled.
- **NFR-02 text-warm resolved (2026-08-13, same day).** The action item
  above was run for real: three consecutive `/upload` submissions of the
  same sample through one continuously-running dashboard process, no
  restart — 32.584s cold, then 2.625s, then 1.560s, both warm runs
  comfortably inside the 5s budget. Confirms the profiling prediction
  (~2.0-2.3s) and settles the earlier 16.882s row as a post-restart cold
  classifier load, not a real per-request cost — `README.md`'s NFR-02 text
  row now states this as the settled result, not an open question.
  Two follow-ups came out of the same profiling pass, both closed the same
  day:
  - **Fixed**: `watchdog/preprocessor.py`'s `preprocess_text()` now caches
    its spaCy pipeline at module level via a new `_load_spacy_model()`
    (same lazy-singleton pattern as `nlp_engine`'s two loaders and
    `ip_similarity._get_embedder()`) instead of calling
    `spacy.load("en_core_web_sm")` fresh on every call. Re-profiling after
    the fix: warm full-pipeline runs dropped to **0.78s-0.96s** (from
    ~2.0-2.3s pre-fix), comfortably clearing the "under 1.5s" bar — the
    ~0.5-0.9s of the old per-call `spacy.load()` cost is gone from
    `fusion.assess_text()`'s warm path. `tests/test_preprocessor.py`'s
    `test_gracefully_handles_missing_spacy_model` regression test had to
    move with it: it previously monkeypatched `spacy.load` directly, which
    the new module-level cache would silently make a no-op (a real load
    earlier in the same test session leaves `_load_spacy_model()` never
    touching `spacy.load` again) — the test now patches
    `preprocessor._load_spacy_model` itself instead, the same idiom
    `tests/test_nlp_engine.py` already uses for
    `nlp_engine._load_perplexity_model`. Full suite still green (200/200)
    after the change.
  - **Decided, not removed**: `nlp_engine.analyze_text()` stays on the live
    text path despite costing ~0.5-1.0s of a now-~0.8-1.0s warm request —
    the single largest remaining line item — rather than being pruned as
    dead weight now that `p_text` doesn't use its output. Two reasons,
    written directly into `fusion.assess_text()`'s docstring so a future
    pass doesn't rediscover and delete this without the context: (1) it's
    the only place perplexity/burstiness run against real submitted
    content, which is what keeps the NFR-03 register-bias finding
    (DistilGPT-2 misreading real Nigerian journalism as AI-like)
    observable per-sample going forward, not just frozen in the one
    historical `nlp_heuristic_eval.csv` run; (2) its output lands in
    `evidence_json` on every violation via `heuristic_diagnostic`, so a
    human reviewer or opposing counsel can see the perplexity/burstiness
    signal was computed and what it showed, even though the system's own
    decision didn't rely on it — a forensic-completeness argument that
    outweighs ~1s against a 5s budget in a system whose output feeds legal
    complaints. If this budget ever needs reclaiming, the documented next
    move is speeding up `compute_perplexity()` itself (e.g. a word-count
    floor below which it's skipped as uninformative), not deleting the
    call outright.
- **Days 23-25 Chapter 4 evaluation artifacts (2026-08-13):
  `generate_chapter4_evaluation.py`, three outputs.** First, two stale
  README.md rows were corrected before anything else: NFR-01 had read "no
  classifier trained yet" since before Day 8-9 (the classifier has existed
  and been validated since 2026-08-04); NFR-03 showed only the 38.2%
  heuristic-vote FPR, not the fine-tuned classifier's own 8.8% (both now
  reported side by side — 8.8% is a real 4.3x improvement from
  localisation, but still above the 5% target, reported as such rather than
  rounded into a pass).
  - **`results/confusion_matrix_nlp_classifier.png`**: built from
    `results/nlp_classifier_cv_eval.csv`'s raw per-sample predictions (the
    same pooled group-CV data `nlp_classifier_fold_metrics.csv`'s AGGREGATE
    row summarises) — 31 TN / 3 FP / 5 FN / 26 TP, n=65. The generator
    recomputes precision/recall/F1/FPR from these raw counts and hard-fails
    if they disagree with the AGGREGATE row by more than float rounding,
    so this figure can't silently drift from the number already cited
    elsewhere.
  - **`results/register_bias_perplexity_by_genre.png`**: mean DistilGPT-2
    perplexity by `synthetic_genre` (joining `nlp_heuristic_eval.csv`'s
    perplexity column to `nlp_classifier_cv_eval.csv`'s genre tags by
    `sample_id` — both score the same 65-sample corpus). Confirms the
    9.4x Pidgin-vs-bureaucratic gap exactly (Pidgin genres pooled: mean
    218.2, n=4; bureaucratic: mean 23.2, n=4; ratio 9.39x) — a
    within-SYNTHETIC-class comparison, so the entire spread is attributable
    to register/dialect familiarity to DistilGPT-2's Western training
    distribution, not to authenticity (every sample compared here has the
    same true label). AUTHENTIC's own mean (42.2) is drawn as a reference
    line for scale, not as an extra bar — it isn't a genre. This is this
    project's strongest original NLP finding and had only ever existed as
    numbers in a CSV before this figure.
  - **`results/nfr_verification_table.csv`**: every NFR, target, measured
    result, pass/fail, and exact source file, in one place — condensed
    version now in README.md's NFR table too, so the two don't drift.
  - **NFR-01 framing — no fused system-level F1, and why that's correct,
    not a gap.** Reported as three separate rows instead of one
    over-claimed number: (a) NLP classifier F1 0.867 (the real, held-out,
    group-CV-validated result) as the measured detection performance for
    the only pathway with a reliable in-domain score; (b) the CV pathway
    as a documented negative result — best held-out video *accuracy* (not
    even F1) was 0.86 on FF++, a different benchmark from this project's
    own corpus, against a 0.90 gate agreed in advance and not cleared
    after three runs (§4's dedicated entry above); (c) an explicit
    "not computable" row for the fused figure, with the actual reason
    spelled out: `data/manifest.csv`'s `media_type` is video XOR text per
    row, so no sample in this corpus has ever had both modalities scored
    together — `fusion.compute_psynth()`'s two-modality branch
    (`alpha*p_visual + (1-alpha)*p_text`) has never once executed against
    real data. This is a stronger, more specific reason than "the CV
    pathway is unreliable so we can't average it in" — there is no
    dual-modality item in the corpus to average over even in principle,
    independent of CV's reliability. No better framing was found; this is
    what got implemented.

---

## 5. Roadmap status

28-day plan, seven phases. Days 1–3 complete.

- **Days 1–4 — Setup & data.** Environment, PostgreSQL schema with NFR-04/05
  enforcement, scraper, preprocessor, dataset manifest, collection + labelling CLI.
  *Mostly done; dataset expansion still outstanding.*
- **Days 5–9 — NLP engine.** Perplexity + burstiness (done, Day 3), IP similarity
  matrix via sentence-transformers (Copyright Act 2022 s.36(2), corrected from the
  wrong "s.108" reference — see the legal-correction entry below), DistilBERT
  fine-tuning (done, moved earlier — see the classifier-pivot entry above).
- **Days 10–15 — CV engine. Closed (2026-08-12) as a documented negative result,
  not outstanding work.** OpenCV frame extraction, MTCNN facial alignment, and the
  four Stage 4 heuristics (blink-rate, emotional-rigidity, viseme–phoneme sync,
  blending artifacts) were built and evaluated earlier (§4's full 32-clip
  evaluation — none discriminate cleanly, the direct trigger for the classifier
  pivot below). The EfficientNet-B0 classifier was then built, reviewed, and run
  to completion across three deliberately-capped training schedules (see the
  dedicated entry above, full three-run table) — none cleared the 0.90 FF++
  held-out credibility gate set in advance (0.86 / 0.78 / 0.79). Diagnosed as an
  identity-group-count ceiling (80 FF++ training identity groups), not a fixable
  training-procedure defect: a ~9-point gap between two same-pool identity-grouped
  splits on the best-converged run means the 0.86-0.79 spread is plausibly
  split-luck noise, not a real schedule-quality signal — see that entry for why a
  fourth run was deliberately not attempted. No CV transfer number is reported;
  `evaluate_cv_transfer.py` remains built and reviewed but unrun (no checkpoint
  ever cleared the gate to produce one). This is this project's §1.7 limitation +
  §5.5 further-studies item, not a gap to close before submission.
- **Days 16–20 — Fusion + auto-complaint. Done (2026-08-06) — Objective 3, the
  project's stated novel contribution, is now built end to end.** `watchdog/fusion.py`
  (§3.11 weighted fusion → `Psynth`, with the missing-modality contract described in
  the design-decision entry below), `watchdog/rules_engine.py` (§3.13 Rules-as-Code —
  the single `Psynth` gate for all three violation types, including the deliberate
  text-without-IP-match non-escalation case), `watchdog/complaint_generator.py`
  (§3.13 ReportLab PDF, always-present human-review caveat + two conditional
  caveats), `main_pipeline.py` (orchestrator), and a `violations.evidence_json`
  schema addition for the audit trail. See the dedicated design-decision and
  real-sample entries below for the full reasoning and what running it against
  real data actually found. CV's own trained classifier (EfficientNet-B0, the
  Days 10-15 CV_MODEL_PATH item) did not clear its credibility gate (see above) —
  `fusion.combine_cv_heuristics()` runs on the existing Stage 4 heuristics only,
  which is exactly why it carries the CV-heuristic-reliability caveat, and will
  continue to for the rest of this project's timeline.
- **Days 21–22 — Flask admin dashboard. Done (2026-08-13).** `dashboard/app.py` +
  `templates/` — see the dedicated entry above for the four routes, the
  caveat-reuse/abstention-rendering decisions, and the NFR-02 latency-logging
  addition (`config.TEXT_LATENCY_BUDGET_S`,
  `results/dashboard_pipeline_latency.csv`). 21 new tests
  (`tests/test_dashboard.py`); verified against the real Postgres DB, not
  just the SQLite test suite.
- **Days 23–25 — Evaluation. NFR-02 latency and the three Chapter 4 artifacts
  done (2026-08-13); Chapter 4 screenshots still outstanding.**
  `results/dashboard_pipeline_latency.csv` now holds real submissions (the
  NFR-02 entries above), and `generate_chapter4_evaluation.py` produces the
  confusion matrix, the register-bias figure, and a consolidated
  `results/nfr_verification_table.csv` — see the dedicated entry below for
  what each contains and the NFR-01 framing decision. Still open: the actual
  Chapter 4 screenshots of the dashboard UI itself (not a data artifact,
  needs a real browser session against the running dashboard).
- **Days 26–28 — Write-up.** Chapter 4 (design + results) and Chapter 5 (summary,
  conclusions, recommendations, contributions, further studies).

**Next up: Days 23–25's remaining item — Chapter 4 dashboard screenshots —
then Days 26–28 — Write-up.** Days 21-22 give a human reviewer somewhere to
actually see and act on `PENDING_REVIEW` reports, Days 10-15/16-20 are both
closed (one as a built success, one as a documented negative result), and
Days 23-25's data-artifact work (NFR-02 real measurements, confusion matrix,
register-bias figure, NFR verification table) is done as of 2026-08-13. (This
line previously said "Day 4 — CV/deepfake engine", then "Days 21-22 — Flask
admin dashboard", then "Days 23-25 — Evaluation" with no work done yet on it
— each stale by real completed work; corrected again 2026-08-13 rather than
left misleading the next person to read this file.)

---

## 6. Working style that has worked well

- **Test against real sources early.** Nearly every bug above came from real data,
  not synthetic fixtures. Format mismatches, filesystem quirks, cookie parsing, FK
  ordering — none would have surfaced otherwise.
- **Verify before claiming.** Where a fix was made, it was reproduced first
  (the failing case simulated), then fixed, then re-verified. Bug #2 exists
  specifically because a test suite passed while production broke.
- **Say when something is uncertain.** Several dataset entries carry
  `expected_label=None` with "NEEDS MANUAL VERIFICATION" precisely because guessing
  a label would silently corrupt ground truth.
