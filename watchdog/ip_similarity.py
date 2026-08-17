# =============================================================================
# watchdog/ip_similarity.py  —  Day 7: IP Similarity Matrix (§3.12 B)
# =============================================================================
# PURPOSE:
#   Last unbuilt piece of Objective 2 (§2 — protecting Nigerian creators' IP
#   from AI-plagiarism). Embeds a "protected corpus" of known-authentic,
#   verified human-authored works once via sentence-transformers, then scores
#   arbitrary input text against it by cosine similarity — the same signal
#   the complaint generator (§3.13) needs to cite Copyright Act 2022 s.36(2)
#   ("reproduction of the whole or a substantial part of the work ... in any
#   form recognisably derived from the original") against a NAMED work, not
#   just a bare similarity number. See docs/PROJECT_CONTEXT.md's 2026-08-04
#   entry for the correction from the write-up's wrong "s.108" citation
#   (s.108 is the Interpretation section — definitions only, no cause of
#   action) to s.36(2) (infringement) / s.37(2) (remedies).
#
# THE PROTECTED CORPUS, AND ITS LIMITATION:
#   Reuses the 34 already-collected, human-reviewed AUTHENTIC TEXTUAL_CORPUS
#   samples (Punch, Premium Times, Vanguard, BusinessDay, TheCable,
#   LEADERSHIP) as the protected-works index — real, bylined, human-authored
#   Nigerian journalism, exactly what Objective 2 exists to protect, and
#   already verified rather than freshly scraped for this purpose.
#   build_protected_corpus_index() below does NOT copy those raw text files
#   into data/protected_ip/ — the manifest's local_path already points at
#   them under data/raw/text/, and duplicating them would create exactly the
#   "files on disk without manifest rows" bug pattern CLAUDE.md rule 4 warns
#   about. Instead it writes a lightweight, re-buildable INDEX (sample_id +
#   provenance, no file copies) to data/protected_ip/protected_corpus_index.csv
#   — an auditable record of exactly which works the matcher currently checks
#   against, without a second copy of the data to drift out of sync.
#
#   LIMITATION (§1.7): this is a small (n=34), journalism-only corpus. A real
#   deployment protecting Nigerian creators per §1.1 would also index
#   Nollywood scripts, Afrobeats lyrics, and other creative works — none of
#   which this project has collected. Every similarity score below is
#   therefore only ever "how much does this resemble one of 34 news
#   articles," not "how much does this resemble Nigerian creative IP in
#   general."
#
# WHOLE-DOCUMENT SCORING, SENTENCE-LEVEL CITATION:
#   The THRESHOLD DECISION (flagged or not) is whole-document embedding
#   cosine similarity — input text vs. each protected work's full text. This
#   keeps the decision consistent with how it's validated (leave-one-out and
#   paraphrase testing both operate on whole documents — see
#   validate_ip_similarity_threshold.py). But "a legal document must name the
#   work allegedly infringed, not just report a number" needs something more
#   specific than "these two 800-word articles are 0.79 similar" — so each
#   protected work is ALSO split into sentences (reusing
#   preprocessor.preprocess_text()) and sentence-embedded once at index-build
#   time. When reporting a match, the single sentence within that matched
#   document closest to the input text is surfaced as `matched_passage` — a
#   quotable excerpt a complaint can point at, not just a filename. This is
#   still coarser than true passage-alignment (it doesn't find the best
#   INPUT sentence too, just the best PROTECTED sentence against the whole
#   input) — a real limitation, not hidden: see the module-level note in
#   check_similarity()'s docstring.
# =============================================================================

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import IP_EMBEDDINGS_PATH, IP_SIMILARITY_THRESHOLD, MANIFEST_PATH, PROTECTED_IP_DIR

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2: 384-dim, ~80MB, the standard CPU-viable sentence-transformers
# choice (§1.7 CPU-only constraint) — fast enough to embed a few dozen documents
# and their sentences in seconds, not minutes.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Below this word count, a document embedding is mostly noise (see
# nlp_engine.compute_burstiness()'s analogous "< 2 sentences" abstention) —
# same conservative posture as the NLP/CV engines: abstain rather than
# fabricate a confident similarity score from too little text (NFR-01).
MIN_WORDS_TO_EMBED = 20

DEFAULT_TOP_K = 3

PROTECTED_CORPUS_INDEX_PATH = PROTECTED_IP_DIR / "protected_corpus_index.csv"


# =============================================================================
# Protected corpus index — provenance only, no file copies (see module docstring)
# =============================================================================

def build_protected_corpus_index(path: Path = PROTECTED_CORPUS_INDEX_PATH) -> Path:
    """
    Writes data/protected_ip/protected_corpus_index.csv from the manifest's
    AUTHENTIC TEXTUAL_CORPUS rows — a provenance record of which existing
    samples make up the protected-works corpus, not a copy of their text.

    Idempotent (CLAUDE.md rule 2): always a full rewrite from the current
    manifest state, never an append — re-running after the manifest changes
    (e.g. more authentic samples added) picks up the new state cleanly
    instead of accumulating stale/duplicate rows.
    """
    import pandas as pd
    from watchdog.dataset_manager import CATEGORY_TEXTUAL_CORPUS, LABEL_AUTHENTIC, require_manifest_exists

    require_manifest_exists(MANIFEST_PATH)

    df = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    authentic = df[
        (df["category"] == CATEGORY_TEXTUAL_CORPUS) & (df["label"] == str(LABEL_AUTHENTIC))
    ].copy()

    rows = []
    for _, row in authentic.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            logger.warning(f"Protected-corpus candidate missing on disk, skipping: {local_path}")
            continue
        rows.append({
            "sample_id": row["sample_id"],
            "local_path": row["local_path"],
            "publisher_name": row["publisher_name"],
            "source_url": row["source_url"],
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    index_df = pd.DataFrame(rows, columns=["sample_id", "local_path", "publisher_name", "source_url"])
    index_df.to_csv(path, index=False)
    logger.info(f"Wrote protected-corpus index: {len(index_df)} works -> {path}")
    return path


def _load_protected_works(index_path: Path = PROTECTED_CORPUS_INDEX_PATH) -> list[dict]:
    """Loads the index and the actual text each row points at (still on disk under data/raw/text/)."""
    import pandas as pd

    if not index_path.exists():
        return []

    index_df = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    works = []
    for _, row in index_df.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            logger.warning(f"Protected work missing on disk (index is stale?), skipping: {local_path}")
            continue
        text = local_path.read_text(encoding="utf-8", errors="replace")
        works.append({
            "sample_id": row["sample_id"],
            "publisher_name": row["publisher_name"],
            "source_url": row["source_url"],
            "text": text,
        })
    return works


# =============================================================================
# Embedding model (lazy singleton, same pattern as nlp_engine._load_perplexity_model)
# =============================================================================

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading {EMBEDDING_MODEL_NAME} for IP similarity (first call only)...")
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"{EMBEDDING_MODEL_NAME} loaded.")
    return _EMBEDDER


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (d,), b: (n, d), both assumed already L2-normalised -> (n,) cosine similarities."""
    return b @ a


# =============================================================================
# Embeddings index — build once, cache to config.IP_EMBEDDINGS_PATH
# =============================================================================

def build_embeddings_index(force: bool = False) -> Optional[dict]:
    """
    Builds (or loads the cache for) the protected corpus's document- and
    sentence-level embeddings. Rebuilds the provenance index first if it
    doesn't exist yet. Idempotent: skips recomputation if
    config.IP_EMBEDDINGS_PATH already exists and force=False — sentence-
    transformer inference over the whole corpus is real, avoidable cost.

    Returns the index dict (also cached to disk), or None if the protected
    corpus is empty (nothing to build).
    """
    if IP_EMBEDDINGS_PATH.exists() and not force:
        logger.info(f"Embeddings index already cached at {IP_EMBEDDINGS_PATH} — loading, not rebuilding.")
        return load_embeddings_index()

    if not PROTECTED_CORPUS_INDEX_PATH.exists():
        build_protected_corpus_index()

    works = _load_protected_works()
    if not works:
        logger.warning("Protected corpus is empty — nothing to embed.")
        return None

    from watchdog.preprocessor import preprocess_text

    embedder = _get_embedder()

    t0 = time.perf_counter()
    doc_texts = [w["text"] for w in works]
    doc_embeddings = embedder.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)

    sentence_lists = []
    sentence_embeddings_list = []
    for w in works:
        sentences = [s for s in preprocess_text(w["text"])["sentences"] if s.strip()]
        if not sentences:
            sentences = [w["text"][:200]]  # degenerate fallback: no sentence split found
        sentence_lists.append(sentences)
        sentence_embeddings_list.append(
            embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        )
    build_time_s = time.perf_counter() - t0

    index = {
        "model_name": EMBEDDING_MODEL_NAME,
        "sample_ids": [w["sample_id"] for w in works],
        "publisher_names": [w["publisher_name"] for w in works],
        "source_urls": [w["source_url"] for w in works],
        "doc_embeddings": np.asarray(doc_embeddings, dtype=np.float32),
        "sentences": sentence_lists,
        "sentence_embeddings": sentence_embeddings_list,
        "build_time_s": build_time_s,
    }

    IP_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(IP_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"Built embeddings index: {len(works)} works, {build_time_s:.1f}s -> {IP_EMBEDDINGS_PATH}")

    return index


def load_embeddings_index() -> Optional[dict]:
    """Loads the cached index from config.IP_EMBEDDINGS_PATH, or None if it hasn't been built yet."""
    if not IP_EMBEDDINGS_PATH.exists():
        return None
    with open(IP_EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Similarity check — the public entry point
# =============================================================================

@dataclass
class SimilarityMatch:
    sample_id: str
    publisher_name: str
    source_url: str
    similarity: float               # [-1, 1], cosine similarity of whole-document embeddings
    matched_passage: str             # closest single sentence in the matched work — for citation


@dataclass
class IPSimilarityResult:
    matches: list = field(default_factory=list)   # list[SimilarityMatch], ranked, len <= top_k
    flagged: bool = False                           # True if matches[0].similarity >= IP_SIMILARITY_THRESHOLD
    abstain_reason: Optional[str] = None
    time_s: float = 0.0


def check_similarity(text: str, top_k: int = DEFAULT_TOP_K, index: Optional[dict] = None) -> IPSimilarityResult:
    """
    Embeds `text` and returns the top-k most similar protected works by
    whole-document cosine similarity, each with enough provenance
    (publisher, source URL, a quoted matched sentence) for
    watchdog.complaint_generator (§3.13, not yet built) to name a specific
    allegedly-infringed work rather than report a bare score.

    Abstains (empty matches, flagged=False, abstain_reason set) — same
    conservative posture as nlp_engine.analyze_text()'s UNCERTAIN vote and
    cv_engine's HeuristicResult — when:
      - the protected corpus index hasn't been built / is empty, or
      - `text` is too short to embed meaningfully (< MIN_WORDS_TO_EMBED words).
    Never raises for either case.

    `index`: pass a pre-loaded index (e.g. for a batch of calls) to skip
    reloading from disk each time; defaults to loading the cache.
    """
    t0 = time.perf_counter()

    word_count = len((text or "").split())
    if word_count < MIN_WORDS_TO_EMBED:
        return IPSimilarityResult(
            abstain_reason=(f"input is {word_count} words, below the {MIN_WORDS_TO_EMBED}-word "
                             f"minimum for a meaningful embedding"),
            time_s=time.perf_counter() - t0,
        )

    if index is None:
        index = load_embeddings_index()
    if index is None or len(index["sample_ids"]) == 0:
        return IPSimilarityResult(
            abstain_reason="protected corpus index is empty or not built — run build_embeddings_index() first",
            time_s=time.perf_counter() - t0,
        )

    embedder = _get_embedder()
    query_embedding = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

    doc_sims = _cosine_similarity(query_embedding, index["doc_embeddings"])
    ranked = np.argsort(-doc_sims)[:top_k]

    matches = []
    for i in ranked:
        i = int(i)
        sent_embeddings = index["sentence_embeddings"][i]
        sent_sims = _cosine_similarity(query_embedding, sent_embeddings)
        best_sent_idx = int(np.argmax(sent_sims))
        matches.append(SimilarityMatch(
            sample_id=index["sample_ids"][i],
            publisher_name=index["publisher_names"][i],
            source_url=index["source_urls"][i],
            similarity=float(doc_sims[i]),
            matched_passage=index["sentences"][i][best_sent_idx],
        ))

    flagged = bool(matches) and matches[0].similarity >= IP_SIMILARITY_THRESHOLD

    return IPSimilarityResult(
        matches=matches, flagged=flagged, abstain_reason=None, time_s=time.perf_counter() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    build_protected_corpus_index()
    build_embeddings_index(force=True)
