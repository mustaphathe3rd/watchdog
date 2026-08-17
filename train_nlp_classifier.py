# =============================================================================
# train_nlp_classifier.py  —  Day 8-9 Deliverable: Fine-Tuned DistilBERT
# =============================================================================
# PURPOSE:
#   Fine-tunes config.NLP_BASE_MODEL (distilbert-base-uncased) for binary
#   authentic/synthetic classification on the real, labelled TEXTUAL_CORPUS
#   (65 samples as of this writing). This is the classification half of the
#   hybrid design nlp_engine.py's module docstring describes: DistilGPT-2
#   supplies perplexity as a statistical feature (Day 3), DistilBERT is the
#   learned classifier (Day 8-9) reserved for exactly this.
#
# WHY STRATIFIED *GROUP* K-FOLD, NOT A SINGLE TRAIN/TEST SPLIT:
#   At n=65, a single 80/20 split's 13-sample test set makes any accuracy
#   figure noisy enough to be nearly meaningless (one flipped prediction
#   moves it ~8 points). 5-fold CV means every sample is scored exactly once
#   as held-out, giving both a per-fold spread AND a pooled aggregate over
#   all 65 out-of-fold predictions.
#
#   Plain StratifiedKFold isn't enough on its own: samples aren't
#   independent draws. The 31 synthetic samples came from 10 distinct
#   generation batches (5 news + 5 ad_copy under the original
#   generate_synthetic_corpus.py run, tagged "claude-sonnet-5" in
#   source_url; 21 more across ad_copy/bureaucratic/long_news/news/opinion/
#   pidgin_ad/pidgin_news/pidgin_social under expand_synthetic_corpus.py,
#   tagged "claude") — each batch was hand-authored together in one pass, so
#   samples within a batch share surface-level phrasing/structural patterns
#   a model could latch onto as a shortcut. The 34 authentic samples group
#   naturally by publisher (7 publishers) for the same reason — one outlet's
#   house style is a stronger shared signal than "authentic Nigerian news"
#   in general. If a batch or publisher straddled train/test, the model
#   could score well by recognising the SOURCE rather than the
#   authentic-vs-synthetic distinction this project actually cares about.
#   StratifiedGroupKFold keeps every group intact within a single fold while
#   still trying to balance the label ratio across folds.
#
# EXPECTATIONS GOING IN, PER THE PIVOT DECISION:
#   Both heuristic pathways (NLP: results/nlp_heuristic_eval.csv, CV:
#   results/cv_heuristic_eval.csv) return UNCERTAIN/no-separation often
#   enough that layering a fusion module on top of them first wasn't worth
#   it — this script is Day 8-9 moved earlier specifically to find out
#   whether a trained classifier does better than the heuristics before any
#   more heuristic-layer work happens.
#
# USAGE:
#   python train_nlp_classifier.py
#
# OUTPUT:
#   - Per-fold progress + metrics printed to the terminal
#   - results/nlp_classifier_cv_eval.csv     — per-sample out-of-fold predictions
#   - results/nlp_classifier_fold_metrics.csv — per-fold + aggregate metrics
#   - config.NLP_MODEL_PATH                   — final model, fit on ALL 65
#     labelled samples after the CV loop (standard practice: k-fold estimates
#     honest generalisation error, then one more fit on everything available
#     produces the actual deployment artifact — its own performance is NOT
#     independently validated beyond the CV estimate above it).
#
# NOTE ON RUNTIME:
#   CPU-only per CLAUDE.md (no local CUDA). Real per-step timing is measured
#   and reported, not assumed — see the "Try CPU first" instruction this
#   script follows.
# =============================================================================

import logging
import os
import sys
import time
from pathlib import Path

# Model is expected to already be cached locally (confirmed present in this
# environment) — HF_HUB_OFFLINE skips the network HEAD-check every one of
# the 6 from_pretrained() calls below (5 folds + 1 final fit) would otherwise
# make, which on this WSL2 connection was observed adding well over a
# minute of pure network latency per call. If the model is NOT cached yet,
# unset this (or run once without it) so the first download can happen.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_SPLITS = 5
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

PIDGIN_GENRES = {"pidgin_ad", "pidgin_social", "pidgin_news"}


def _parse_synthetic_batch(source_url: str) -> tuple[str, str]:
    """
    source_url for LLM-generated samples is 'llm-generated:MODEL:GENRE:filename'
    (see generate_synthetic_corpus.py / expand_synthetic_corpus.py). Returns
    (model_tag, genre) — model_tag distinguishes the original 10-sample batch
    ("claude-sonnet-5") from the 21-sample expansion ("claude").
    """
    parts = source_url.split(":")
    model_tag = parts[1] if len(parts) > 1 else "unknown"
    genre = parts[2] if len(parts) > 2 else "unknown"
    return model_tag, genre


def _build_group(row) -> str:
    """
    Authentic samples group by publisher (house style is the shared,
    non-label signal to keep out of both train and test within one fold).
    Synthetic samples group by (model_tag, genre) — i.e. by generation
    batch, per the module docstring above.
    """
    from watchdog.dataset_manager import LABEL_AUTHENTIC

    if int(row["label"]) == LABEL_AUTHENTIC:
        return f"authentic:{row['publisher_name']}"
    model_tag, genre = _parse_synthetic_batch(row["source_url"])
    return f"synthetic:{model_tag}:{genre}"


def _load_labelled_corpus():
    import pandas as pd
    from config import MANIFEST_PATH
    from watchdog.dataset_manager import (
        CATEGORY_TEXTUAL_CORPUS, LABEL_AUTHENTIC, LABEL_SYNTHETIC, require_manifest_exists,
    )

    require_manifest_exists(MANIFEST_PATH)

    df = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    text_rows = df[
        (df["category"] == CATEGORY_TEXTUAL_CORPUS)
        & (df["label"].isin([str(LABEL_AUTHENTIC), str(LABEL_SYNTHETIC)]))
    ].copy()

    texts, labels, groups, sample_ids, genres, publishers = [], [], [], [], [], []
    for _, row in text_rows.iterrows():
        local_path = Path(row["local_path"])
        if not local_path.exists():
            logger.warning(f"File missing on disk, skipping: {local_path}")
            continue
        texts.append(local_path.read_text(encoding="utf-8", errors="replace"))
        label = int(row["label"])
        labels.append(label)
        groups.append(_build_group(row))
        sample_ids.append(row["sample_id"][:8])
        if label == LABEL_SYNTHETIC:
            _, genre = _parse_synthetic_batch(row["source_url"])
            genres.append(genre)
            publishers.append("")
        else:
            genres.append("")
            publishers.append(row["publisher_name"])

    return texts, labels, groups, sample_ids, genres, publishers


class _TextDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _train_one_fold(train_texts, train_labels, tokenizer, epochs=EPOCHS):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
    from config import NLP_BASE_MODEL

    model = AutoModelForSequenceClassification.from_pretrained(NLP_BASE_MODEL, num_labels=2)
    model.train()

    encodings = tokenizer(train_texts, truncation=True, max_length=MAX_LENGTH)
    dataset = [
        {**{k: v[i] for k, v in encodings.items()}, "labels": train_labels[i]}
        for i in range(len(train_labels))
    ]
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_steps = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            labels_batch = batch.pop("labels")
            optimizer.zero_grad()
            outputs = model(**batch, labels=labels_batch)
            outputs.loss.backward()
            optimizer.step()
            epoch_loss += outputs.loss.item()
            n_steps += 1
        logger.info(f"    epoch {epoch + 1}/{epochs}  mean_loss={epoch_loss / max(1, len(loader)):.4f}")
    train_time_s = time.perf_counter() - t0

    return model, train_time_s, n_steps


def _predict(model, texts, tokenizer):
    import torch
    from transformers import DataCollatorWithPadding

    model.eval()
    encodings = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    batch = collator([{k: v[i] for k, v in encodings.items()} for i in range(len(texts))])

    with torch.no_grad():
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(synthetic)
        preds = torch.argmax(logits, dim=-1)

    return preds.tolist(), probs.tolist()


def _fold_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true_arr = y_true
    tn = sum(1 for t, p in zip(y_true_arr, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true_arr, y_pred) if t == 0 and p == 1)
    n_authentic = tn + fp
    fpr_authentic = fp / n_authentic if n_authentic else float("nan")

    return {
        "precision": precision_score(y_true_arr, y_pred, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred, zero_division=0),
        "fpr_authentic": fpr_authentic,
        "n_authentic": n_authentic,
        "n_fp_authentic": fp,
    }


def run_training():
    from sklearn.model_selection import StratifiedGroupKFold
    from transformers import AutoTokenizer
    import pandas as pd
    from config import NLP_BASE_MODEL, NLP_MODEL_PATH

    texts, labels, groups, sample_ids, genres, publishers = _load_labelled_corpus()
    n = len(texts)
    logger.info(f"Loaded {n} labelled TEXTUAL_CORPUS samples "
                f"({sum(1 for l in labels if l == 0)} authentic, {sum(1 for l in labels if l == 1)} synthetic).")

    unique_groups = sorted(set(groups))
    logger.info(f"{len(unique_groups)} distinct groups: {unique_groups}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(NLP_BASE_MODEL)

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    per_sample_pred = [None] * n
    per_sample_prob = [None] * n
    fold_rows = []
    overall_t0 = time.perf_counter()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels, groups)):
        fold_train_groups = sorted(set(groups[i] for i in train_idx))
        fold_test_groups = sorted(set(groups[i] for i in test_idx))
        assert not (set(fold_train_groups) & set(fold_test_groups)), "group leaked across train/test!"

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        logger.info(
            f"Fold {fold_idx + 1}/{N_SPLITS}: train={len(train_idx)} "
            f"(auth={sum(1 for l in train_labels if l == 0)}/synth={sum(1 for l in train_labels if l == 1)}), "
            f"test={len(test_idx)} "
            f"(auth={sum(1 for l in test_labels if l == 0)}/synth={sum(1 for l in test_labels if l == 1)})"
        )
        logger.info(f"    test groups: {fold_test_groups}")

        model, train_time_s, n_steps = _train_one_fold(train_texts, train_labels, tokenizer)
        preds, probs = _predict(model, test_texts, tokenizer)

        for local_i, global_i in enumerate(test_idx):
            per_sample_pred[global_i] = preds[local_i]
            per_sample_prob[global_i] = probs[local_i]

        metrics = _fold_metrics(test_labels, preds)
        metrics.update({
            "fold": fold_idx + 1, "n_train": len(train_idx), "n_test": len(test_idx),
            "train_time_s": round(train_time_s, 1), "n_steps": n_steps,
            "test_groups": ";".join(fold_test_groups),
        })
        fold_rows.append(metrics)
        logger.info(
            f"    fold {fold_idx + 1} done in {train_time_s:.1f}s — "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} FPR(authentic)={metrics['fpr_authentic']:.3f} "
            f"({metrics['n_fp_authentic']}/{metrics['n_authentic']})"
        )
        del model
        print()

    total_cv_time_s = time.perf_counter() - overall_t0

    assert all(p is not None for p in per_sample_pred), "every sample must get exactly one out-of-fold prediction"

    # ---- per-sample out-of-fold predictions ----
    per_sample_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": ["AUTHENTIC" if l == 0 else "SYNTHETIC" for l in labels],
        "publisher": publishers,
        "synthetic_genre": genres,
        "group": groups,
        "predicted_label": ["AUTHENTIC" if p == 0 else "SYNTHETIC" for p in per_sample_pred],
        "prob_synthetic": per_sample_prob,
        "correct": [t == p for t, p in zip(labels, per_sample_pred)],
    })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    per_sample_path = output_dir / "nlp_classifier_cv_eval.csv"
    per_sample_df.to_csv(per_sample_path, index=False)

    # ---- aggregate (pooled out-of-fold) metrics ----
    aggregate = _fold_metrics(labels, per_sample_pred)
    aggregate.update({
        "fold": "AGGREGATE", "n_train": "-", "n_test": n,
        "train_time_s": round(total_cv_time_s, 1), "n_steps": sum(r["n_steps"] for r in fold_rows),
        "test_groups": "-",
    })
    fold_df = pd.DataFrame(fold_rows + [aggregate])
    fold_path = output_dir / "nlp_classifier_fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)

    # ---- report ----
    print("=" * 90)
    print("  PER-FOLD METRICS")
    print("=" * 90)
    print(fold_df[["fold", "n_train", "n_test", "precision", "recall", "f1",
                    "fpr_authentic", "train_time_s"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("  COMPARISON — Fine-tuned DistilBERT (aggregate, pooled out-of-fold) vs Day 3 heuristic baseline")
    print("=" * 90)
    baseline_path = output_dir / "nlp_heuristic_eval.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline_votable = baseline_df[baseline_df["heuristic_vote"] != "UNCERTAIN"]
        b_auth = baseline_df[baseline_df["true_label"] == "AUTHENTIC"]
        b_fpr = (b_auth["heuristic_vote"] == "AI").mean()
        b_recall = (
            baseline_df[baseline_df["true_label"] == "SYNTHETIC"]["heuristic_vote"] == "AI"
        ).mean()
        print(f"  Day 3 heuristic baseline  (n={len(baseline_df)}, "
              f"{len(baseline_votable)} non-UNCERTAIN votes):")
        print(f"    FPR on authentic (NFR-03 target metric): {b_fpr:.3f} ({int((b_auth['heuristic_vote']=='AI').sum())}/{len(b_auth)})")
        print(f"    Recall on synthetic (vote=AI):            {b_recall:.3f}")
        print(f"    UNCERTAIN rate:                           {(baseline_df['heuristic_vote']=='UNCERTAIN').mean():.3f}")
    else:
        print(f"  (no baseline found at {baseline_path} — run evaluate_nlp_heuristics.py first)")

    print(f"\n  Fine-tuned DistilBERT (n={n}, 5-fold CV, pooled out-of-fold predictions):")
    print(f"    FPR on authentic (NFR-03 target metric): {aggregate['fpr_authentic']:.3f} "
          f"({aggregate['n_fp_authentic']}/{aggregate['n_authentic']})")
    print(f"    Precision: {aggregate['precision']:.3f}   Recall: {aggregate['recall']:.3f}   F1: {aggregate['f1']:.3f}")
    print(f"    No UNCERTAIN option — the classifier always makes a call, unlike the heuristic vote.")

    # ---- register effect (Pidgin) check ----
    print("\n" + "=" * 90)
    print("  REGISTER EFFECT — does the Pidgin anomaly persist after fine-tuning?")
    print("=" * 90)
    synth_df = per_sample_df[per_sample_df["true_label"] == "SYNTHETIC"]
    pidgin_df = synth_df[synth_df["synthetic_genre"].isin(PIDGIN_GENRES)]
    non_pidgin_df = synth_df[~synth_df["synthetic_genre"].isin(PIDGIN_GENRES)]
    if len(pidgin_df) > 0:
        print(f"  Pidgin-genre synthetic samples (n={len(pidgin_df)}: "
              f"{pidgin_df['synthetic_genre'].tolist()}):")
        print(f"    mean P(synthetic) = {pidgin_df['prob_synthetic'].mean():.3f}  "
              f"(individual: {[round(p, 3) for p in pidgin_df['prob_synthetic']]})")
        print(f"    correctly classified: {pidgin_df['correct'].sum()}/{len(pidgin_df)}")
        print(f"  Non-Pidgin synthetic samples (n={len(non_pidgin_df)}):")
        print(f"    mean P(synthetic) = {non_pidgin_df['prob_synthetic'].mean():.3f}")
        print(f"    correctly classified: {non_pidgin_df['correct'].sum()}/{len(non_pidgin_df)}")
        diff = pidgin_df["prob_synthetic"].mean() - non_pidgin_df["prob_synthetic"].mean()
        print(f"  Difference (Pidgin mean - non-Pidgin mean): {diff:+.3f}")
        print(
            "  Day 3's NLP heuristic finding was that perplexity ran BACKWARDS on Nigerian-register\n"
            "  text (synthetic scored MORE surprising to DistilGPT-2, plausibly because Pidgin/Nigerian\n"
            "  phrasing is out-of-distribution for a Western-English-trained model regardless of author).\n"
            "  Check above whether the fine-tuned classifier still under-detects Pidgin synthetic samples\n"
            "  specifically (lower P(synthetic), more misclassifications) — if so, the same register\n"
            "  confound Day 3 found in the statistical heuristics has NOT been resolved by fine-tuning,\n"
            "  just possibly hidden inside one aggregate accuracy number."
        )
    else:
        print("  No Pidgin-genre synthetic samples found in the loaded corpus.")

    # ---- caveats that must not be buried ----
    print("\n" + "=" * 90)
    print("  CAVEATS")
    print("=" * 90)
    n_word_authentic = [len(t.split()) for t, l in zip(texts, labels) if l == 0]
    n_word_synthetic = [len(t.split()) for t, l in zip(texts, labels) if l == 1]
    import statistics
    print(
        f"  1. SINGLE-LLM SYNTHETIC SOURCE: all {len(n_word_synthetic)} synthetic samples were generated by\n"
        f"     one LLM family (Claude — source_url prefix 'llm-generated:claude' / 'llm-generated:claude-sonnet-5').\n"
        f"     Whatever the classifier has learned to detect is Claude's writing style specifically, not\n"
        f"     'AI-generated text' in general — it has never seen GPT, Gemini, Llama, or any other model's\n"
        f"     output. Generalisation to other LLMs' synthetic text is UNTESTED and should not be assumed.\n"
    )
    print(
        f"  2. LENGTH CONFOUND: authentic mean {statistics.mean(n_word_authentic):.0f} words "
        f"(range {min(n_word_authentic)}-{max(n_word_authentic)}) vs synthetic mean "
        f"{statistics.mean(n_word_synthetic):.0f} words (range {min(n_word_synthetic)}-{max(n_word_synthetic)}).\n"
        f"     This is the same confound flagged in docs/PROJECT_CONTEXT.md §4 for the Day 3 heuristics,\n"
        f"     now at n=65 instead of n=14 — there is real overlap this time (unlike Day 3's zero-overlap\n"
        f"     case), but the means are still very different. A classifier can score well here partly or\n"
        f"     wholly by learning 'longer text -> authentic' as a shortcut, which would NOT generalise to\n"
        f"     a longer LLM-generated article or a short authentic wire brief. Truncation at "
        f"{MAX_LENGTH} tokens\n"
        f"     (this script's MAX_LENGTH) caps how much of this raw length signal survives tokenization,\n"
        f"     but does not remove it.\n"
    )

    # ---- final model: fit on ALL labelled data, save for deployment ----
    print("=" * 90)
    print(f"  FINAL FIT — training one more model on all {n} labelled samples for deployment")
    print("=" * 90)
    final_model, final_train_time_s, _ = _train_one_fold(texts, labels, tokenizer, epochs=EPOCHS)
    import torch
    NLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "base_model": NLP_BASE_MODEL,
        "max_length": MAX_LENGTH,
        "label_map": {0: "AUTHENTIC", 1: "SYNTHETIC"},
    }, NLP_MODEL_PATH)
    print(f"  Saved final model (trained on all {n} samples, {final_train_time_s:.1f}s) to {NLP_MODEL_PATH}")
    print(
        "  NOTE: this final model's own accuracy has not been independently measured (it was trained on\n"
        "  everything) — the 5-fold CV numbers above are the honest generalisation estimate for what this\n"
        "  training procedure produces, not a claim about this specific checkpoint."
    )

    print(f"\nSaved per-sample out-of-fold predictions to {per_sample_path}")
    print(f"Saved per-fold + aggregate metrics to {fold_path}")
    print(f"\nTotal wall-clock time (5-fold CV + final fit): {total_cv_time_s + final_train_time_s:.1f}s\n")


if __name__ == "__main__":
    run_training()
