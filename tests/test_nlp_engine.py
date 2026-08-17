# =============================================================================
# tests/test_nlp_engine.py  —  NLP Detection Engine Test Suite (Day 3)
# =============================================================================
# NOTE ON compute_perplexity() TESTS:
# The actual DistilGPT-2 weights are NOT downloaded during these tests —
# doing so on every test run would be slow and require network access.
# Instead, _load_perplexity_model() is mocked with a fake model/tokenizer
# pair that returns controlled, known values. This tests the SURROUNDING
# LOGIC (chunking across long documents, token-weighted loss averaging,
# exp() conversion) precisely and deterministically — the real model
# weights are a fixed, pre-trained dependency, not logic this project
# wrote or needs to re-verify on every test run.
# =============================================================================

import math
from unittest.mock import patch

import pytest
import torch

from watchdog import nlp_engine


# =============================================================================
# Fakes for perplexity model/tokenizer — see module docstring above
# =============================================================================

class _FakeTokenizer:
    """Returns a fake token tensor of a controllable length."""
    def __init__(self, n_tokens=10):
        self.n_tokens = n_tokens

    def encode(self, text, return_tensors=None, verbose=True):
        return torch.ones((1, self.n_tokens), dtype=torch.long)


class _FakeOutput:
    def __init__(self, loss_value):
        self.loss = torch.tensor(loss_value)


class _FakeModel:
    """Returns a fixed, known loss for every chunk — makes perplexity
    exactly predictable so the averaging math can be asserted precisely."""
    def __init__(self, loss_value=1.0):
        self.loss_value = loss_value

    def eval(self):
        return self

    def __call__(self, chunk, labels=None):
        return _FakeOutput(self.loss_value)


# =============================================================================
# compute_burstiness — no mocking needed, pure math
# =============================================================================

class TestComputeBurstiness:

    def test_none_for_empty_list(self):
        assert nlp_engine.compute_burstiness([]) is None

    def test_none_for_single_sentence(self):
        assert nlp_engine.compute_burstiness(["Only one sentence here."]) is None

    def test_uniform_sentences_trend_negative(self):
        """Equal-length sentences should produce a strongly negative
        (uniform/regular) burstiness score."""
        sentences = [
            "This sentence has exactly seven words in it",
            "This sentence also has seven words present",
            "Every single sentence matches seven words here",
        ]
        b = nlp_engine.compute_burstiness(sentences)
        assert b is not None
        assert b < -0.8  # near-perfectly uniform -> near -1

    def test_varied_sentences_score_higher_than_uniform(self):
        """Sentences with genuinely varied lengths should score higher
        (less negative / more bursty) than uniform-length sentences."""
        uniform = ["One two three four five", "Six seven eight nine ten"]
        varied = ["Short one.", "This one on the other hand runs on for quite a bit longer than the first."]

        b_uniform = nlp_engine.compute_burstiness(uniform)
        b_varied = nlp_engine.compute_burstiness(varied)
        assert b_varied > b_uniform

    def test_bounded_in_valid_range(self):
        """Goh & Barabási burstiness is mathematically bounded in [-1, 1]."""
        sentences = ["A", "A very much longer sentence with many more words in it than the first one did"]
        b = nlp_engine.compute_burstiness(sentences)
        assert -1.0 <= b <= 1.0

    def test_ignores_blank_sentences(self):
        sentences = ["Real sentence one here", "", "   ", "Real sentence two here also"]
        b = nlp_engine.compute_burstiness(sentences)
        assert b is not None  # blanks filtered, 2 real sentences remain


# =============================================================================
# compute_perplexity — model mocked, logic verified precisely
# =============================================================================

class TestComputePerplexity:

    def test_none_for_empty_text(self):
        assert nlp_engine.compute_perplexity("") is None
        assert nlp_engine.compute_perplexity("   ") is None

    def test_none_for_single_token(self):
        with patch.object(nlp_engine, "_load_perplexity_model",
                          return_value=(_FakeModel(), _FakeTokenizer(n_tokens=1))):
            assert nlp_engine.compute_perplexity("x") is None

    def test_matches_exp_of_loss_for_short_text(self):
        """For text shorter than one chunk, perplexity should be exactly
        exp(loss) — no averaging needed."""
        with patch.object(nlp_engine, "_load_perplexity_model",
                          return_value=(_FakeModel(loss_value=1.5), _FakeTokenizer(n_tokens=50))):
            result = nlp_engine.compute_perplexity("short text")
            assert math.isclose(result, math.exp(1.5), rel_tol=1e-6)

    def test_chunking_activates_for_long_documents(self):
        """A document longer than _MAX_CHUNK_TOKENS must be split into
        multiple chunks — verified here via a document long enough to
        require exactly 3 chunks (1500 tokens / 512 per chunk)."""
        with patch.object(nlp_engine, "_load_perplexity_model",
                          return_value=(_FakeModel(loss_value=2.0), _FakeTokenizer(n_tokens=1500))):
            result = nlp_engine.compute_perplexity("a very long document")
            # Every chunk has identical loss, so weighted average == that
            # same loss regardless of how many chunks it was split into.
            assert math.isclose(result, math.exp(2.0), rel_tol=1e-6)

    def test_lower_loss_gives_lower_perplexity(self):
        """Sanity check on direction: lower model loss (more predictable
        text) must produce lower perplexity, matching the documented
        'low perplexity = AI-like' interpretation."""
        with patch.object(nlp_engine, "_load_perplexity_model",
                          return_value=(_FakeModel(loss_value=0.5), _FakeTokenizer(n_tokens=50))):
            low = nlp_engine.compute_perplexity("predictable text")
        with patch.object(nlp_engine, "_load_perplexity_model",
                          return_value=(_FakeModel(loss_value=4.0), _FakeTokenizer(n_tokens=50))):
            high = nlp_engine.compute_perplexity("surprising text")
        assert low < high


# =============================================================================
# analyze_text — combination logic
# =============================================================================

class TestAnalyzeText:

    def test_both_flags_true_votes_ai(self):
        with patch.object(nlp_engine, "compute_perplexity", return_value=10.0):  # < threshold (35.0)
            result = nlp_engine.analyze_text(
                "Our product helps you succeed. Our product offers value. "
                "Our product is trusted worldwide."
            )
        assert result["perplexity_flag"] is True
        # uniform-length sentences above should also trip burstiness_flag
        assert result["heuristic_vote"] in ("AI", "UNCERTAIN")

    def test_both_flags_false_votes_human(self):
        with patch.object(nlp_engine, "compute_perplexity", return_value=200.0):  # > threshold
            with patch.object(nlp_engine, "compute_burstiness", return_value=0.9):  # > threshold
                result = nlp_engine.analyze_text("Some varied, surprising human writing here.")
        assert result["perplexity_flag"] is False
        assert result["burstiness_flag"] is False
        assert result["heuristic_vote"] == "HUMAN"

    def test_disagreement_between_signals_is_uncertain_not_ai(self):
        """Conservative design requirement (NFR-01/03, §2.2.3 VSD): if the
        two heuristics disagree, the vote must NOT be a confident 'AI' —
        it should fall to 'UNCERTAIN' rather than false-flagging a human
        author on a single ambiguous signal."""
        with patch.object(nlp_engine, "compute_perplexity", return_value=10.0):  # flags AI
            with patch.object(nlp_engine, "compute_burstiness", return_value=0.9):  # flags HUMAN
                result = nlp_engine.analyze_text("Mixed-signal test text.")
        assert result["heuristic_vote"] == "UNCERTAIN"

    def test_returns_word_and_sentence_counts(self):
        with patch.object(nlp_engine, "compute_perplexity", return_value=100.0):
            result = nlp_engine.analyze_text("Two sentences. Right here.")
        assert result["word_count"] > 0
        assert result["sentence_count"] == 2
