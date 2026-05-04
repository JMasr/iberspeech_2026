"""Optional neural LM rescoring (P4 only).

Fine-tune a small causal LM (default: GPT-2-small) on the cleaned 23h
transcripts + retained pseudo-labels — no external text. Score n-best
hypotheses by negative log-likelihood per token.

Decision gate (PROGRESS.md §): only enable if KenLM-only rescore has plateaued
on dev (last two iterations within 0.3% absolute WER).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from src.fusion.mbr import NBestEntry


@dataclass(frozen=True)
class NeuralLMConfig:
    model_dir: Path
    alpha: float = 0.3
    beta: float = 1.0


class NeuralLMScorer:
    def __init__(self, model_dir: str | Path):
        self._model_dir = Path(model_dir)
        self._tok = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:  # pragma: no cover (heavy)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._model = AutoModelForCausalLM.from_pretrained(str(self._model_dir))
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

    def score(self, text: str) -> float:
        """Return total log probability (natural log) under the model."""
        self._ensure_loaded()
        import torch

        ids = self._tok(text, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            ids = ids.to("cuda")
        with torch.no_grad():
            out = self._model(ids, labels=ids)
        # HF returns mean NLL; convert to total log prob.
        return float(-out.loss.item() * (ids.shape[1] - 1))


def rescore(
    nbest: list[NBestEntry],
    scorer: NeuralLMScorer,
    cfg: NeuralLMConfig,
) -> NBestEntry:
    """Combine acoustic + neural LM scores."""
    if not nbest:
        raise ValueError("rescore requires at least one n-best entry")
    best = None
    best_score = -math.inf
    for entry in nbest:
        words = entry.text.split()
        lm_log = scorer.score(entry.text)
        combined = entry.score + cfg.alpha * lm_log + cfg.beta * len(words)
        if combined > best_score:
            best_score = combined
            best = entry
    assert best is not None
    return best
