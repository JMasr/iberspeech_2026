"""KenLM 5-gram build and n-best rescoring.

We train on cleaned in-domain text only:
  - The 23h validated transcripts (after non-speech token strip).
  - The retained pseudo-labels from ``src/data/pseudo_label.py``.

External text is forbidden by the eval rules. Do not mix Common Voice / OSCAR / etc.

Build: writes an ARPA file via the KenLM CLI (``lmplz``). The Python bindings
are only required at rescore time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess

from src.fusion.mbr import NBestEntry


@dataclass(frozen=True)
class LMConfig:
    arpa_path: Path
    alpha: float = 0.5  # LM weight
    beta: float = 1.5  # word-insertion bonus (per-word)


def build_arpa(text_path: str | Path, out_path: str | Path, order: int = 5) -> Path:
    """Build a KenLM ARPA file. Requires the ``lmplz`` binary on PATH.

    Output: ``out_path`` (an ARPA file). We do not invoke ``build_binary``;
    rescoring works fine on the ARPA, and binary builds add platform fragility.
    """
    text_path = Path(text_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lmplz",
        "-o",
        str(order),
        "--text",
        str(text_path),
        "--arpa",
        str(out_path),
        "--discount_fallback",
    ]
    subprocess.run(cmd, check=True)
    return out_path


class KenLMScorer:
    """Lazy KenLM wrapper. Loads on first call to score()."""

    def __init__(self, arpa_path: str | Path):
        self._arpa_path = Path(arpa_path)
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                import kenlm
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "kenlm Python bindings not installed. Install with `uv pip install -e '.[kenlm]'`."
                ) from e
            self._model = kenlm.Model(str(self._arpa_path))

    def score(self, text: str) -> float:
        """Return total log10 probability under the LM (KenLM convention)."""
        self._ensure_loaded()
        return float(self._model.score(text, bos=True, eos=True))


def rescore(
    nbest: list[NBestEntry],
    scorer: KenLMScorer,
    cfg: LMConfig,
) -> NBestEntry:
    """Combine acoustic and LM scores. Returns the highest combined-score entry.

    combined = ac_score + alpha * lm_score / ln10 + beta * num_words

    KenLM emits log10 probabilities, so we divide by ``ln10`` to convert to nats
    before mixing with avg-logprob.
    """
    if not nbest:
        raise ValueError("rescore requires at least one n-best entry")
    log10 = math.log(10.0)
    best = None
    best_score = float("-inf")
    for entry in nbest:
        words = entry.text.split()
        lm_log10 = scorer.score(entry.text)
        combined = entry.score + cfg.alpha * (lm_log10 / log10) + cfg.beta * len(words)
        if combined > best_score:
            best_score = combined
            best = entry
    assert best is not None
    return best
