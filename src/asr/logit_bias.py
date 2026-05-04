"""Province/topic-conditioned logit biasing for Whisper inference.

Replaces fragile free-text Whisper prompts. We build small lexicons from the
cleaned 23h transcripts (top-N tokens by tf-idf within each (province, topic)
bucket), then bias the decoder logits by a small positive value on the matching
tokens at inference time.

Decision gate (PROGRESS.md): keep biasing only if dev WER improves ≥0.3% abs.

Usage::

    bias = build_lexicons(transcripts_df)  # one-time
    proc = LogitsProcessor(bias.token_ids_for(province="LUG", topic="ganado"), bias_value=0.5)
    pipeline.generate(..., logits_processor=[proc])
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

_TOKEN_RE = re.compile(r"[\wáéíóúñü']+", re.UNICODE)


@dataclass
class BiasLexicon:
    """A flat dict of (province, topic) → list[token]."""

    by_bucket: dict[tuple[str, str], list[str]]
    universal: list[str]

    def words_for(self, province: str | None = None, topic: str | None = None) -> list[str]:
        words = list(self.universal)
        if province is not None and topic is not None:
            words.extend(self.by_bucket.get((province, topic), []))
        elif province is not None:
            for (p, _t), ws in self.by_bucket.items():
                if p == province:
                    words.extend(ws)
        return list(dict.fromkeys(words))

    def to_json(self) -> dict:
        return {
            "universal": self.universal,
            "by_bucket": [
                {"province": p, "topic": t, "words": ws} for (p, t), ws in self.by_bucket.items()
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "BiasLexicon":
        bb = {(e["province"], e["topic"]): list(e["words"]) for e in data.get("by_bucket", [])}
        return cls(by_bucket=bb, universal=list(data.get("universal", [])))


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def build_lexicons(
    transcripts: list[dict],
    *,
    top_n: int = 64,
    min_doc_freq: int = 3,
) -> BiasLexicon:
    """Build per-(province, topic) lexicons from transcript rows.

    Each row is ``{"transcript": str, "province": str|None, "topic": str|None}``.
    A token enters the bucket lexicon if its tf-idf score within the bucket is
    in the top-N AND its document frequency in the bucket is ≥ ``min_doc_freq``.
    """
    from collections import defaultdict

    bucket_docs: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    bucket_token_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    universal_counts: dict[str, int] = defaultdict(int)
    total_universal_docs = 0

    for row in transcripts:
        toks = _tokenize(row.get("transcript", ""))
        if not toks:
            continue
        province = row.get("province") or "_"
        topic = row.get("topic") or "_"
        bucket = (province, topic)
        bucket_docs[bucket].append(set(toks))
        for t in toks:
            bucket_token_counts[bucket][t] += 1
            universal_counts[t] += 1
        total_universal_docs += 1

    if total_universal_docs == 0:
        return BiasLexicon(by_bucket={}, universal=[])

    by_bucket: dict[tuple[str, str], list[str]] = {}
    for bucket, docs in bucket_docs.items():
        token_counts = bucket_token_counts[bucket]
        df: dict[str, int] = defaultdict(int)
        for s in docs:
            for t in s:
                df[t] += 1
        tfidf: dict[str, float] = {}
        n_docs = len(docs)
        for t, tf in token_counts.items():
            if df[t] < min_doc_freq:
                continue
            idf_global = math.log((1 + total_universal_docs) / (1 + universal_counts[t]))
            tfidf[t] = (tf / max(1, n_docs)) * idf_global
        top = sorted(tfidf.items(), key=lambda kv: -kv[1])[:top_n]
        by_bucket[bucket] = [t for t, _ in top]

    # Universal: top high-frequency tokens (function words mostly).
    universal = sorted(universal_counts.items(), key=lambda kv: -kv[1])
    universal_words = [t for t, c in universal if c >= 50][:32]

    return BiasLexicon(by_bucket=by_bucket, universal=universal_words)


def make_logits_processor(
    words: list[str], bias_value: float, processor_factory
):  # pragma: no cover (heavy)
    """Build a HuggingFace LogitsProcessor that adds ``bias_value`` to the logits
    of every token id in ``words`` (after tokenization with the model's tokenizer).

    ``processor_factory`` is ``transformers.LogitsProcessor`` or compatible. We
    keep this as a thin shim so the deterministic logit_bias logic is testable
    without transformers installed.
    """
    raise NotImplementedError(
        "make_logits_processor must be wired by the caller with the model tokenizer."
    )


def save_lexicons(lex: BiasLexicon, path: str | Path) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lex.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_lexicons(path: str | Path) -> BiasLexicon:
    import json

    return BiasLexicon.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
