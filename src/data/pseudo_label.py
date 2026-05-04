"""Pseudo-labeling for the 10h dirty long-form pool.

Tri-condition gate (all must hold for an utterance to be retained):
  1. Token-level Whisper↔XLS-R agreement ≥ 0.85.
  2. Whisper avg-logprob > per-SNR-bucket threshold.
  3. Length-normalized score within distribution (z-score < 2).

Inputs:
  - Whisper hypotheses with avg_logprob, duration, words.
  - XLS-R 1-best transcripts.
  - Stage 0 chunk parquet (for SNR bucket).

Output: ``data/interim/pseudo_labels.parquet`` with one row per retained chunk:
  chunk_id, record_id, audio_path, start_s, end_s, transcript, snr_db, agreement.

We do NOT mix in any text from the dirty long-form transcripts directly; those
are noisy and would poison FT. Only ensemble agreement keeps a chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SNR_BUCKETS = [(-1e9, 5.0), (5.0, 15.0), (15.0, 1e9)]
DEFAULT_LOGPROB_THRESHOLDS = {0: -0.55, 1: -0.45, 2: -0.35}


@dataclass(frozen=True)
class GateConfig:
    agreement_min: float = 0.85
    snr_buckets: list[tuple[float, float]] | None = None
    logprob_thresholds: dict[int, float] | None = None
    length_z_max: float = 2.0


def token_agreement(words_a: list[str], words_b: list[str]) -> float:
    """Length-normalized longest common subsequence over lowercase tokens.

    Returns a score in [0, 1]. Empty inputs → 0.
    """
    a = [w.lower() for w in words_a if w]
    b = [w.lower() for w in words_b if w]
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        ai = a[i]
        row = dp[i + 1]
        prev = dp[i]
        for j in range(m):
            if ai == b[j]:
                row[j + 1] = prev[j] + 1
            else:
                row[j + 1] = max(prev[j + 1], row[j])
    lcs = dp[n][m]
    return float(2 * lcs / (n + m))


def snr_bucket(snr_db: float, buckets: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= snr_db < hi:
            return i
    return len(buckets) - 1


def length_z_score(words: list[str], duration_s: float, mean_wps: float, std_wps: float) -> float:
    if duration_s <= 0 or std_wps <= 0:
        return 0.0
    wps = len(words) / duration_s
    return (wps - mean_wps) / std_wps


def gate(
    *,
    whisper_words: list[str],
    xlsr_words: list[str],
    whisper_logprob: float,
    snr_db: float,
    duration_s: float,
    mean_wps: float,
    std_wps: float,
    config: GateConfig = GateConfig(),
) -> tuple[bool, dict]:
    """Apply the tri-condition gate. Returns (kept, diagnostics)."""
    buckets = config.snr_buckets or DEFAULT_SNR_BUCKETS
    thresholds = config.logprob_thresholds or DEFAULT_LOGPROB_THRESHOLDS

    agreement = token_agreement(whisper_words, xlsr_words)
    bucket_idx = snr_bucket(snr_db, buckets)
    logprob_threshold = thresholds.get(bucket_idx, thresholds[0])
    z = abs(length_z_score(whisper_words, duration_s, mean_wps, std_wps))

    kept = (
        agreement >= config.agreement_min
        and whisper_logprob > logprob_threshold
        and z < config.length_z_max
    )
    return kept, {
        "agreement": agreement,
        "snr_bucket": bucket_idx,
        "logprob_threshold": logprob_threshold,
        "length_z": z,
        "passed": kept,
    }


def build_pseudo_labels(
    hypotheses_parquet: str | Path,
    chunks_parquet: str | Path,
    out_path: str | Path,
    *,
    config: GateConfig = GateConfig(),
) -> Path:
    """Apply the gate and write the retained pseudo-labels.

    ``hypotheses_parquet`` schema (one row per chunk):
      chunk_id, record_id, audio_path, start_s, end_s,
      whisper_words (list[str]), whisper_logprob (float),
      xlsr_words (list[str]).

    ``chunks_parquet`` is the union of Stage 0 outputs (one row per chunk with snr_db).
    """
    import numpy as np
    import pandas as pd

    hyps = pd.read_parquet(hypotheses_parquet)
    chunks = pd.read_parquet(chunks_parquet)
    df = hyps.merge(chunks[["chunk_id", "wada_snr_db"]], on="chunk_id", how="left")

    df["duration_s"] = df["end_s"] - df["start_s"]
    word_counts = df["whisper_words"].apply(len)
    wps = word_counts / df["duration_s"].clip(lower=0.1)
    mean_wps = float(wps.mean())
    std_wps = float(wps.std()) or 1.0

    rows = []
    for _, r in df.iterrows():
        kept, diag = gate(
            whisper_words=list(r["whisper_words"]),
            xlsr_words=list(r["xlsr_words"]),
            whisper_logprob=float(r["whisper_logprob"]),
            snr_db=float(r.get("wada_snr_db", 0.0) or 0.0),
            duration_s=float(r["duration_s"]),
            mean_wps=mean_wps,
            std_wps=std_wps,
            config=config,
        )
        if not kept:
            continue
        rows.append(
            {
                "chunk_id": r["chunk_id"],
                "record_id": r["record_id"],
                "audio_path": r["audio_path"],
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "transcript": " ".join(r["whisper_words"]),
                "snr_db": float(r.get("wada_snr_db", 0.0) or 0.0),
                "agreement": diag["agreement"],
            }
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    # Manifest summary alongside the parquet, for the PROGRESS log.
    summary = {
        "n_input": int(len(df)),
        "n_kept": int(len(rows)),
        "yield_seconds": float(sum(r["end_s"] - r["start_s"] for r in rows)),
        "mean_wps": mean_wps,
        "std_wps": std_wps,
    }
    np.save(out_path.with_suffix(".summary.npy"), summary, allow_pickle=True)
    return out_path
