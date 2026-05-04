"""Build a unified COSER manifest and a stratified train/dev split.

Inputs (under ``data/raw/``):
  - ``segments/``  — short validated WAV files (≤30s) and matching .txt transcripts.
  - ``longform/``  — full-length recordings with un-cleaned transcripts.
  - ``meta.csv``   — per-recording metadata (record_id, province, age, education,
                     topic, year, channel, source).
  - ``rttm/``      — per-recording RTTM diarization.

Outputs:
  - ``data/interim/manifest.parquet`` — one row per training/eval segment with all
    metadata and split label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

ManifestRow = dict


@dataclass(frozen=True)
class IngestConfig:
    raw_root: Path
    out_path: Path
    dev_fraction: float = 0.1
    seed: int = 20260501


def _province_age_topic_key(row: ManifestRow) -> str:
    age = row.get("speaker_age")
    age_decile = "NA" if age is None else f"d{int(age) // 10 * 10}"
    return f"{row.get('province', 'XX')}|{age_decile}|{row.get('topic', 'na')}"


def _stable_hash(s: str, salt: int) -> float:
    h = hashlib.sha1(f"{salt}:{s}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF


def stratified_split(rows: list[ManifestRow], dev_fraction: float, seed: int) -> list[str]:
    """Return a parallel list of split labels ('train' or 'dev').

    Stratification: bucket by (province × age decile × topic). Within each bucket
    we deterministically pick the first ``ceil(dev_fraction * N)`` items by
    stable hash so re-running ingestion is reproducible. We further protect
    against single-bucket dominance: any bucket with < 5 items is fully assigned
    to train (too small to split safely).
    """
    from collections import defaultdict
    import math

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[_province_age_topic_key(row)].append(i)

    splits = ["train"] * len(rows)
    for key, idx_list in buckets.items():
        if len(idx_list) < 5:
            continue
        scored = sorted(
            idx_list, key=lambda i: _stable_hash(rows[i].get("segment_id", str(i)), seed)
        )
        n_dev = max(1, math.ceil(dev_fraction * len(scored)))
        for j in scored[:n_dev]:
            splits[j] = "dev"
    return splits


def build_manifest(cfg: IngestConfig) -> Path:
    """Materialize ``manifest.parquet`` and return its path.

    Heavy imports (pandas, pyarrow) are local; the parsing logic is plain stdlib.
    """
    import pandas as pd

    raw = Path(cfg.raw_root)
    seg_dir = raw / "segments"
    longform_dir = raw / "longform"
    meta_csv = raw / "meta.csv"

    if not meta_csv.exists():
        raise FileNotFoundError(
            f"Expected metadata CSV at {meta_csv}. Place the COSER metadata there before "
            "running `make ingest`. See data/raw/README.md (created on first ingest)."
        )
    meta = pd.read_csv(meta_csv)
    meta = meta.set_index("record_id", drop=False)

    rows: list[ManifestRow] = []
    for wav in sorted(seg_dir.glob("*.wav")) if seg_dir.exists() else []:
        record_id = wav.stem.split("__")[0]  # convention: <record_id>__<seg_idx>.wav
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        m = meta.loc[record_id].to_dict() if record_id in meta.index else {}
        rows.append(
            {
                "segment_id": wav.stem,
                "record_id": record_id,
                "audio_path": str(wav),
                "transcript_path": str(txt),
                "transcript": txt.read_text(encoding="utf-8").strip(),
                "duration_s": _wav_duration(wav),
                "kind": "segment",
                "province": m.get("province"),
                "speaker_age": m.get("speaker_age"),
                "speaker_id": m.get("speaker_id"),
                "topic": m.get("topic"),
                "year": m.get("year"),
                "channel": m.get("channel"),
            }
        )
    for wav in sorted(longform_dir.glob("*.wav")) if longform_dir.exists() else []:
        record_id = wav.stem
        txt = wav.with_suffix(".txt")
        m = meta.loc[record_id].to_dict() if record_id in meta.index else {}
        rows.append(
            {
                "segment_id": record_id,
                "record_id": record_id,
                "audio_path": str(wav),
                "transcript_path": str(txt) if txt.exists() else None,
                "transcript": txt.read_text(encoding="utf-8").strip() if txt.exists() else None,
                "duration_s": _wav_duration(wav),
                "kind": "longform",
                "province": m.get("province"),
                "speaker_age": m.get("speaker_age"),
                "speaker_id": m.get("speaker_id"),
                "topic": m.get("topic"),
                "year": m.get("year"),
                "channel": m.get("channel"),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No audio found under {seg_dir} or {longform_dir}. The training data drop is "
            "due 2026-05-01. Verify the raw directory contents."
        )

    splits = stratified_split(
        [r for r in rows if r["kind"] == "segment"], cfg.dev_fraction, cfg.seed
    )
    seg_idx = 0
    for r in rows:
        if r["kind"] == "segment":
            r["split"] = splits[seg_idx]
            seg_idx += 1
        else:
            # Long-form is for self-training (round-2 FT) and held-out probes — not in train.
            r["split"] = "longform"

    df = pd.DataFrame(rows)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_path, index=False)
    return cfg.out_path


def _wav_duration(path: Path) -> float:
    """Cheap duration via WAV header — avoids loading full audio."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        return f.frames / float(f.samplerate)
