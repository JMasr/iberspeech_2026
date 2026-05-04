"""Common Voice ES manifest builder.

Two paths:
  - ``from_hf`` — pull a tiny slice via ``datasets`` in streaming mode (no full
    download). Default for `make bench-cv-es`.
  - ``from_tsv`` — parse a locally extracted Common Voice TSV (e.g.
    ``cv-corpus-XX-2024-XX-XX/es/test.tsv``) plus the matching ``clips/``
    directory. Useful when you've already downloaded the corpus.

Both paths produce ``data/bench/cv_es/manifest.parquet`` with schema:
  segment_id, audio_path, transcript, duration_s, source = "cv_es", split = "bench".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUT = Path("data/bench/cv_es")


@dataclass
class CVESConfig:
    out_dir: Path = DEFAULT_OUT
    n_samples: int = 50
    hf_dataset: str = "mozilla-foundation/common_voice_17_0"
    hf_split: str = "test"
    hf_language: str = "es"


def from_hf(cfg: CVESConfig = CVESConfig()) -> Path:  # pragma: no cover (heavy + network)
    """Stream a small slice from HF datasets and materialize WAVs + a manifest."""
    from datasets import load_dataset
    import pandas as pd
    import soundfile as sf

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = cfg.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg.hf_dataset, cfg.hf_language, split=cfg.hf_split, streaming=True)

    rows = []
    for i, sample in enumerate(ds):
        if i >= cfg.n_samples:
            break
        audio = sample["audio"]
        wav_path = clips_dir / f"cv_es_{i:04d}.wav"
        sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
        duration = len(audio["array"]) / audio["sampling_rate"]
        rows.append(
            {
                "segment_id": wav_path.stem,
                "audio_path": str(wav_path),
                "transcript": sample.get("sentence", ""),
                "duration_s": float(duration),
                "source": "cv_es",
                "split": "bench",
            }
        )
    out = cfg.out_dir / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def from_tsv(tsv_path: Path, clips_dir: Path, cfg: CVESConfig = CVESConfig()) -> Path:
    """Build a manifest from a locally extracted Common Voice TSV.

    The CV TSV has columns: client_id, path, sentence, up_votes, down_votes,
    age, gender, accents, locale, segment, variant. We use ``path`` and ``sentence``.

    Heavy imports (pandas, soundfile) are local. Returns the parquet path.
    """
    import pandas as pd
    import soundfile as sf

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.head(cfg.n_samples)

    rows = []
    for _, r in df.iterrows():
        clip_path = clips_dir / r["path"]
        if not clip_path.exists():
            continue
        with sf.SoundFile(str(clip_path)) as f:
            duration = f.frames / float(f.samplerate)
        rows.append(
            {
                "segment_id": Path(r["path"]).stem,
                "audio_path": str(clip_path),
                "transcript": str(r.get("sentence", "")),
                "duration_s": duration,
                "source": "cv_es",
                "split": "bench",
            }
        )
    out = cfg.out_dir / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out
