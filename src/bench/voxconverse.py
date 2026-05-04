"""VoxConverse loader for SD pipeline validation.

VoxConverse is a multi-speaker conversational dataset with full RTTM diarization
labels (CC-BY-4.0). We use the ``dev`` set for SD pipeline validation:
  - Verify pyannote VAD + segmentation produce sensible turns.
  - Verify VBx clustering finds multiple speakers.
  - Measure boundary-snap lift on no-collar DER (with vs without).

Inputs:
  - ``audio_dir`` — VoxConverse audio (we re-encode to 16k mono WAV at load).
  - ``rttm_dir``  — VoxConverse dev RTTMs.

Output: ``data/bench/voxconverse/manifest.parquet`` with one row per recording:
  record_id, audio_path, rttm_path, duration_s, n_speakers (from RTTM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUT = Path("data/bench/voxconverse")


@dataclass
class VoxConverseConfig:
    audio_dir: Path
    rttm_dir: Path
    out_dir: Path = DEFAULT_OUT
    n_recordings: int = 5  # keep tiny for smoke tests; use 0 for "all".


def build_manifest(cfg: VoxConverseConfig) -> Path:
    """Materialize the manifest. Heavy imports local."""
    import pandas as pd
    import soundfile as sf

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rttms = sorted(Path(cfg.rttm_dir).glob("*.rttm"))
    if cfg.n_recordings:
        rttms = rttms[: cfg.n_recordings]
    for rttm in rttms:
        rec_id = rttm.stem
        wav = Path(cfg.audio_dir) / f"{rec_id}.wav"
        if not wav.exists():
            # VoxConverse ships .wav under audio/{dev,test}; tolerate that.
            for cand in Path(cfg.audio_dir).rglob(f"{rec_id}.wav"):
                wav = cand
                break
        if not wav.exists():
            continue
        with sf.SoundFile(str(wav)) as f:
            duration = f.frames / float(f.samplerate)
        speakers = _count_speakers(rttm)
        rows.append(
            {
                "record_id": rec_id,
                "audio_path": str(wav),
                "rttm_path": str(rttm),
                "duration_s": duration,
                "n_speakers": speakers,
                "source": "voxconverse",
                "split": "bench",
            }
        )
    out = cfg.out_dir / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _count_speakers(rttm_path: Path) -> int:
    speakers = set()
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0] == "SPEAKER":
            speakers.add(parts[7])
    return len(speakers)
