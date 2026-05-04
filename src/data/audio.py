"""Audio I/O and chunking utilities.

The competition WAVs are 16 kHz mono PCM 16-bit. We trust that contract for the
test set; for training audio we still convert defensively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SAMPLE_RATE = 16_000
CHUNK_SECONDS_DEFAULT = 30.0
OVERLAP_SECONDS_DEFAULT = 1.0


@dataclass(frozen=True)
class Chunk:
    """A speech-bearing audio chunk."""

    chunk_id: str
    record_id: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def load_mono_16k(path: str | Path):
    """Load a WAV as float32 mono 16 kHz numpy array.

    Heavy import is local so unrelated tests do not pull soundfile/librosa.
    """
    import numpy as np
    import soundfile as sf

    path = Path(path)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa  # local import on purpose

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype="float32")


def chunk_intervals(
    speech_segments: list[tuple[float, float]],
    chunk_s: float = CHUNK_SECONDS_DEFAULT,
    overlap_s: float = OVERLAP_SECONDS_DEFAULT,
    record_id: str = "rec",
) -> Iterator[Chunk]:
    """Iterate over chunks bounded by VAD speech segments.

    Each speech segment is sliced into ``chunk_s``-long windows with
    ``overlap_s`` overlap. Trailing chunks shorter than half the chunk size are
    merged into the previous chunk (Whisper degrades on very short windows).

    ``speech_segments`` is a list of (start, end) seconds in the source audio.
    """
    if chunk_s <= overlap_s:
        raise ValueError("chunk_s must exceed overlap_s")
    step = chunk_s - overlap_s
    chunk_idx = 0
    for seg_start, seg_end in speech_segments:
        if seg_end - seg_start < 0.25:
            continue
        cursor = seg_start
        # Pre-compute window starts so we can fold the trailing tail.
        starts = []
        while cursor + chunk_s <= seg_end:
            starts.append(cursor)
            cursor += step
        # Tail: include if it is at least half a chunk; otherwise merge with last.
        if not starts:
            starts.append(seg_start)
            tail_end = seg_end
        else:
            tail_start = starts[-1] + chunk_s
            if seg_end > tail_start:
                if seg_end - tail_start >= chunk_s / 2:
                    starts.append(tail_start - overlap_s)
                    tail_end = seg_end
                else:
                    tail_end = seg_end  # absorb into last
            else:
                tail_end = starts[-1] + chunk_s
        for i, start in enumerate(starts):
            end = tail_end if i == len(starts) - 1 else min(start + chunk_s, seg_end)
            yield Chunk(
                chunk_id=f"{record_id}_c{chunk_idx:05d}",
                record_id=record_id,
                start_s=round(start, 3),
                end_s=round(end, 3),
            )
            chunk_idx += 1


def slice_audio(audio, start_s: float, end_s: float, sr: int = SAMPLE_RATE):
    """Slice a numpy float32 audio array by seconds."""
    a = int(round(start_s * sr))
    b = int(round(end_s * sr))
    return audio[a:b]
