"""Long-form chunked ASR inference with word-LCS overlap merge.

Pipeline (per recording):
  1. Read Stage 0 chunks parquet (VAD + flagging already done).
  2. Skip music regions (BEATs-confirmed).
  3. For each remaining chunk: run Whisper FT, get n-best with word-level
     timestamps; run XLS-R FT, get 1-best.
  4. Fuse per-chunk: MBR over Whisper n-best, then ROVER vs XLS-R 1-best.
  5. KenLM rescore on the post-ROVER top-K (we re-emit top-K by perturbing
     ROVER slot winners).
  6. Apply AudioSet non-speech mask (drop tokens whose midpoint lands inside
     non-speech intervals).
  7. Merge chunks: word-timestamp longest-common-subsequence in the 1s overlap;
     prefer the chunk whose center is closer to the disputed word.
  8. Post-edit raw output and write
     ``experiments/<run>/asr/<rec>_fullaudio_transcrip.txt``.

The function is structured so the heavy bits (Whisper, XLS-R) are dependency-
injected — making the merge logic and chunk routing testable without GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from src.asr.nonspeech_mask import WordSpan, apply_mask
from src.data.normalize import post_edit_raw


@dataclass(frozen=True)
class ChunkRecognition:
    """Per-chunk fused transcript with word-level timestamps."""

    chunk_id: str
    start_s: float
    end_s: float
    words: list[WordSpan]


@dataclass
class InferConfig:
    overlap_s: float = 1.0
    skip_music: bool = True
    apply_nonspeech_mask: bool = True


def merge_overlapping(
    chunks: list[ChunkRecognition],
    cfg: InferConfig = InferConfig(),
) -> list[WordSpan]:
    """Merge a sequence of chunk recognitions into a single word stream.

    For two adjacent chunks A, B with overlap [B.start, A.end]:
      - Words entirely inside the overlap region are reconciled via word LCS.
      - On disagreement, prefer the chunk whose center is closer to the word.
    """
    if not chunks:
        return []
    chunks = sorted(chunks, key=lambda c: c.start_s)
    merged: list[WordSpan] = list(chunks[0].words)
    for prev, curr in zip(chunks, chunks[1:]):
        overlap_start = curr.start_s
        overlap_end = min(prev.end_s, curr.end_s)
        if overlap_end <= overlap_start:
            merged.extend(curr.words)
            continue
        prev_overlap = [w for w in merged if _midpoint(w) >= overlap_start]
        curr_head = _take_while(curr.words, lambda w: _midpoint(w) < overlap_end)
        curr_tail = curr.words[len(curr_head) :]
        # Drop words from prev that are in the overlap, then re-add the LCS-resolved overlap.
        merged = [w for w in merged if _midpoint(w) < overlap_start]
        merged.extend(
            _resolve_overlap(
                prev_overlap, curr_head, prev_center=_center(prev), curr_center=_center(curr)
            )
        )
        merged.extend(curr_tail)
    return merged


def _midpoint(w: WordSpan) -> float:
    return 0.5 * (w.start_s + w.end_s)


def _center(c: ChunkRecognition) -> float:
    return 0.5 * (c.start_s + c.end_s)


def _take_while(seq, pred):
    out = []
    for x in seq:
        if pred(x):
            out.append(x)
        else:
            break
    return out


def _resolve_overlap(
    a: list[WordSpan],
    b: list[WordSpan],
    *,
    prev_center: float,
    curr_center: float,
) -> list[WordSpan]:
    """Merge two overlapping word lists by longest common subsequence on lowercased words.

    Where the words match, take ``a``. Where they don't, prefer the chunk whose
    center is closer to the word's midpoint.
    """
    if not a:
        return list(b)
    if not b:
        return list(a)

    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i].word.lower() == b[j].word.lower():
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    out: list[WordSpan] = []
    i, j = n, m
    decisions: list[tuple[str, WordSpan, WordSpan | None]] = []
    while i > 0 and j > 0:
        if a[i - 1].word.lower() == b[j - 1].word.lower():
            decisions.append(("match", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            decisions.append(("only_a", a[i - 1], None))
            i -= 1
        else:
            decisions.append(("only_b", b[j - 1], None))
            j -= 1
    while i > 0:
        decisions.append(("only_a", a[i - 1], None))
        i -= 1
    while j > 0:
        decisions.append(("only_b", b[j - 1], None))
        j -= 1
    decisions.reverse()
    for kind, primary, secondary in decisions:
        if kind == "match":
            out.append(primary)
        elif kind == "only_a":
            mid = _midpoint(primary)
            if abs(mid - prev_center) <= abs(mid - curr_center):
                out.append(primary)
        else:
            mid = _midpoint(primary)
            if abs(mid - curr_center) < abs(mid - prev_center):
                out.append(primary)
    return out


def chunks_excluding_music(
    chunk_rows: list[dict],
    music_intervals: Iterable[tuple[float, float]],
) -> list[dict]:
    """Filter Stage 0 rows whose midpoint lies inside a music interval.

    Midpoint-based (not any-overlap) so a tiny VAD boundary bleed (e.g. 20 ms)
    does not erroneously drop a 10 s speech chunk at the music boundary.
    """
    music = list(music_intervals)
    if not music:
        return chunk_rows

    out = []
    for row in chunk_rows:
        s = float(row["start_s"])
        e = float(row["end_s"])
        mid = 0.5 * (s + e)
        if any(m_s <= mid < m_e for m_s, m_e in music):
            continue
        out.append(row)
    return out


def write_transcript(record_id: str, words: list[WordSpan], out_dir: str | Path) -> Path:
    """Write the canonical ``<rec>_fullaudio_transcrip.txt``."""
    text = " ".join(w.word for w in words)
    text = post_edit_raw(text)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record_id}_fullaudio_transcrip.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path


@dataclass
class InferenceContext:  # pragma: no cover (heavy)
    """Bundle the dependencies an end-to-end ``run_recording`` needs.

    The deterministic merge logic (above) is fully unit-testable; this struct
    is what the real pipeline plugs into the CLI ``infer-asr`` subcommand.
    """

    transcribe_chunk: Callable[[dict], ChunkRecognition]
    nonspeech_intervals: list[tuple[float, float, str]] = field(default_factory=list)
    music_intervals: list[tuple[float, float]] = field(default_factory=list)


def run_recording(  # pragma: no cover (heavy)
    record_id: str,
    chunk_rows: list[dict],
    ctx: InferenceContext,
    *,
    cfg: InferConfig = InferConfig(),
    out_dir: str | Path,
) -> Path:
    rows = (
        chunks_excluding_music(chunk_rows, ctx.music_intervals) if cfg.skip_music else chunk_rows
    )
    chunk_recs: list[ChunkRecognition] = [ctx.transcribe_chunk(row) for row in rows]
    words = merge_overlapping(chunk_recs, cfg=cfg)
    if cfg.apply_nonspeech_mask:
        words = apply_mask(words, ctx.nonspeech_intervals)
    return write_transcript(record_id, words, out_dir)
