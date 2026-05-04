"""End-to-end SD pipeline orchestrator.

For one recording:
  1. Load Stage 0 chunks (already VAD'd).
  2. Run pyannote segmentation-3.0 → primary segments + overlap events.
  3. For each segment, extract a WeSpeaker embedding from a 1.5s window centered
     at the segment midpoint.
  4. VBx-cluster embeddings → cluster ids per segment.
  5. Assign overlap top-2 speakers (uses the same cluster ids).
  6. Snap boundaries to ASR word edges (the no-collar lever).
  7. Cleanup: min-duration, same-speaker merge, file-duration clamp.
  8. Write ``<rec>.rttm`` to the experiment dir.

Heavy imports stay local. The function ``run_recording`` takes a small context
struct so the deterministic pieces are testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.sd.refine import Segment, cleanup, snap_boundaries, write_rttm


@dataclass
class SDContext:  # pragma: no cover (heavy)
    """Bundle of dependencies the SD pipeline plugs into."""

    segment_recording: Callable[[str], tuple[list[Segment], list]]  # (segments, overlap_events)
    embed_segment: Callable[[Segment], object]
    cluster_embeddings: Callable[[list[object]], list[int]]
    file_duration_s: Callable[[str], float]


def run_recording(  # pragma: no cover (heavy)
    record_id: str,
    audio_path: str,
    ctx: SDContext,
    word_edges: list[float],
    out_dir: str | Path,
) -> Path:
    primary, overlap_events = ctx.segment_recording(audio_path)
    embeddings = [ctx.embed_segment(s) for s in primary]
    cluster_labels = ctx.cluster_embeddings(embeddings)
    labeled = [
        Segment(start_s=s.start_s, end_s=s.end_s, speaker=f"SPEAKER_{lbl:02d}")
        for s, lbl in zip(primary, cluster_labels)
    ]
    from src.sd.overlap import assign_overlap

    labeled = assign_overlap(labeled, overlap_events, cluster_labels)
    snapped = snap_boundaries(labeled, word_edges)
    duration = ctx.file_duration_s(audio_path)
    final = cleanup(snapped, duration)
    out_path = Path(out_dir) / f"{record_id}.rttm"
    return write_rttm(record_id, final, out_path)
