"""Top-2 speaker assignment in overlap regions.

pyannote segmentation-3.0's powerset head emits per-frame overlap probabilities.
For frames with overlap, we assign the two cluster centroids closest to the
embedding extracted from a small window around that frame.

Output: extra ``Segment`` objects (one per overlapping speaker) appended to the
diarization, with the same start/end as the dominant speaker's segment.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.sd.refine import Segment


@dataclass(frozen=True)
class OverlapEvent:
    start_s: float
    end_s: float
    primary_emb_index: int  # index into the embedding array
    secondary_emb_index: int


def assign_overlap(
    primary_segments: list[Segment],
    overlap_events: list[OverlapEvent],
    cluster_labels: list[int],
) -> list[Segment]:
    """Inject extra segments for overlap regions.

    Each overlap event becomes one additional ``Segment`` whose speaker label is
    the cluster id of the *secondary* embedding. The primary segments are
    untouched — VBx clustering already covers them.
    """
    extras: list[Segment] = []
    for ev in overlap_events:
        secondary = cluster_labels[ev.secondary_emb_index]
        primary = cluster_labels[ev.primary_emb_index]
        if secondary == primary:
            # Both heads picked the same cluster; nothing to add.
            continue
        extras.append(
            Segment(
                start_s=ev.start_s,
                end_s=ev.end_s,
                speaker=f"SPEAKER_{secondary:02d}",
            )
        )
    return primary_segments + extras
