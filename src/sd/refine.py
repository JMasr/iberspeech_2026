"""ASR-anchored boundary refinement and RTTM cleanup.

The single biggest no-collar lever (per the plan): for each diarization segment
boundary, snap to the nearest ASR word edge within ±200ms. Standard pyannote
pipelines lose 5–10% absolute DER moving from a 250ms collar to no-collar; this
counters that.

RTTM cleanup is deterministic and conservative:
  - Drop sub-200ms segments.
  - Merge adjacent same-speaker segments with <500ms gap between them.
  - Clamp every segment to [0, file_duration].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MIN_SEGMENT_S = 0.20
MERGE_GAP_S = 0.50
SNAP_RADIUS_S = 0.20


@dataclass
class Segment:
    start_s: float
    end_s: float
    speaker: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class WordEdge:
    time_s: float
    kind: str  # "start" or "end"


def collect_word_edges(words) -> list[float]:
    """Flatten word-span starts and ends into a sorted list of times."""
    edges = []
    for w in words:
        edges.append(float(w.start_s))
        edges.append(float(w.end_s))
    edges.sort()
    return edges


def snap_boundaries(
    segments: list[Segment],
    word_edges: list[float],
    radius_s: float = SNAP_RADIUS_S,
) -> list[Segment]:
    """For each segment boundary, snap to the nearest word edge within radius."""
    if not word_edges:
        return list(segments)
    edges = sorted(word_edges)
    out = []
    for s in segments:
        new_start = _nearest_edge(edges, s.start_s, radius_s)
        new_end = _nearest_edge(edges, s.end_s, radius_s)
        if new_end <= new_start:
            new_end = max(new_start + MIN_SEGMENT_S, new_end)
        out.append(Segment(start_s=new_start, end_s=new_end, speaker=s.speaker))
    return out


def _nearest_edge(edges_sorted: list[float], target: float, radius: float) -> float:
    """Binary search for the closest edge within ``radius``; otherwise return target."""
    import bisect

    i = bisect.bisect_left(edges_sorted, target)
    candidates = []
    if i < len(edges_sorted):
        candidates.append(edges_sorted[i])
    if i > 0:
        candidates.append(edges_sorted[i - 1])
    if not candidates:
        return target
    nearest = min(candidates, key=lambda x: abs(x - target))
    return nearest if abs(nearest - target) <= radius else target


def cleanup(segments: list[Segment], file_duration_s: float) -> list[Segment]:
    """Apply min-duration, same-speaker merge, and clamp."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s.start_s, s.end_s))
    # Clamp to file duration.
    clamped = []
    for s in segments:
        a = max(0.0, s.start_s)
        b = min(file_duration_s, s.end_s)
        if b - a >= MIN_SEGMENT_S:
            clamped.append(Segment(a, b, s.speaker))
    if not clamped:
        return []
    # Merge same-speaker if gap < MERGE_GAP_S.
    merged: list[Segment] = [clamped[0]]
    for nxt in clamped[1:]:
        prev = merged[-1]
        if nxt.speaker == prev.speaker and nxt.start_s - prev.end_s < MERGE_GAP_S:
            merged[-1] = Segment(prev.start_s, max(prev.end_s, nxt.end_s), prev.speaker)
        else:
            merged.append(nxt)
    return merged


def write_rttm(record_id: str, segments: list[Segment], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in segments:
        if s.duration_s <= 0:
            continue
        lines.append(
            f"SPEAKER {record_id} 1 {s.start_s:.3f} {s.duration_s:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def parse_rttm(path: str | Path) -> list[Segment]:
    """Read an RTTM file. Convenient for tests + the boundary-snap CLI."""
    out: list[Segment] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        out.append(Segment(start, start + dur, spk))
    return out
