"""SD scoring: DER, no collar, overlap included.

We use ``pyannote.metrics`` as the primary scorer because installing ``dscore``
on CI is brittle. The ALBAYZIN scoring tool is dscore; both produce the same
DER under the same configuration. We wrap pyannote.metrics with the explicit
``collar=0.0, skip_overlap=False`` flags that match the eval rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SDResult:
    der: float
    miss: float
    false_alarm: float
    confusion: float
    n_records: int
    per_record: dict[str, dict[str, float]]


def _parse_rttm(path: Path):  # pragma: no cover (used at scoring time)
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        speaker = parts[7]
        ann[Segment(start, start + dur)] = speaker
    return ann


def score_directory(hyp_dir: str | Path, ref_dir: str | Path) -> SDResult:
    """Score every ``<rec>.rttm`` in ``hyp_dir`` against ``<rec>.rttm`` in ref_dir."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    hyp_dir = Path(hyp_dir)
    ref_dir = Path(ref_dir)

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    per_record = {}
    n = 0
    for hyp_rttm in sorted(hyp_dir.glob("*.rttm")):
        rec_id = hyp_rttm.stem
        ref_rttm = ref_dir / f"{rec_id}.rttm"
        if not ref_rttm.exists():
            continue
        ref = _parse_rttm(ref_rttm)
        hyp = _parse_rttm(hyp_rttm)
        result = metric(ref, hyp, detailed=True)
        per_record[rec_id] = {
            "der": float(result["diarization error rate"]),
            "miss": float(result.get("missed detection", 0.0)),
            "false_alarm": float(result.get("false alarm", 0.0)),
            "confusion": float(result.get("confusion", 0.0)),
        }
        n += 1

    if n == 0:
        raise FileNotFoundError(f"No matching RTTM pairs under {hyp_dir} / {ref_dir}")

    aggregated = abs(metric)
    detail = metric[:]  # type: ignore[index]
    return SDResult(
        der=float(aggregated),
        miss=float(detail.get("missed detection", 0.0)),
        false_alarm=float(detail.get("false alarm", 0.0)),
        confusion=float(detail.get("confusion", 0.0)),
        n_records=n,
        per_record=per_record,
    )
