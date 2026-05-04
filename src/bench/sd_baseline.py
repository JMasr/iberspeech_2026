"""Zero-shot pyannote SD baseline + boundary-snap delta on VoxConverse.

Validates:

  H4  pyannote-3.0 zero-shot DER (no collar, overlap incl.) on VoxConverse dev
      is in the 15–25% range — the model is correctly configured.
  H5  Our ASR-anchored boundary snap reduces no-collar DER measurably (≥ 2%
      absolute). Without this lift, the no-collar setting is uneconomic on COSER.

The bench produces three score files:
  - ``baseline_der.json``     — pyannote-3.0 zero-shot, our cleanup applied.
  - ``snapped_der.json``      — same RTTM but with boundary snap to ASR words.
  - ``delta_summary.json``    — paired delta (snap - baseline) per recording.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.sd.refine import cleanup, parse_rttm, snap_boundaries, write_rttm


@dataclass
class SDBaselineConfig:
    manifest_parquet: Path
    asr_words_dir: Path | None = None
    out_dir: Path = Path("experiments/bench/voxconverse")
    pipeline_id: str = "pyannote/speaker-diarization-3.1"


def run(cfg: SDBaselineConfig):  # pragma: no cover (heavy)
    """Run pyannote, score, then snap and re-score."""
    import pandas as pd
    from pyannote.metrics.diarization import DiarizationErrorRate

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.manifest_parquet)

    pipeline = _load_pipeline(cfg.pipeline_id)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    metric_snap = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    baseline_records = {}
    snapped_records = {}
    for _, r in df.iterrows():
        rec_id = r["record_id"]
        baseline_segments = _run_pipeline(pipeline, r["audio_path"])
        baseline_segments = cleanup(baseline_segments, file_duration_s=float(r["duration_s"]))
        baseline_rttm = cfg.out_dir / "baseline" / f"{rec_id}.rttm"
        write_rttm(rec_id, baseline_segments, baseline_rttm)

        ref = _read_rttm_as_pyannote(r["rttm_path"])
        hyp = _segments_to_pyannote(rec_id, baseline_segments)
        baseline_records[rec_id] = float(metric(ref, hyp, detailed=True)["diarization error rate"])

        # Snap if we have ASR words for this recording.
        if cfg.asr_words_dir:
            words_path = cfg.asr_words_dir / f"{rec_id}.json"
            if words_path.exists():
                edges = json.loads(words_path.read_text(encoding="utf-8"))
                snapped = snap_boundaries(baseline_segments, edges)
                snapped = cleanup(snapped, file_duration_s=float(r["duration_s"]))
                snapped_rttm = cfg.out_dir / "snapped" / f"{rec_id}.rttm"
                write_rttm(rec_id, snapped, snapped_rttm)
                hyp_snap = _segments_to_pyannote(rec_id, snapped)
                snapped_records[rec_id] = float(
                    metric_snap(ref, hyp_snap, detailed=True)["diarization error rate"]
                )

    (cfg.out_dir / "baseline_der.json").write_text(
        json.dumps(baseline_records, indent=2), encoding="utf-8"
    )
    if snapped_records:
        delta = {
            rec: {
                "baseline": baseline_records[rec],
                "snapped": snapped_records[rec],
                "delta": snapped_records[rec] - baseline_records[rec],
            }
            for rec in snapped_records
        }
        (cfg.out_dir / "snapped_der.json").write_text(
            json.dumps(snapped_records, indent=2), encoding="utf-8"
        )
        (cfg.out_dir / "delta_summary.json").write_text(
            json.dumps(delta, indent=2), encoding="utf-8"
        )
    return cfg.out_dir


def _load_pipeline(pipeline_id: str):  # pragma: no cover (heavy)
    from pyannote.audio import Pipeline
    import torch

    pipe = Pipeline.from_pretrained(pipeline_id)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    return pipe


def _run_pipeline(pipeline, audio_path: str):  # pragma: no cover (heavy)
    from src.sd.refine import Segment

    annotation = pipeline(audio_path)
    segs = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        segs.append(Segment(start_s=float(turn.start), end_s=float(turn.end), speaker=str(label)))
    return segs


def _read_rttm_as_pyannote(path: str):  # pragma: no cover (heavy)
    from pyannote.core import Annotation
    from pyannote.core import Segment as PSegment

    ann = Annotation()
    for s in parse_rttm(path):
        ann[PSegment(s.start_s, s.end_s)] = s.speaker
    return ann


def _segments_to_pyannote(rec_id: str, segs):  # pragma: no cover (heavy)
    from pyannote.core import Annotation
    from pyannote.core import Segment as PSegment

    ann = Annotation(uri=rec_id)
    for s in segs:
        ann[PSegment(s.start_s, s.end_s)] = s.speaker
    return ann
