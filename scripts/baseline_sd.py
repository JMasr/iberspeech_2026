"""COSER SD zero-shot baseline (pyannote-3.1, no collar, overlap included).

Runs ``pyannote/speaker-diarization-3.1`` on the 8 full-length recordings in
``data/raw/data_SD_track/train_dev/audio/`` and scores the resulting RTTM
hypotheses against the reference RTTM directory produced by
``scripts/coser_labels_to_rttm.py``.

Outputs to ``experiments/baseline_sd/<pipeline>/``:
  - ``hyp_rttm/<rec>.rttm`` — diarization hypotheses (post-cleanup)
  - ``summary.json``       — per-record RTF + segment count + per-record DER
                              + aggregate DER/miss/false-alarm/confusion

Run:
  python scripts/baseline_sd.py
  python scripts/baseline_sd.py --pipeline-id pyannote/speaker-diarization-3.1

Pre-requisites:
  uv pip install --python .venv/bin/python "pyannote.audio>=3.1" "pyannote.metrics>=3.2"
  huggingface-cli login --token "$HF_TOKEN"     # terms accepted on the gated repo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.score_sd import score_directory
from src.sd.refine import Segment, cleanup, write_rttm

DATA_SD = ROOT / "data/raw/data_SD_track/train_dev"


def run_pipeline(pipeline, audio_path: Path) -> list[Segment]:
    """Run pyannote diarization on one audio file; return raw Segment list."""
    annotation = pipeline(str(audio_path))
    return [
        Segment(start_s=float(turn.start), end_s=float(turn.end), speaker=str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def file_duration_s(audio_path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline-id",
        default="pyannote/speaker-diarization-3.1",
        help="HF model ID for the diarization pipeline.",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=DATA_SD / "audio",
        help="Directory of full-length .wav files to diarize.",
    )
    ap.add_argument(
        "--ref-rttm",
        type=Path,
        default=ROOT / "data/interim/sd_ref_rttm",
        help="Directory of reference <rec>.rttm files (output of coser_labels_to_rttm.py).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to experiments/baseline_sd/<safe_pipeline_id>/.",
    )
    ap.add_argument(
        "--rec-ids",
        default=None,
        help="Comma-separated record IDs to keep (e.g. COSER-3228-01). Default: all.",
    )
    args = ap.parse_args()
    rec_filter = (
        {r.strip() for r in args.rec_ids.split(",") if r.strip()} if args.rec_ids else None
    )

    out_dir = args.out or (
        ROOT / "experiments" / "baseline_sd" / args.pipeline_id.replace("/", "_")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir = out_dir / "hyp_rttm"
    hyp_dir.mkdir(exist_ok=True)

    print(f"[setup] pipeline    = {args.pipeline_id}")
    print(f"[setup] audio dir   = {args.audio_dir}")
    print(f"[setup] ref RTTM    = {args.ref_rttm}")
    print(f"[setup] out dir     = {out_dir}")

    if not args.ref_rttm.exists() or not any(args.ref_rttm.glob("*.rttm")):
        raise SystemExit(
            f"no reference RTTM found under {args.ref_rttm}. Run "
            "scripts/coser_labels_to_rttm.py first."
        )

    from pyannote.audio import Pipeline
    import torch

    print(f"[setup] loading {args.pipeline_id} …")
    pipeline = Pipeline.from_pretrained(args.pipeline_id)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("[setup] on CUDA")
    else:
        print("[setup] on CPU (will be slow)")

    summary: dict = {
        "pipeline": args.pipeline_id,
        "per_record": {},
    }

    audio_paths = sorted(args.audio_dir.glob("*.wav"))
    if rec_filter is not None:
        audio_paths = [p for p in audio_paths if p.stem in rec_filter]
    print(f"\n[sd] {len(audio_paths)} full audios")

    for i, wav in enumerate(audio_paths, 1):
        rec_id = wav.stem
        print(f"[sd] [{i}/{len(audio_paths)}] {rec_id} …", flush=True)
        try:
            t0 = time.time()
            segs = run_pipeline(pipeline, wav)
            dur = file_duration_s(wav)
            segs = cleanup(segs, file_duration_s=dur)
            write_rttm(rec_id, segs, hyp_dir / f"{rec_id}.rttm")
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        rtf = elapsed / max(dur, 1e-6)
        n_spk = len({s.speaker for s in segs})
        summary["per_record"][rec_id] = {
            "duration_s": dur,
            "elapsed_s": elapsed,
            "rtf": rtf,
            "n_segments": len(segs),
            "n_speakers_hyp": n_spk,
        }
        print(
            f"  dur={dur/60:.1f}min  RTF={rtf:.2f}  segs={len(segs)}  spk_hyp={n_spk}",
            flush=True,
        )

    print("\n[score] computing DER against reference RTTM …")
    result = score_directory(hyp_dir, args.ref_rttm)
    summary["aggregate"] = {
        "der": result.der,
        "miss": result.miss,
        "false_alarm": result.false_alarm,
        "confusion": result.confusion,
        "n_records": result.n_records,
    }
    summary["per_record_der"] = result.per_record

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\n[done] DER={result.der*100:.2f}%  miss={result.miss*100:.2f}%  "
        f"fa={result.false_alarm*100:.2f}%  conf={result.confusion*100:.2f}%  "
        f"n={result.n_records}"
    )
    print(f"[done] summary → {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
