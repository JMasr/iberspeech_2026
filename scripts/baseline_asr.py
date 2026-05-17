"""COSER ASR zero-shot baseline.

Runs faster-whisper (default: large-v3) on the COSER train/dev partition:
  - 8 full audios → chunked decode → reference = concatenated soft labels.
  - 14 487 short segments → per-segment decode → reference from JSONL.

Scores raw + normalized WER per record using meeteval and writes:
  - experiments/baseline_asr/<model>/per_record.csv
  - experiments/baseline_asr/<model>/per_segment_aggregate.csv
  - experiments/baseline_asr/<model>/hypotheses_long.jsonl
  - experiments/baseline_asr/<model>/hypotheses_segments.jsonl
  - experiments/baseline_asr/<model>/summary.json

Run:
  python scripts/baseline_asr.py --model large-v3
  python scripts/baseline_asr.py --model small --segments-only
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.normalize import clean_coser_reference, normalize_for_eval, post_edit_raw

DATA_ASR = ROOT / "data/raw/data_ASR_track/train_dev"
LABEL_LINE = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([^:]+):\s*(.*)$")


@dataclass
class Hyp:
    record: str
    raw: str
    normalized: str


def parse_soft_labels_concat(label_path: Path) -> tuple[str, str]:
    """Concatenate all turn texts in time order; return (raw, normalized) reference."""
    cleaned_turns = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        m = LABEL_LINE.match(line)
        if not m:
            continue
        cleaned_turns.append(clean_coser_reference(m.group(4)))
    raw_ref = " ".join(t for t in cleaned_turns if t)
    norm_ref = " ".join(normalize_for_eval(t) for t in cleaned_turns if t)
    return post_edit_raw(raw_ref), norm_ref


def wer(ref: str, hyp: str) -> dict:
    """meeteval single-stream WER. Returns dict with WER and counts."""
    from meeteval.wer.wer import siso_word_error_rate

    res = siso_word_error_rate(reference=ref, hypothesis=hyp)
    return {
        "wer": res.error_rate,
        "errors": res.errors,
        "n_ref_words": res.length,
        "ins": res.insertions,
        "del": res.deletions,
        "sub": res.substitutions,
    }


def transcribe_file(model, path: Path, language: str | None) -> tuple[str, dict]:
    """Run faster-whisper on a file; return (text, info_dict)."""
    t0 = time.time()
    segments, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
    )
    text_parts = []
    for s in segments:
        text_parts.append(s.text.strip())
    elapsed = time.time() - t0
    return " ".join(text_parts), {
        "duration_s": info.duration,
        "elapsed_s": elapsed,
        "rtf": elapsed / max(info.duration, 1e-6),
        "lang_detected": info.language,
        "lang_prob": info.language_probability,
    }


def transcribe_segment(model, path: Path, language: str | None) -> tuple[str, dict]:
    """Single short segment (≤30 s) — no VAD filter, decode in one go."""
    t0 = time.time()
    segments, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
    )
    text_parts = [s.text.strip() for s in segments]
    elapsed = time.time() - t0
    return " ".join(text_parts), {
        "duration_s": info.duration,
        "elapsed_s": elapsed,
        "lang_detected": info.language,
        "lang_prob": info.language_probability,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument(
        "--language",
        default=None,
        help="Force decoding language (e.g. 'es'). Default: auto-detect per chunk.",
    )
    ap.add_argument("--long-only", action="store_true", help="Skip the 14k segments.")
    ap.add_argument("--segments-only", action="store_true", help="Skip the 8 long audios.")
    ap.add_argument("--max-segments", type=int, default=None, help="Cap segment count (for smoke).")
    ap.add_argument(
        "--rec-ids",
        default=None,
        help="Comma-separated record IDs to keep (e.g. COSER-3228-01). Default: all.",
    )
    args = ap.parse_args()
    rec_filter = (
        {r.strip() for r in args.rec_ids.split(",") if r.strip()} if args.rec_ids else None
    )

    out_dir = ROOT / "experiments" / "baseline_asr" / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] outputs → {out_dir}")

    print(f"[setup] loading faster-whisper {args.model} on {args.device} ({args.compute_type})…")
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("[setup] model loaded.")

    summary: dict = {"model": args.model, "language": args.language, "long": [], "segments": {}}

    # -----------------------------------------------------------------------
    # LONG AUDIOS
    # -----------------------------------------------------------------------
    if not args.segments_only:
        long_audios = sorted((DATA_ASR / "audio").glob("*.wav"))
        if rec_filter is not None:
            long_audios = [p for p in long_audios if p.stem in rec_filter]
        print(f"\n[long] {len(long_audios)} full audios")
        per_record_rows = []
        long_hyps_path = out_dir / "hypotheses_long.jsonl"
        with long_hyps_path.open("w", encoding="utf-8") as fh:
            for i, audio_path in enumerate(long_audios, 1):
                rec_id = audio_path.stem
                label_path = DATA_ASR / "labels" / f"{rec_id}.txt"
                print(f"[long] [{i}/{len(long_audios)}] {rec_id} …", flush=True)
                try:
                    hyp_text, info = transcribe_file(model, audio_path, args.language)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue
                ref_raw, ref_norm = parse_soft_labels_concat(label_path)
                hyp_raw = post_edit_raw(hyp_text)
                hyp_norm = normalize_for_eval(hyp_text)
                w_raw = wer(ref_raw, hyp_raw)
                w_norm = wer(ref_norm, hyp_norm)
                row = {
                    "record": rec_id,
                    "duration_s": info["duration_s"],
                    "elapsed_s": info["elapsed_s"],
                    "rtf": info["rtf"],
                    "lang_detected": info["lang_detected"],
                    "lang_prob": info["lang_prob"],
                    "n_ref_words_raw": w_raw["n_ref_words"],
                    "wer_raw": w_raw["wer"],
                    "ins_raw": w_raw["ins"],
                    "del_raw": w_raw["del"],
                    "sub_raw": w_raw["sub"],
                    "n_ref_words_norm": w_norm["n_ref_words"],
                    "wer_norm": w_norm["wer"],
                    "ins_norm": w_norm["ins"],
                    "del_norm": w_norm["del"],
                    "sub_norm": w_norm["sub"],
                }
                per_record_rows.append(row)
                fh.write(
                    json.dumps(
                        {"record": rec_id, "hyp": hyp_text, "ref_raw": ref_raw, "ref_norm": ref_norm},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    f"  dur={info['duration_s']/60:.1f}min  "
                    f"RTF={info['rtf']:.2f}  "
                    f"lang={info['lang_detected']}({info['lang_prob']:.2f})  "
                    f"WER raw={w_raw['wer']*100:.1f}%  norm={w_norm['wer']*100:.1f}%",
                    flush=True,
                )
        # CSV
        csv_path = out_dir / "per_record.csv"
        if per_record_rows:
            keys = list(per_record_rows[0].keys())
            with csv_path.open("w", encoding="utf-8") as fh:
                fh.write(",".join(keys) + "\n")
                for r in per_record_rows:
                    fh.write(",".join(f"{r[k]}" for k in keys) + "\n")
        # Aggregate
        if per_record_rows:
            tot_err_raw = sum(r["ins_raw"] + r["del_raw"] + r["sub_raw"] for r in per_record_rows)
            tot_n_raw = sum(r["n_ref_words_raw"] for r in per_record_rows)
            tot_err_norm = sum(r["ins_norm"] + r["del_norm"] + r["sub_norm"] for r in per_record_rows)
            tot_n_norm = sum(r["n_ref_words_norm"] for r in per_record_rows)
            agg_raw = tot_err_raw / max(tot_n_raw, 1)
            agg_norm = tot_err_norm / max(tot_n_norm, 1)
            summary["long"] = {
                "n_records": len(per_record_rows),
                "agg_wer_raw": agg_raw,
                "agg_wer_norm": agg_norm,
                "tot_ref_words_raw": tot_n_raw,
                "tot_ref_words_norm": tot_n_norm,
            }
            print(f"\n[long] aggregate WER raw={agg_raw*100:.2f}%  norm={agg_norm*100:.2f}%")

    # -----------------------------------------------------------------------
    # SEGMENTS
    # -----------------------------------------------------------------------
    if not args.long_only:
        jsonl_path = DATA_ASR / "labels/segments_labels_ASR_track.jsonl"
        seg_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        if rec_filter is not None:
            seg_rows = [r for r in seg_rows if r["audio"] in rec_filter]
        if args.max_segments:
            seg_rows = seg_rows[: args.max_segments]
        print(f"\n[seg] {len(seg_rows)} segments")
        seg_hyps_path = out_dir / "hypotheses_segments.jsonl"
        per_audio_acc: dict[str, dict[str, float]] = defaultdict(
            lambda: {"err_raw": 0.0, "n_raw": 0, "err_norm": 0.0, "n_norm": 0, "n_seg": 0}
        )
        t0 = time.time()
        with seg_hyps_path.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(seg_rows):
                seg_path = DATA_ASR / row["path"]
                if not seg_path.exists():
                    continue
                ref_text = clean_coser_reference(row["text"])
                ref_raw = post_edit_raw(ref_text)
                ref_norm = normalize_for_eval(ref_text)
                try:
                    hyp_text, info = transcribe_segment(model, seg_path, args.language)
                except Exception as e:
                    print(f"  seg {row['path']} FAILED: {e}")
                    continue
                hyp_raw = post_edit_raw(hyp_text)
                hyp_norm = normalize_for_eval(hyp_text)
                w_raw = wer(ref_raw, hyp_raw) if ref_raw else {"errors": 0, "n_ref_words": 0}
                w_norm = wer(ref_norm, hyp_norm) if ref_norm else {"errors": 0, "n_ref_words": 0}
                acc = per_audio_acc[row["audio"]]
                acc["err_raw"] += w_raw["errors"]
                acc["n_raw"] += w_raw["n_ref_words"]
                acc["err_norm"] += w_norm["errors"]
                acc["n_norm"] += w_norm["n_ref_words"]
                acc["n_seg"] += 1
                fh.write(
                    json.dumps(
                        {
                            "path": row["path"],
                            "audio": row["audio"],
                            "ref": row["text"],
                            "hyp": hyp_text,
                            "wer_raw": w_raw.get("wer", 0.0),
                            "wer_norm": w_norm.get("wer", 0.0),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if (i + 1) % 200 == 0:
                    el = time.time() - t0
                    eta = el / (i + 1) * (len(seg_rows) - (i + 1))
                    tot_n_norm_so_far = sum(a["n_norm"] for a in per_audio_acc.values())
                    tot_e_norm_so_far = sum(a["err_norm"] for a in per_audio_acc.values())
                    cur_norm = tot_e_norm_so_far / max(tot_n_norm_so_far, 1)
                    print(
                        f"[seg] {i+1}/{len(seg_rows)}  "
                        f"elapsed={el/60:.1f}min  ETA={eta/60:.1f}min  "
                        f"running WER norm={cur_norm*100:.2f}%",
                        flush=True,
                    )
        # Per-source-record CSV
        csv_path = out_dir / "per_segment_aggregate.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("audio,n_segments,wer_raw,wer_norm,n_ref_words_norm\n")
            for audio, acc in sorted(per_audio_acc.items()):
                w_raw = acc["err_raw"] / max(acc["n_raw"], 1)
                w_norm = acc["err_norm"] / max(acc["n_norm"], 1)
                fh.write(
                    f"{audio},{int(acc['n_seg'])},{w_raw:.4f},{w_norm:.4f},{int(acc['n_norm'])}\n"
                )
        tot_err_raw = sum(a["err_raw"] for a in per_audio_acc.values())
        tot_n_raw = sum(a["n_raw"] for a in per_audio_acc.values())
        tot_err_norm = sum(a["err_norm"] for a in per_audio_acc.values())
        tot_n_norm = sum(a["n_norm"] for a in per_audio_acc.values())
        agg_raw = tot_err_raw / max(tot_n_raw, 1)
        agg_norm = tot_err_norm / max(tot_n_norm, 1)
        summary["segments"] = {
            "n_segments_scored": sum(int(a["n_seg"]) for a in per_audio_acc.values()),
            "n_source_records": len(per_audio_acc),
            "agg_wer_raw": agg_raw,
            "agg_wer_norm": agg_norm,
            "tot_ref_words_raw": tot_n_raw,
            "tot_ref_words_norm": tot_n_norm,
        }
        print(
            f"\n[seg] aggregate WER raw={agg_raw*100:.2f}%  norm={agg_norm*100:.2f}%  "
            f"({summary['segments']['n_segments_scored']} segments)"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[done] summary → {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
