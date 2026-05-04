"""Synthetic end-to-end test exercising the full ASR + SD pipeline wiring.

No GPU, no internet, no heavy deps. We:

1. Build a fake "recording" of 3 chunks with overlapping windows.
2. Stand up two stub recognizers: ``StubWhisper`` (returns canned n-best per
   chunk) and ``StubXLSR`` (returns canned 1-best per chunk).
3. Run per-chunk MBR + ROVER fusion (real code paths) → words per chunk.
4. Run word-LCS overlap merge → unified word stream.
5. Apply non-speech mask using a synthetic interval.
6. Score the recovered transcript against a known reference using meeteval-free
   pure-Python WER.

7. Build a fake diarization with mis-aligned boundaries; verify the boundary
   snap moves them to the ASR word edges; score DER (no collar) before/after
   using a tiny in-tree DER (no pyannote.metrics) to confirm the snap reduces
   DER.

Each step reuses the production functions in ``src/`` (no monkey-patches);
this is a real end-to-end smoke test of every deterministic piece.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.asr.infer_long import ChunkRecognition, merge_overlapping
from src.asr.nonspeech_mask import WordSpan, apply_mask
from src.data.normalize import normalize_for_eval
from src.eval.leaderboard import build_submission
from src.fusion.mbr import NBestEntry, mbr, wer
from src.fusion.rover import from_words, rover
from src.sd.refine import Segment, cleanup, snap_boundaries, write_rttm


@dataclass
class SyntheticReport:
    asr_raw_wer: float
    asr_normalized_wer: float
    der_baseline: float
    der_snapped: float
    snap_delta: float
    submission_zip: Path


def run_synthetic_e2e(out_dir: str | Path = "experiments/bench/synthetic") -> SyntheticReport:
    """Run the full synthetic harness; return a report.

    All numbers are deterministic — the test in ``tests/test_bench_synthetic.py``
    asserts exact values.
    """
    out = Path(out_dir)
    asr_dir = out / "asr"
    sd_dir = out / "sd"
    asr_dir.mkdir(parents=True, exist_ok=True)
    sd_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) ASR per-chunk fusion (MBR over n-best, ROVER vs 2nd voter).
    # ------------------------------------------------------------------
    chunks = _stub_chunks()
    chunk_recs = []
    for chunk_id, start_s, end_s, whisper_nbest, xlsr_words in chunks:
        # MBR on Whisper n-best.
        mbr_text = mbr(whisper_nbest).text
        # ROVER vs XLS-R 1-best.
        fused = rover([from_words(mbr_text.split()), from_words(xlsr_words)])
        # Synthesize word timestamps uniformly within the chunk.
        words = _synthesize_word_spans(fused, start_s, end_s)
        chunk_recs.append(
            ChunkRecognition(chunk_id=chunk_id, start_s=start_s, end_s=end_s, words=words)
        )

    # ------------------------------------------------------------------
    # 2) Long-form merge with word-LCS in the overlap.
    # ------------------------------------------------------------------
    merged_words = merge_overlapping(chunk_recs)

    # ------------------------------------------------------------------
    # 3) Non-speech mask: drop tokens midpointing into a known laughter span.
    # ------------------------------------------------------------------
    nonspeech = [(8.5, 9.5, "laughter")]
    masked = apply_mask(merged_words, nonspeech)

    # ------------------------------------------------------------------
    # 4) Score against the canonical reference.
    # ------------------------------------------------------------------
    hyp_text = " ".join(w.word for w in masked)
    ref_text = "buenas tardes señora cómo está usted hoy en el pueblo"
    raw_wer = wer(ref_text.split(), hyp_text.split())
    norm_wer = wer(normalize_for_eval(ref_text).split(), normalize_for_eval(hyp_text).split())

    # Write the ASR transcript so the submission packager has something.
    rec_id = "synrec_001"
    transcript_path = asr_dir / f"{rec_id}_fullaudio_transcrip.txt"
    transcript_path.write_text(hyp_text + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 5) SD: build a reference + a misaligned hypothesis, then snap & re-score.
    # ------------------------------------------------------------------
    ref_segs = [
        Segment(0.00, 5.00, "SPEAKER_00"),
        Segment(5.00, 12.00, "SPEAKER_01"),
        Segment(12.00, 18.00, "SPEAKER_00"),
    ]
    # Hypothesis is misaligned by ~150ms at every boundary.
    hyp_segs = [
        Segment(0.00, 4.85, "SPEAKER_00"),
        Segment(4.85, 11.85, "SPEAKER_01"),
        Segment(11.85, 18.00, "SPEAKER_00"),
    ]
    # ASR word edges (from `masked`) anchor the snap.
    word_edges = [w.start_s for w in masked] + [w.end_s for w in masked]
    snapped = snap_boundaries(hyp_segs, word_edges, radius_s=0.20)
    snapped = cleanup(snapped, file_duration_s=18.0)
    write_rttm(rec_id, ref_segs, sd_dir / f"{rec_id}_ref.rttm")
    write_rttm(rec_id, hyp_segs, sd_dir / f"{rec_id}_baseline.rttm")
    write_rttm(rec_id, snapped, sd_dir / f"{rec_id}.rttm")

    der_baseline = _no_collar_der(ref_segs, hyp_segs, file_dur=18.0)
    der_snapped = _no_collar_der(ref_segs, snapped, file_dur=18.0)

    # ------------------------------------------------------------------
    # 6) Build a (dry-run) submission to exercise the packager.
    # ------------------------------------------------------------------
    plan = build_submission(
        track="asr",
        hyp_dir=asr_dir,
        out_dir=out / "submission",
        group_id="UVigoBalideaBench",
    )
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "asr_raw_wer": raw_wer,
                "asr_normalized_wer": norm_wer,
                "der_baseline_no_collar": der_baseline,
                "der_snapped_no_collar": der_snapped,
                "snap_delta": der_snapped - der_baseline,
                "submission_zip": str(plan.out_zip),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SyntheticReport(
        asr_raw_wer=raw_wer,
        asr_normalized_wer=norm_wer,
        der_baseline=der_baseline,
        der_snapped=der_snapped,
        snap_delta=der_snapped - der_baseline,
        submission_zip=plan.out_zip,
    )


# ---------------------------------------------------------------------------
# Stubs and helpers — kept simple so the test asserts are stable.
# ---------------------------------------------------------------------------


def _stub_chunks():
    """Return a list of (chunk_id, start_s, end_s, whisper_nbest, xlsr_words).

    Three overlapping chunks. Whisper gets the right answer most of the time;
    XLS-R disagrees at a single slot per chunk; ROVER consensus picks Whisper.
    The 2nd chunk contains a [risas] inside the laughter mask region (8.5-9.5)
    that should be dropped from the final hypothesis.
    """
    chunks = [
        (
            "c00",
            0.0,
            5.0,
            [
                NBestEntry("buenas tardes señora", -0.10),
                NBestEntry("buenas tardes señora", -0.20),
                NBestEntry("buenas tardez señora", -0.40),
            ],
            ["buenas", "tardes", "señorita"],
        ),
        (
            "c01",
            4.0,
            12.0,
            [
                NBestEntry("señora cómo está [risas] usted hoy", -0.15),
                NBestEntry("señora cómo está [risas] usted hoy", -0.20),
                NBestEntry("señor cómo está [risas] usted hoy", -0.45),
            ],
            ["señora", "cómo", "está", "[risas]", "usted", "hoy"],
        ),
        (
            "c02",
            11.0,
            18.0,
            [
                NBestEntry("hoy en el pueblo", -0.10),
                NBestEntry("hoy en el pueblo", -0.18),
                NBestEntry("hoy en pueblo", -0.40),
            ],
            ["hoy", "en", "él", "pueblo"],
        ),
    ]
    return chunks


def _synthesize_word_spans(words: list[str], start_s: float, end_s: float) -> list[WordSpan]:
    """Spread words evenly across [start, end]."""
    if not words:
        return []
    dur = (end_s - start_s) / len(words)
    return [
        WordSpan(word=w, start_s=start_s + i * dur, end_s=start_s + (i + 1) * dur)
        for i, w in enumerate(words)
    ]


def _no_collar_der(ref: list[Segment], hyp: list[Segment], file_dur: float) -> float:
    """Tiny in-tree DER computation (no collar, overlap-included).

    DER = (false_alarm + missed + speaker_confusion) / total_speaker_time.

    We discretize time at 10ms and label each frame with the dominant speaker
    (the lower-id one when overlap exists). This produces stable, deterministic
    numbers good enough for the synthetic harness; the production scorer uses
    pyannote.metrics.
    """
    step = 0.010
    n = int(round(file_dur / step))

    def _frame_speakers(segs):
        out = [set() for _ in range(n)]
        for s in segs:
            a = max(0, int(round(s.start_s / step)))
            b = min(n, int(round(s.end_s / step)))
            for i in range(a, b):
                out[i].add(s.speaker)
        return out

    ref_frames = _frame_speakers(ref)
    hyp_frames = _frame_speakers(hyp)
    total_ref_time = sum(len(s) for s in ref_frames) * step
    if total_ref_time <= 0:
        return 0.0

    err = 0.0
    for r_set, h_set in zip(ref_frames, hyp_frames):
        if not r_set and not h_set:
            continue
        # missed: ref speakers not in hyp
        # false alarm: hyp speakers not in ref
        # confusion: ref ∩ hyp empty when both non-empty
        if r_set and not h_set:
            err += step * len(r_set)
        elif h_set and not r_set:
            err += step * len(h_set)
        else:
            # paired: count missed + false_alarm + confusion conservatively
            missed = len(r_set - h_set) * step
            falarm = len(h_set - r_set) * step
            err += missed + falarm
    return err / total_ref_time
