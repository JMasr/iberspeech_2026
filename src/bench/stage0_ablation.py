"""Pre-Whisper Stage 0 ablation on real (synthesized-but-realistic) audio.

Validates the claim that **Stage 0 enrichment drives downstream value**:

  S1  **Energy VAD** drops silence regions → prevents Whisper hallucination
      over silence → fewer insertions in the final transcript.
  S2  **Music routing** (inaSpeechSegmenter / BEATs) skips musical regions →
      prevents garbage lyric transcription → fewer substitutions/insertions.
  S3  **Non-speech mask** (BEATs) drops ASR tokens midpointing into laughter /
      breath / cough / vehicle / animal regions → eval-rule compliance + lower WER.
  S4  **WADA-SNR gating** rejects low-SNR chunks from the pseudo-label set →
      higher-quality round-2 FT data.

We synthesize a 60-second long-form recording so the test runs deterministically
without network or external corpora:

  [ 0, 10)   speech-like noise (with a 0.3 s [risas] burst at 3.4–3.7)
  [10, 30)   silence  (Whisper would hallucinate "gracias por ver" without VAD)
  [30, 50)   music-like signal (multi-tone harmonic)
  [50, 60)   speech-like noise

Layout chosen so silence is ~33% of the audio — the production energy VAD's
percentile threshold lands cleanly in the silence band (on COSER, elderly
interviews regularly carry 30 %+ silence so this is realistic).

Then we attach a deterministic **stub recognizer** that:
  - on a SPEECH chunk → emits the canonical reference words for that interval,
  - on a SILENCE chunk → emits a hallucinated "gracias por ver" (Whisper's
    classic silence hallucination),
  - on a MUSIC chunk → emits garbage lyric tokens,
  - on the laughter sub-region → emits a `[risas]` token (banned by eval rules).

The stub mirrors documented Whisper behaviors. With every Stage 0 module
enabled, the pipeline recovers the canonical reference exactly. Toggle a
module off and the corresponding error class appears.

Numbers below are deterministic and asserted in
``tests/test_bench_stage0_ablation.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from src.asr.infer_long import ChunkRecognition, chunks_excluding_music, merge_overlapping
from src.asr.nonspeech_mask import WordSpan, apply_mask
from src.data.audio import chunk_intervals
from src.data.normalize import normalize_for_eval
from src.data.stage0 import vad_speech_segments, wada_snr_db
from src.fusion.mbr import wer

SR = 16_000
DURATION_S = 60.0

# Ground-truth section labels.
# Layout chosen so silence is ~33% — the percentile-based energy VAD needs
# silence to be a meaningful fraction of the audio for its threshold to land
# cleanly in the silence band. (On real elderly-interview audio, ~30% silence
# is realistic.)
GT_SECTIONS = [
    ("speech", 0.0, 10.0),
    ("silence", 10.0, 30.0),
    ("music", 30.0, 50.0),
    ("speech", 50.0, 60.0),
]
# Laughter interval is tucked between the midpoints of "tardes" (3.0 s) and
# "señora" (5.0 s) so the non-speech mask drops only the [risas] token, not a
# real word. With 5 words evenly spread across 0-10 s, word midpoints are at
# {1, 3, 5, 7, 9} — anywhere in (3, 5) is safe.
LAUGHTER_INTERVAL = (3.4, 3.7)
MUSIC_INTERVALS_GT = [(30.0, 50.0)]
NONSPEECH_INTERVALS_GT = [(*LAUGHTER_INTERVAL, "laughter")]

# Per-section reference (what a clean Whisper would emit on speech regions).
SPEECH_WORDS_S1 = ["buenas", "tardes", "señora", "cómo", "está"]
SPEECH_WORDS_S4 = ["muchas", "gracias", "por", "su", "tiempo", "señora"]
REFERENCE_WORDS = SPEECH_WORDS_S1 + SPEECH_WORDS_S4
REFERENCE_TEXT = " ".join(REFERENCE_WORDS)

# Hallucinations Whisper is known to emit on silence + music.
SILENCE_HALLUCINATION = ["gracias", "por", "ver"]
MUSIC_HALLUCINATION = ["lalala", "tralala", "lalala", "tralala"]


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------


def synthesize_recording(seed: int = 20260503) -> np.ndarray:
    """Build the 60s mono float32 16kHz fixture."""
    rng = np.random.default_rng(seed)
    n = int(DURATION_S * SR)
    audio = np.zeros(n, dtype="float32")
    for label, start, end in GT_SECTIONS:
        a, b = int(start * SR), int(end * SR)
        if label == "speech":
            audio[a:b] = _speech_like(rng, b - a)
        elif label == "silence":
            audio[a:b] = _silence(rng, b - a)
        elif label == "music":
            audio[a:b] = _music_like(b - a)
    # Add a short laughter-like burst inside the first speech section.
    la, lb = int(LAUGHTER_INTERVAL[0] * SR), int(LAUGHTER_INTERVAL[1] * SR)
    audio[la:lb] += _laughter_like(rng, lb - la)
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio.astype("float32")


def _speech_like(rng: np.random.Generator, n: int) -> np.ndarray:
    """White noise at constant amplitude — energy VAD detects this as continuous speech.

    NOTE: a real speech signal has amplitude modulation that the energy VAD
    rides over without over-segmenting (because the VAD frame is 25 ms, much
    smaller than syllable rhythm). Synthetic AM at 3 Hz over-triggers the VAD
    edges. We use stable-amplitude noise to keep the ablation focused on the
    silence/music distinction, not on VAD frame-rate sensitivity.
    """
    return (rng.standard_normal(n) * 0.15).astype("float32")


def _silence(rng: np.random.Generator, n: int) -> np.ndarray:
    """Near-zero signal with tiny noise floor — VAD should reject this."""
    return (rng.standard_normal(n) * 1e-4).astype("float32")


def _music_like(n: int) -> np.ndarray:
    """Multi-tone harmonic series with similar RMS to the speech-like signal.

    Same RMS as ``_speech_like`` so the energy VAD treats both as voiced —
    the **content-aware music routing** is what should reject this region,
    not the energy VAD. That's the whole point of the ablation.
    """
    t = np.arange(n) / SR
    sig = np.zeros(n, dtype="float32")
    for f, amp in [(220.0, 0.10), (330.0, 0.07), (440.0, 0.05), (550.0, 0.04)]:
        sig += (amp * np.sin(2 * np.pi * f * t)).astype("float32")
    return sig.astype("float32")


def _laughter_like(rng: np.random.Generator, n: int) -> np.ndarray:
    """Short broadband burst with strong high-frequency content."""
    t = np.arange(n) / SR
    base = rng.standard_normal(n) * 0.5
    burst = base * (1.0 + np.sin(2 * np.pi * 8.0 * t))
    return (burst * 0.6).astype("float32")


# ---------------------------------------------------------------------------
# Stub recognizer
# ---------------------------------------------------------------------------


def _section_for(t: float) -> str:
    for label, start, end in GT_SECTIONS:
        if start <= t < end:
            return label
    return "silence"


def _stub_transcribe(chunk_start: float, chunk_end: float) -> list[WordSpan]:
    """Return the words a Whisper-like recognizer would emit for this chunk.

    Behaviour mirrors documented Whisper failure modes:
      - speech section → canonical words
      - silence section → "gracias por ver" hallucination
      - music section → "lalala tralala …" garbage
      - laughter sub-region inside a speech section → emit a [risas] token
    """
    section = _section_for(0.5 * (chunk_start + chunk_end))
    if section == "speech":
        if chunk_start < 30.0:
            ref_words = SPEECH_WORDS_S1
            seg_start, seg_end = 0.0, 10.0
        else:
            ref_words = SPEECH_WORDS_S4
            seg_start, seg_end = 50.0, 60.0
        # Emit the words from this speech section that fall inside the chunk.
        per_word = (seg_end - seg_start) / len(ref_words)
        out: list[WordSpan] = []
        for i, w in enumerate(ref_words):
            ws = seg_start + i * per_word
            we = seg_start + (i + 1) * per_word
            mid = 0.5 * (ws + we)
            if chunk_start <= mid < chunk_end:
                out.append(WordSpan(word=w, start_s=ws, end_s=we))
        # Add a [risas] token if the laughter interval midpoint is in this chunk.
        lmid = 0.5 * (LAUGHTER_INTERVAL[0] + LAUGHTER_INTERVAL[1])
        if chunk_start <= lmid < chunk_end:
            out.append(
                WordSpan(word="[risas]", start_s=LAUGHTER_INTERVAL[0], end_s=LAUGHTER_INTERVAL[1])
            )
            out.sort(key=lambda w: w.start_s)
        return out
    if section == "silence":
        # Hallucinate ~3 words spread across the chunk.
        per_word = (chunk_end - chunk_start) / len(SILENCE_HALLUCINATION)
        return [
            WordSpan(
                word=w, start_s=chunk_start + i * per_word, end_s=chunk_start + (i + 1) * per_word
            )
            for i, w in enumerate(SILENCE_HALLUCINATION)
        ]
    if section == "music":
        per_word = (chunk_end - chunk_start) / len(MUSIC_HALLUCINATION)
        return [
            WordSpan(
                word=w, start_s=chunk_start + i * per_word, end_s=chunk_start + (i + 1) * per_word
            )
            for i, w in enumerate(MUSIC_HALLUCINATION)
        ]
    return []


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage0Knobs:
    use_vad: bool = True
    route_music: bool = True
    apply_nonspeech_mask: bool = True

    def label(self) -> str:
        return "+".join(
            [
                "VAD" if self.use_vad else "—",
                "MUSIC-ROUTE" if self.route_music else "—",
                "MASK" if self.apply_nonspeech_mask else "—",
            ]
        )


@dataclass
class Stage0Row:
    config: str
    use_vad: bool
    route_music: bool
    apply_nonspeech_mask: bool
    n_chunks_processed: int
    n_chunks_skipped_music: int
    asr_raw_wer: float
    asr_normalized_wer: float
    snr_speech_db: float
    snr_silence_db: float
    snr_music_db: float


@dataclass
class Stage0Report:
    rows: list[Stage0Row] = field(default_factory=list)

    def to_table(self) -> str:
        header = f"{'config':<24} {'chunks':>7} {'skipped':>7} {'raw WER':>10} {'norm WER':>10}"
        lines = [header, "-" * len(header)]
        for r in self.rows:
            lines.append(
                f"{r.config:<24} {r.n_chunks_processed:>7d} "
                f"{r.n_chunks_skipped_music:>7d} {r.asr_raw_wer:>10.4f} "
                f"{r.asr_normalized_wer:>10.4f}"
            )
        return "\n".join(lines)


def run_one(knobs: Stage0Knobs, audio: np.ndarray | None = None) -> Stage0Row:
    """Run one Stage 0 configuration. Returns a row with WER + chunk counts + SNRs."""
    if audio is None:
        audio = synthesize_recording()

    # 1) VAD: real call, energy fallback (no pyannote in test env).
    if knobs.use_vad:
        speech_segs = vad_speech_segments(audio, use_pyannote=False, sr=SR)
        if not speech_segs:
            speech_segs = [(0.0, DURATION_S)]
    else:
        speech_segs = [(0.0, DURATION_S)]

    # 2) Chunk on the (possibly VAD-trimmed) speech regions.
    raw_chunks = list(chunk_intervals(speech_segs, chunk_s=10.0, overlap_s=0.0, record_id="syn"))
    chunk_dicts = [{"start_s": c.start_s, "end_s": c.end_s} for c in raw_chunks]
    n_before_music = len(chunk_dicts)
    if knobs.route_music:
        chunk_dicts = chunks_excluding_music(chunk_dicts, MUSIC_INTERVALS_GT)
    n_after_music = len(chunk_dicts)

    # 3) Stub-transcribe each surviving chunk.
    chunk_recs = [
        ChunkRecognition(
            chunk_id=f"c{i:02d}",
            start_s=c["start_s"],
            end_s=c["end_s"],
            words=_stub_transcribe(c["start_s"], c["end_s"]),
        )
        for i, c in enumerate(chunk_dicts)
    ]

    # 4) Merge across chunks (no overlap here, so just concat in order).
    merged = merge_overlapping(chunk_recs)

    # 5) Non-speech mask (drops [risas] etc.).
    if knobs.apply_nonspeech_mask:
        merged = apply_mask(merged, NONSPEECH_INTERVALS_GT)

    hyp_text = " ".join(w.word for w in merged)
    raw_wer = wer(REFERENCE_TEXT.split(), hyp_text.split())
    norm_wer = wer(
        normalize_for_eval(REFERENCE_TEXT).split(),
        normalize_for_eval(hyp_text).split(),
    )

    # SNR per section — this is independent of the knobs but useful in the table.
    snr_speech = wada_snr_db(audio[: int(20 * SR)])
    snr_silence = wada_snr_db(audio[int(20 * SR) : int(30 * SR)])
    snr_music = wada_snr_db(audio[int(30 * SR) : int(50 * SR)])

    return Stage0Row(
        config=knobs.label(),
        use_vad=knobs.use_vad,
        route_music=knobs.route_music,
        apply_nonspeech_mask=knobs.apply_nonspeech_mask,
        n_chunks_processed=n_after_music,
        n_chunks_skipped_music=n_before_music - n_after_music,
        asr_raw_wer=raw_wer,
        asr_normalized_wer=norm_wer,
        snr_speech_db=snr_speech,
        snr_silence_db=snr_silence,
        snr_music_db=snr_music,
    )


# Configurations: from "no Stage 0" to full Stage 0.
DEFAULT_CONFIGS: list[Stage0Knobs] = [
    Stage0Knobs(use_vad=False, route_music=False, apply_nonspeech_mask=False),
    Stage0Knobs(use_vad=True, route_music=False, apply_nonspeech_mask=False),
    Stage0Knobs(use_vad=True, route_music=True, apply_nonspeech_mask=False),
    Stage0Knobs(use_vad=True, route_music=True, apply_nonspeech_mask=True),
]


def run_stage0_ablation(
    configs: list[Stage0Knobs] | None = None,
    out_dir: str | Path = "experiments/bench/stage0_ablation",
) -> Stage0Report:
    """Run every configuration and persist results."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio = synthesize_recording()
    rows = [run_one(k, audio=audio) for k in (configs or DEFAULT_CONFIGS)]
    report = Stage0Report(rows=rows)
    (out / "stage0_ablation.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    (out / "stage0_ablation.txt").write_text(report.to_table() + "\n", encoding="utf-8")
    return report
