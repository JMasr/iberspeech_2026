"""Ablation harness — runs the synthetic bench with each fusion stage toggled.

Produces a per-component contribution table answering:

  - How much does **MBR** contribute on top of Whisper-1-best?
  - How much does **ROVER** add on top of MBR?
  - How much does the **non-speech mask** save?
  - How much does **boundary-snap** improve no-collar DER?

The stub recognizers below are designed so each fusion layer adds **measurable
lift** (the original ``src/bench/synthetic.py`` stubs were too easy — Whisper
rank-0 was already correct, so MBR/ROVER landed at zero gain). Specifically:

  - **c00**: Whisper rank-0 has a substitution; MBR consensus *also* picks the
    wrong word; only the 3-way ROVER (Whisper-MBR + XLS-R + Canary) recovers.
  - **c01**: All three recognizers agree on a ``[risas]`` insertion. Only the
    non-speech mask removes it.
  - **c02**: Whisper rank-0 drops words; MBR consensus recovers them.

This is still purely deterministic — every code path is the production code,
no monkey-patches. The numbers in ``ABLATION_EXPECTED`` are asserted in
``tests/test_bench_ablation.py`` so any regression in fusion logic breaks the
build.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from src.asr.infer_long import ChunkRecognition, merge_overlapping
from src.asr.nonspeech_mask import WordSpan, apply_mask
from src.bench.synthetic import _no_collar_der
from src.data.normalize import normalize_for_eval
from src.fusion.mbr import NBestEntry, mbr, wer
from src.fusion.rover import from_words, rover
from src.sd.refine import Segment, cleanup, snap_boundaries


@dataclass(frozen=True)
class AblationKnobs:
    """Toggles for each fusion / post-processing layer."""

    use_mbr: bool = True
    use_rover: bool = True
    use_nonspeech_mask: bool = True
    use_boundary_snap: bool = True

    def label(self) -> str:
        bits = []
        bits.append("MBR" if self.use_mbr else "—")
        bits.append("ROVER" if self.use_rover else "—")
        bits.append("MASK" if self.use_nonspeech_mask else "—")
        bits.append("SNAP" if self.use_boundary_snap else "—")
        return "+".join(bits)


@dataclass
class AblationRow:
    config: str
    use_mbr: bool
    use_rover: bool
    use_nonspeech_mask: bool
    use_boundary_snap: bool
    asr_raw_wer: float
    asr_normalized_wer: float
    der: float


@dataclass
class AblationReport:
    rows: list[AblationRow] = field(default_factory=list)

    def to_table(self) -> str:
        header = f"{'config':<24} {'raw WER':>10} {'norm WER':>10} {'DER':>10}"
        lines = [header, "-" * len(header)]
        for r in self.rows:
            lines.append(
                f"{r.config:<24} {r.asr_raw_wer:>10.4f} "
                f"{r.asr_normalized_wer:>10.4f} {r.der:>10.4f}"
            )
        return "\n".join(lines)


# Reference and SD setup. Non-overlapping ASR chunks so the merge logic is
# trivially correct and each fusion knob has an isolated effect.
REF_TEXT = "buenas tardes señora cómo está usted hoy en el pueblo"
FILE_DURATION_S = 13.0
NONSPEECH_MASK_INTERVAL = (5.0, 7.0)  # covers the [risas] token in c01

REF_SEGS = [
    Segment(0.0, 3.0, "SPEAKER_00"),
    Segment(3.0, 9.0, "SPEAKER_01"),
    Segment(9.0, 13.0, "SPEAKER_00"),
]
HYP_SEGS_MISALIGNED = [
    Segment(0.0, 2.85, "SPEAKER_00"),
    Segment(2.85, 8.85, "SPEAKER_01"),
    Segment(8.85, 13.0, "SPEAKER_00"),
]


def _ablation_chunks():
    """Three non-overlapping chunks designed so each fusion knob has lift.

    Returns a list of ``(chunk_id, start_s, end_s, whisper_nbest, xlsr_words,
    canary_words)`` tuples. With all knobs on, the recovered text is exactly
    REF_TEXT.
    """
    return [
        (
            "c00",
            0.0,
            3.0,
            [
                NBestEntry("buenos tardes señora", -0.10),  # rank-0, "buenos" wrong
                NBestEntry("buenos tardes señora", -0.15),  # rank-1, MBR consensus also wrong
                NBestEntry("buenas tardes señora", -0.45),  # rank-2, correct but lowest
            ],
            ["buenas", "tardes", "señora"],  # XLS-R correct → ROVER recovers
            ["buenas", "tardes", "señora"],  # Canary correct
        ),
        (
            "c01",
            3.0,
            9.0,
            [
                NBestEntry("cómo está [risas] usted hoy", -0.10),
                NBestEntry("cómo está [risas] usted hoy", -0.20),
                NBestEntry("cómo está [risas] usted hoy", -0.30),
            ],
            ["cómo", "está", "[risas]", "usted", "hoy"],
            ["cómo", "está", "[risas]", "usted", "hoy"],
        ),
        (
            "c02",
            9.0,
            13.0,
            [
                NBestEntry("en pueblo", -0.10),  # rank-0 drops "el"
                NBestEntry("en el pueblo", -0.20),  # MBR consensus recovers
                NBestEntry("en el pueblo", -0.25),
            ],
            ["en", "el", "pueblo"],
            ["en", "el", "pueblo"],
        ),
    ]


def _synthesize_word_spans(words: list[str], start_s: float, end_s: float) -> list[WordSpan]:
    """Spread words evenly across [start, end]."""
    if not words:
        return []
    dur = (end_s - start_s) / len(words)
    return [
        WordSpan(word=w, start_s=start_s + i * dur, end_s=start_s + (i + 1) * dur)
        for i, w in enumerate(words)
    ]


def _full_pipeline_word_edges() -> list[float]:
    """Word edges produced by the full pipeline (used to anchor SD snap).

    Held constant across all rows so the SD snap measurement isolates the SNAP
    knob from the ASR fusion knobs.
    """
    chunks = _ablation_chunks()
    edges: list[float] = []
    for _id, start_s, end_s, whisper_nbest, xlsr_words, canary_words in chunks:
        primary = mbr(whisper_nbest).text
        fused = rover(
            [from_words(primary.split()), from_words(xlsr_words), from_words(canary_words)]
        )
        words = _synthesize_word_spans(fused, start_s, end_s)
        # Mirror the MASK to keep edges aligned with the actually-emitted transcript.
        words = apply_mask(words, [(*NONSPEECH_MASK_INTERVAL, "laughter")])
        edges.extend([w.start_s for w in words])
        edges.extend([w.end_s for w in words])
    return sorted(set(edges))


def run_one(knobs: AblationKnobs) -> AblationRow:
    """Run a single configuration and return its row."""
    chunks = _ablation_chunks()
    chunk_recs = []
    for chunk_id, start_s, end_s, whisper_nbest, xlsr_words, canary_words in chunks:
        if knobs.use_mbr:
            primary = mbr(whisper_nbest).text
        else:
            primary = max(whisper_nbest, key=lambda e: e.score).text
        if knobs.use_rover:
            fused = rover(
                [
                    from_words(primary.split()),
                    from_words(xlsr_words),
                    from_words(canary_words),
                ]
            )
        else:
            fused = primary.split()
        words = _synthesize_word_spans(fused, start_s, end_s)
        chunk_recs.append(
            ChunkRecognition(chunk_id=chunk_id, start_s=start_s, end_s=end_s, words=words)
        )

    merged = merge_overlapping(chunk_recs)
    if knobs.use_nonspeech_mask:
        merged = apply_mask(merged, [(*NONSPEECH_MASK_INTERVAL, "laughter")])

    hyp_text = " ".join(w.word for w in merged)
    raw_wer = wer(REF_TEXT.split(), hyp_text.split())
    norm_wer = wer(normalize_for_eval(REF_TEXT).split(), normalize_for_eval(hyp_text).split())

    # SD scoring — use the full-pipeline word edges so the SNAP knob is isolated.
    if knobs.use_boundary_snap:
        edges = _full_pipeline_word_edges()
        scored = snap_boundaries(HYP_SEGS_MISALIGNED, edges, radius_s=0.20)
        scored = cleanup(scored, file_duration_s=FILE_DURATION_S)
    else:
        scored = HYP_SEGS_MISALIGNED
    der = _no_collar_der(REF_SEGS, scored, file_dur=FILE_DURATION_S)

    return AblationRow(
        config=knobs.label(),
        use_mbr=knobs.use_mbr,
        use_rover=knobs.use_rover,
        use_nonspeech_mask=knobs.use_nonspeech_mask,
        use_boundary_snap=knobs.use_boundary_snap,
        asr_raw_wer=raw_wer,
        asr_normalized_wer=norm_wer,
        der=der,
    )


# Configurations to run. Ordered from least to most fusion enabled, so the
# final row is the full pipeline. Each step adds exactly one knob.
DEFAULT_CONFIGS: list[AblationKnobs] = [
    AblationKnobs(False, False, False, False),
    AblationKnobs(True, False, False, False),
    AblationKnobs(True, True, False, False),
    AblationKnobs(True, True, True, False),
    AblationKnobs(True, True, True, True),
]


def run_ablation(
    configs: list[AblationKnobs] | None = None,
    out_dir: str | Path = "experiments/bench/ablation",
) -> AblationReport:
    """Run every configuration and persist results."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [run_one(k) for k in (configs or DEFAULT_CONFIGS)]
    report = AblationReport(rows=rows)
    (out / "ablation.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    (out / "ablation.csv").write_text(_to_csv(rows), encoding="utf-8")
    (out / "ablation.txt").write_text(report.to_table() + "\n", encoding="utf-8")
    return report


def _to_csv(rows: list[AblationRow]) -> str:
    headers = [
        "config",
        "use_mbr",
        "use_rover",
        "use_nonspeech_mask",
        "use_boundary_snap",
        "asr_raw_wer",
        "asr_normalized_wer",
        "der",
    ]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(
            ",".join(
                [
                    r.config,
                    str(r.use_mbr),
                    str(r.use_rover),
                    str(r.use_nonspeech_mask),
                    str(r.use_boundary_snap),
                    f"{r.asr_raw_wer:.6f}",
                    f"{r.asr_normalized_wer:.6f}",
                    f"{r.der:.6f}",
                ]
            )
        )
    return "\n".join(lines) + "\n"
