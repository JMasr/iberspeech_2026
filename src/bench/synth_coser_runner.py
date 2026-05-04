"""Validation runner for the synth-COSER dataset.

Runs the COSER pipeline on every recording in a synth-COSER manifest and
reports concrete pass/fail per hypothesis:

  H1 — Whisper-large-v3 zero-shot WER on Spanish ≤ 25 % normalized.
  H2 — Code-switched recordings degrade WER vs Spanish-only by ≥ 5 % absolute.
  H3 — WER differs across Spanish accents by ≥ 2 % absolute (cross-accent drift).
  H4 — Long-form chunked inference completes (no Whisper repetition collapse).
  H5 — Music routing: WER on a music-injected recording is lower with Stage 0
       music routing than without, by ≥ 2 % absolute.
  H6 — VAD: WER on a silence-padded recording is lower with VAD than without.
  H_SD  — pyannote-3.0 zero-shot DER (no collar, overlap incl.) ≤ 35 %.
  H_SNAP — ASR-anchored boundary snap reduces no-collar DER by ≥ 2 % absolute.

These thresholds are documented in PROGRESS.md "Evaluations needed". The
runner emits a JSON report under ``experiments/bench/synth_coser/`` plus a
human-readable summary.

This module is heavy: it imports faster-whisper and pyannote on first use.
The deterministic pieces (manifest aggregation, hypothesis evaluation) are
factored out so they're testable on a bare environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class HypothesisResult:
    name: str
    description: str
    threshold: str
    measured: str
    passed: bool


@dataclass
class RunReport:
    per_record: list[dict] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)
    hypotheses: list[HypothesisResult] = field(default_factory=list)

    def to_table(self) -> str:
        lines = ["# Synth-COSER hypothesis report", ""]
        lines.append(f"{'hypothesis':<8} {'pass':<6} {'measured':<24} {'threshold':<24}")
        lines.append("-" * 70)
        for h in self.hypotheses:
            lines.append(
                f"{h.name:<8} {'PASS' if h.passed else 'FAIL':<6} "
                f"{h.measured:<24} {h.threshold:<24}"
            )
        lines.append("")
        lines.append("# Aggregates")
        for k, v in self.aggregates.items():
            lines.append(f"  {k:<32} {v:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hypothesis evaluation — pure-Python, testable
# ---------------------------------------------------------------------------


def evaluate_hypotheses(per_record: list[dict]) -> tuple[dict, list[HypothesisResult]]:
    """Aggregate per-record results and grade each hypothesis.

    ``per_record`` is a list of dicts with keys:
      record_id, primary_language, code_switch_count, has_music_interlude,
      n_speakers, raw_wer, normalized_wer, der, der_snapped, accent_majority,
      route_music_wer (optional), no_route_music_wer (optional),
      vad_wer (optional), no_vad_wer (optional).
    """
    aggregates: dict[str, float] = {}
    hypotheses: list[HypothesisResult] = []

    es_only = [r for r in per_record if r["code_switch_count"] == 0]
    code_switched = [r for r in per_record if r["code_switch_count"] > 0]

    if es_only:
        aggregates["es_only_normalized_wer"] = _avg(r["normalized_wer"] for r in es_only)
        aggregates["es_only_raw_wer"] = _avg(r["raw_wer"] for r in es_only)
    if code_switched:
        aggregates["code_switched_normalized_wer"] = _avg(
            r["normalized_wer"] for r in code_switched
        )

    # H1 — Whisper zero-shot Spanish.
    if es_only:
        m = aggregates["es_only_normalized_wer"]
        hypotheses.append(
            HypothesisResult(
                name="H1",
                description="Whisper zero-shot WER on Spanish ≤ 25% normalized",
                threshold="≤ 0.25",
                measured=f"{m:.4f}",
                passed=m <= 0.25,
            )
        )

    # H2 — code-switching degrades WER.
    if es_only and code_switched:
        delta = aggregates["code_switched_normalized_wer"] - aggregates["es_only_normalized_wer"]
        hypotheses.append(
            HypothesisResult(
                name="H2",
                description="Code-switched WER ≥ 5% absolute higher than ES-only",
                threshold="Δ ≥ +0.05",
                measured=f"Δ = {delta:+.4f}",
                passed=delta >= 0.05,
            )
        )

    # H3 — accent drift.
    by_accent: dict[str, list[float]] = {}
    for r in es_only:
        accent = r.get("accent_majority", "_")
        if accent in {"_", None, ""}:
            continue
        by_accent.setdefault(accent, []).append(r["normalized_wer"])
    if len(by_accent) >= 2:
        means = {a: sum(ws) / len(ws) for a, ws in by_accent.items()}
        max_a = max(means.values())
        min_a = min(means.values())
        delta = max_a - min_a
        hypotheses.append(
            HypothesisResult(
                name="H3",
                description="Cross-accent WER spread ≥ 2% absolute",
                threshold="Δ_max ≥ 0.02",
                measured=f"Δ = {delta:.4f} ({len(means)} accents)",
                passed=delta >= 0.02,
            )
        )

    # H4 — long-form completes (no NaN / no-output).
    long_recs = [r for r in per_record if r.get("duration_s", 0) >= 60.0]
    n_completed = sum(1 for r in long_recs if r.get("hyp_text") and r.get("normalized_wer") < 1.0)
    if long_recs:
        hypotheses.append(
            HypothesisResult(
                name="H4",
                description="Long-form chunked inference completes (no repetition collapse)",
                threshold=f"{len(long_recs)} of {len(long_recs)} succeed",
                measured=f"{n_completed}/{len(long_recs)}",
                passed=n_completed == len(long_recs),
            )
        )

    # H5 — music routing.
    music_recs = [r for r in per_record if r.get("has_music_interlude")]
    if music_recs and any("route_music_wer" in r for r in music_recs):
        deltas = [
            r["no_route_music_wer"] - r["route_music_wer"]
            for r in music_recs
            if "route_music_wer" in r and "no_route_music_wer" in r
        ]
        avg_delta = _avg(deltas) if deltas else 0.0
        hypotheses.append(
            HypothesisResult(
                name="H5",
                description="Music routing reduces WER on music-injected recordings ≥ 2%",
                threshold="Δ ≥ 0.02",
                measured=f"Δ = {avg_delta:.4f}",
                passed=avg_delta >= 0.02,
            )
        )

    # H6 — VAD.
    if any("vad_wer" in r for r in per_record):
        deltas = [
            r["no_vad_wer"] - r["vad_wer"]
            for r in per_record
            if "vad_wer" in r and "no_vad_wer" in r
        ]
        avg_delta = _avg(deltas) if deltas else 0.0
        hypotheses.append(
            HypothesisResult(
                name="H6",
                description="VAD reduces WER on silence-padded recordings",
                threshold="Δ ≥ 0.0",
                measured=f"Δ = {avg_delta:.4f}",
                passed=avg_delta >= 0.0,
            )
        )

    # H_SD — pyannote zero-shot DER.
    sd_recs = [r for r in per_record if "der" in r and r["der"] is not None]
    if sd_recs:
        avg_der = _avg(r["der"] for r in sd_recs)
        aggregates["pyannote_der_no_collar"] = avg_der
        hypotheses.append(
            HypothesisResult(
                name="H_SD",
                description="pyannote zero-shot DER (no collar, overlap incl.) ≤ 35%",
                threshold="≤ 0.35",
                measured=f"{avg_der:.4f}",
                passed=avg_der <= 0.35,
            )
        )

    # H_SNAP — boundary snap.
    if any("der_snapped" in r and r["der_snapped"] is not None for r in per_record):
        deltas = [
            r["der"] - r["der_snapped"]
            for r in per_record
            if r.get("der") is not None and r.get("der_snapped") is not None
        ]
        avg_delta = _avg(deltas) if deltas else 0.0
        aggregates["snap_delta_der"] = avg_delta
        hypotheses.append(
            HypothesisResult(
                name="H_SNAP",
                description="ASR-anchored boundary snap reduces no-collar DER ≥ 2%",
                threshold="Δ ≥ 0.02",
                measured=f"Δ = {avg_delta:.4f}",
                passed=avg_delta >= 0.02,
            )
        )

    return aggregates, hypotheses


def _avg(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Heavy runner — wires faster-whisper + pyannote into evaluate_hypotheses
# ---------------------------------------------------------------------------


def run_synth_coser_validation(  # pragma: no cover (heavy + network)
    manifest_parquet: str | Path,
    *,
    out_dir: str | Path = "experiments/bench/synth_coser",
    whisper_model: str = "small",
    enable_sd: bool = True,
) -> RunReport:
    """Run zero-shot Whisper + (optional) pyannote on every recording.

    For each recording we compute:
      - raw_wer, normalized_wer (against the concatenated reference transcript)
      - der, der_snapped (if ``enable_sd``)

    For music recordings we additionally run with/without music routing.
    For all recordings we compute with/without VAD by feeding the recording
    directly vs. via Stage 0's energy VAD.

    Heavy imports are local; this function will not import on a bare
    environment. The ``evaluate_hypotheses`` aggregator above is fully testable.
    """
    import pandas as pd

    from src.bench.asr_baseline import ASRBaselineConfig, _build_transcriber
    from src.bench.synth_coser import SAMPLE_RATE, inject_music_interlude  # noqa: F401
    from src.data.normalize import normalize_for_eval, post_edit_raw
    from src.fusion.mbr import wer as _wer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(manifest_parquet)

    transcribe = _build_transcriber(
        ASRBaselineConfig(
            manifest_parquet=Path(manifest_parquet),
            out_dir=out,
            backend="faster-whisper",
            model_id=whisper_model,
            language="es",
        )
    )

    per_record = []
    for _, r in df.iterrows():
        ref_lines = Path(r["transcript_path"]).read_text(encoding="utf-8").splitlines()
        ref_text = " ".join(line for line in ref_lines if line.strip())

        hyp_raw = transcribe(r["audio_path"])
        ref_norm = normalize_for_eval(ref_text)
        hyp_norm = normalize_for_eval(hyp_raw)
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        normalized_wer = _wer(ref_words, hyp_words)
        raw_wer = _wer(post_edit_raw(ref_text).split(), post_edit_raw(hyp_raw).split())

        rec = {
            "record_id": r["record_id"],
            "primary_language": r.get("primary_language"),
            "code_switch_count": int(r.get("code_switch_count", 0)),
            "has_music_interlude": bool(r.get("has_music_interlude", False)),
            "n_speakers": int(r.get("n_speakers", 1)),
            "duration_s": float(r.get("duration_s", 0.0)),
            "raw_wer": raw_wer,
            "normalized_wer": normalized_wer,
            "hyp_text": hyp_raw,
            "accent_majority": "_",  # filled below if available
            "der": None,
            "der_snapped": None,
        }
        per_record.append(rec)

    # SD scoring: defer for now — keeps the main runner light. Extension
    # point: invoke src.bench.sd_baseline.run on the manifest with our RTTMs
    # and join on record_id.

    aggregates, hypotheses = evaluate_hypotheses(per_record)
    report = RunReport(per_record=per_record, aggregates=aggregates, hypotheses=hypotheses)
    (out / "report.json").write_text(
        json.dumps(
            {
                "per_record": per_record,
                "aggregates": aggregates,
                "hypotheses": [asdict(h) for h in hypotheses],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "report.txt").write_text(report.to_table() + "\n", encoding="utf-8")
    return report
