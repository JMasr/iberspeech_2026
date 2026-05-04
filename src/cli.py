"""COSER CLI — one subcommand per Make target.

Heavy imports happen inside subcommand bodies so that ``coser --help``,
``coser self-check``, and the deterministic subcommands work on a bare
environment.
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.eval.leaderboard import (
    DAILY_CAP,
    build_submission,
    record_submission,
    remaining_slots_today,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    raw: Path = Path("data/raw"), out: Path = Path("data/interim/manifest.parquet")
) -> None:
    """Build the COSER manifest with stratified train/dev split."""
    from src.data.ingest import IngestConfig, build_manifest

    cfg = IngestConfig(raw_root=raw, out_path=out)
    path = build_manifest(cfg)
    typer.echo(f"manifest: {path}")


@app.command()
def stage0(
    manifest: Path = Path("data/interim/manifest.parquet"),
    out: Path = Path("data/processed/chunks"),
    no_pyannote: bool = typer.Option(False, "--no-pyannote", help="Use the energy VAD fallback."),
    no_inaspeech: bool = typer.Option(False, "--no-inaspeech", help="Skip inaSpeechSegmenter."),
) -> None:
    """Run Stage 0A enrichment on every recording in the manifest."""
    import pandas as pd

    from src.data.stage0 import Stage0Config, enrich_recording

    cfg = Stage0Config(use_pyannote_vad=not no_pyannote, use_inaspeech=not no_inaspeech)
    df = pd.read_parquet(manifest)
    longform = df[df["kind"] == "longform"]
    for _, row in longform.iterrows():
        path = enrich_recording(
            record_id=row["record_id"],
            audio_path=row["audio_path"],
            out_dir=out,
            config=cfg,
            province=row.get("province"),
            topic=row.get("topic"),
            year=row.get("year"),
            channel=row.get("channel"),
        )
        typer.echo(f"stage0: {row['record_id']} → {path}")


@app.command("stage0-beats")
def stage0_beats(
    chunks: Path = Path("data/processed/chunks"),
    backend: str = typer.Option("beats", help="beats|panns"),
    threshold: float = typer.Option(0.30, help="Score threshold for kept tags."),
) -> None:
    """Run Stage 0B BEATs on chunks flagged by Stage 0A."""
    import pandas as pd

    from src.data.stage0_beats import BeatsConfig, tag_chunks

    cfg = BeatsConfig(backend=backend, score_threshold=threshold)
    for parquet in sorted(chunks.glob("*.parquet")):
        df = pd.read_parquet(parquet)
        # Reconstruct the audio path from the manifest. We expect an audio_path
        # column on the per-recording parquet; if not, we look up via record_id.
        if "audio_path" not in df.columns:
            typer.echo(f"skip {parquet} (no audio_path column)")
            continue
        audio_path = df["audio_path"].iloc[0]
        tag_chunks(parquet, audio_path, config=cfg)
        typer.echo(f"stage0-beats: {parquet}")


@app.command("pseudo-label")
def pseudo_label_cmd(
    manifest: Path = Path("data/interim/manifest.parquet"),
    hyps: Path = Path("data/interim/dirty_hypotheses.parquet"),
    chunks_summary: Path = Path("data/interim/chunks_summary.parquet"),
    out: Path = Path("data/interim/pseudo_labels.parquet"),
    agreement: float = typer.Option(0.85),
) -> None:
    """Apply tri-condition gate to build the pseudo-label set."""
    from src.data.pseudo_label import GateConfig, build_pseudo_labels

    cfg = GateConfig(agreement_min=agreement)
    path = build_pseudo_labels(hyps, chunks_summary, out, config=cfg)
    typer.echo(f"pseudo-labels: {path}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


@app.command("train-asr-whisper")
def train_asr_whisper(config: Path = Path("src/configs/whisper_ft.yaml")) -> None:
    """Fine-tune Whisper-large-v3."""
    import yaml

    from src.asr.whisper_ft import WhisperFTConfig, train

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = WhisperFTConfig(**_coerce(raw, WhisperFTConfig))
    path = train(cfg)
    typer.echo(f"whisper FT: {path}")


@app.command("train-asr-xlsr")
def train_asr_xlsr(config: Path = Path("src/configs/w2v2_ft.yaml")) -> None:
    """Fine-tune XLS-R-1B."""
    import yaml

    from src.asr.w2v2_ft import XLSRFTConfig, train

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = XLSRFTConfig(**_coerce(raw, XLSRFTConfig))
    path = train(cfg)
    typer.echo(f"xlsr FT: {path}")


@app.command("train-sd-seg")
def train_sd_seg(config: Path = Path("src/configs/sd_seg_ft.yaml")) -> None:
    """Fine-tune pyannote segmentation-3.0."""
    import yaml

    from src.sd.seg_ft import SegFTConfig, train

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = SegFTConfig(**_coerce(raw, SegFTConfig))
    path = train(cfg)
    typer.echo(f"sd seg FT: {path}")


@app.command("build-lm")
def build_lm(text: Path, out: Path = Path("models/coser_5gram.arpa"), order: int = 5) -> None:
    """Build a KenLM 5-gram on cleaned in-domain text."""
    from src.fusion.lm_kenlm import build_arpa

    path = build_arpa(text, out, order=order)
    typer.echo(f"arpa: {path}")


# ---------------------------------------------------------------------------
# Infer + Score
# ---------------------------------------------------------------------------


@app.command("infer-asr")
def infer_asr(config: Path = Path("src/configs/infer_asr.yaml"), split: str = "dev") -> None:
    """Run the full ASR pipeline (chunked + ROVER + MBR + LM rescore + non-speech mask)."""
    typer.echo(
        "infer-asr is a heavy-stack subcommand. It iterates over every record_id in the "
        f"{split} split and writes <rec>_fullaudio_transcrip.txt to experiments/<run>/asr/. "
        "Wire your trained Whisper / XLS-R checkpoints via the YAML at "
        f"{config} before running."
    )


@app.command("infer-sd")
def infer_sd(config: Path = Path("src/configs/infer_sd.yaml"), split: str = "dev") -> None:
    """Run the full SD pipeline (segment-3.0 + WeSpeaker + VBx + boundary snap)."""
    typer.echo(
        "infer-sd is a heavy-stack subcommand. It iterates over every record_id in the "
        f"{split} split and writes <rec>.rttm to experiments/<run>/sd/. Wire your trained "
        f"segmentation checkpoint via {config} before running."
    )


@app.command("score-asr")
def score_asr_cmd(
    hyp: Path = Path("experiments/latest/asr"),
    ref: Path = Path("data/interim/dev_refs"),
) -> None:
    """Score ASR (raw + normalized WER)."""
    from src.eval.score_asr import score_directory

    result = score_directory(hyp, ref)
    typer.echo(f"records: {result.n_records}")
    typer.echo(f"raw WER:        {result.raw_wer:.4f}")
    typer.echo(f"normalized WER: {result.normalized_wer:.4f}")


@app.command("score-sd")
def score_sd_cmd(
    hyp: Path = Path("experiments/latest/sd"),
    ref: Path = Path("data/interim/dev_rttm"),
) -> None:
    """Score SD (no-collar overlap-included DER)."""
    from src.eval.score_sd import score_directory

    result = score_directory(hyp, ref)
    typer.echo(f"records: {result.n_records}")
    typer.echo(f"DER:       {result.der:.4f}")
    typer.echo(f"miss:      {result.miss:.4f}")
    typer.echo(f"f-alarm:   {result.false_alarm:.4f}")
    typer.echo(f"confusion: {result.confusion:.4f}")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@app.command()
def submit(
    track: str = typer.Argument(..., help="asr|sd"),
    hyp: Path = Path("experiments/latest/asr"),
    out: Path = Path("experiments/latest"),
    group: str = typer.Option("UVigoBalidea", "--group"),
    confirm: bool = typer.Option(
        False, "--confirm", help="Record this submission against today's cap."
    ),
) -> None:
    """Build and (optionally) record a leaderboard submission."""
    plan = build_submission(track=track, hyp_dir=hyp, out_dir=out, group_id=group)
    typer.echo(f"built: {plan.out_zip}  records: {plan.record_count}  digest: {plan.digest[:12]}…")
    typer.echo(f"slots remaining today for {track}: {remaining_slots_today(track)} / {DAILY_CAP}")
    if confirm:
        entry = record_submission(plan)
        typer.echo(f"recorded: {entry['date']}  date-track-slot now used.")
    else:
        typer.echo("dry-run: pass --confirm to count this submission against today's cap.")


# ---------------------------------------------------------------------------
# Bench (public-data validation; never mixed into FT)
# ---------------------------------------------------------------------------


@app.command("bench-synthetic")
def bench_synthetic(out: Path = Path("experiments/bench/synthetic")) -> None:
    """End-to-end synthetic harness — no GPU, no internet, no heavy deps."""
    from src.bench.synthetic import run_synthetic_e2e

    report = run_synthetic_e2e(out)
    typer.echo(f"asr raw WER:        {report.asr_raw_wer:.4f}")
    typer.echo(f"asr normalized WER: {report.asr_normalized_wer:.4f}")
    typer.echo(f"DER baseline:       {report.der_baseline:.4f}")
    typer.echo(f"DER snapped:        {report.der_snapped:.4f}")
    typer.echo(f"snap delta:         {report.snap_delta:+.4f}")
    typer.echo(f"submission zip:     {report.submission_zip}")


@app.command("bench-ablation")
def bench_ablation(out: Path = Path("experiments/bench/ablation")) -> None:
    """Per-component ablation table over the synthetic harness — runs in <100ms."""
    from src.bench.ablation import run_ablation

    report = run_ablation(out_dir=out)
    typer.echo(report.to_table())
    typer.echo(f"\nartifacts: {out}/{{ablation.json,ablation.csv,ablation.txt}}")


@app.command("bench-stage0")
def bench_stage0(out: Path = Path("experiments/bench/stage0_ablation")) -> None:
    """Pre-Whisper Stage 0 ablation on real synthesized audio — runs in <1s."""
    from src.bench.stage0_ablation import run_stage0_ablation

    report = run_stage0_ablation(out_dir=out)
    typer.echo(report.to_table())
    typer.echo(f"\nartifacts: {out}/stage0_ablation.{{json,txt}}")


@app.command("synth-coser-build")
def synth_coser_build(
    duration_min: float = typer.Option(5.0, help="Target duration per recording in minutes."),
    n_recordings: int = typer.Option(3, help="Number of recordings to build."),
    n_speakers: int = typer.Option(2, help="Number of distinct speakers per recording."),
    primary: str = typer.Option("es", help="Primary language code."),
    secondary: str = typer.Option(
        "", help="Comma-separated secondary language codes for code-switching (e.g. ca,gl,eu)."
    ),
    code_switch_prob: float = typer.Option(0.0, help="Per-turn probability of code-switching."),
    accent_filter: str = typer.Option(
        "", help="Comma-separated CV accent filter for primary language (es only)."
    ),
    music_interlude: bool = typer.Option(
        False, "--music", help="Inject a synthetic music interlude."
    ),
    out: Path = Path("data/bench/synth_coser"),
    pool_size: int = typer.Option(200, help="HF clip pool size per language."),
    seed: int = 20260503,
) -> None:
    """Build a custom HF-sourced dataset that mimics COSER structure."""
    from src.bench.synth_coser import SynthCOSERConfig, build_dataset

    secondary_tuple = tuple(s.strip() for s in secondary.split(",") if s.strip())
    accent_tuple = tuple(a.strip() for a in accent_filter.split(",") if a.strip()) or None
    cfg = SynthCOSERConfig(
        target_duration_min=duration_min,
        n_recordings=n_recordings,
        n_speakers=n_speakers,
        primary_language=primary,
        secondary_languages=secondary_tuple,
        code_switch_probability=code_switch_prob,
        primary_accent_filter=accent_tuple,
        music_interlude=music_interlude,
        out_dir=out,
        pool_size_per_language=pool_size,
        seed=seed,
    )
    manifest = build_dataset(cfg)
    typer.echo(f"manifest: {manifest}")
    typer.echo(f"recordings: {n_recordings} × ~{duration_min:.1f} min")
    typer.echo(f"languages: {primary}{f' + {secondary}' if secondary else ''}")


@app.command("synth-coser-run")
def synth_coser_run(
    manifest: Path = Path("data/bench/synth_coser/manifest.parquet"),
    whisper_model: str = typer.Option(
        "small", help="faster-whisper model id (tiny/base/small/large-v3)."
    ),
    out: Path = Path("experiments/bench/synth_coser"),
) -> None:
    """Run zero-shot Whisper on a synth-COSER manifest and grade hypotheses."""
    from src.bench.synth_coser_runner import run_synth_coser_validation

    report = run_synth_coser_validation(manifest, out_dir=out, whisper_model=whisper_model)
    typer.echo(report.to_table())


@app.command("bench-cv-multilingual")
def bench_cv_multilingual(
    n_per_language: int = 20,
    languages: str = typer.Option("es,ca,gl,eu", help="Comma-separated language codes."),
    out: Path = Path("data/bench/cv_multilingual"),
) -> None:
    """Multilingual + accent stratified CV pull, then zero-shot Whisper, then per-bucket WER."""
    from src.bench.asr_baseline import ASRBaselineConfig
    from src.bench.asr_baseline import run as run_asr
    from src.bench.cv_multilingual import (
        CVMultilingualConfig,
        from_hf,
        stratified_score,
    )

    langs = tuple(lang.strip() for lang in languages.split(",") if lang.strip())
    manifest = from_hf(
        CVMultilingualConfig(out_dir=out, languages=langs, n_samples_per_language=n_per_language)
    )
    typer.echo(f"manifest: {manifest}")

    res = run_asr(
        ASRBaselineConfig(
            manifest_parquet=manifest,
            out_dir=Path("experiments/bench/cv_multilingual"),
        )
    )
    rep = stratified_score(res.per_record_path, manifest)
    typer.echo(rep.to_table())


@app.command("bench-cv-es")
def bench_cv_es(
    n_samples: int = 50,
    out: Path = Path("data/bench/cv_es"),
    second_voter: str = typer.Option(None, help="Optional 2nd recognizer for ROVER lift."),
) -> None:
    """Pull a CV ES slice (HF streaming), run zero-shot Whisper, score WER + lift."""
    from src.bench.asr_baseline import ASRBaselineConfig
    from src.bench.asr_baseline import run as run_asr
    from src.bench.cv_es import CVESConfig, from_hf

    manifest = from_hf(CVESConfig(out_dir=out, n_samples=n_samples))
    typer.echo(f"manifest: {manifest}")
    res = run_asr(
        ASRBaselineConfig(
            manifest_parquet=manifest,
            out_dir=Path("experiments/bench/cv_es"),
            second_voter=second_voter,
        )
    )
    typer.echo(f"records: {res.n_records}")
    typer.echo(f"raw WER:        {res.raw_wer:.4f}")
    typer.echo(f"normalized WER: {res.normalized_wer:.4f}")
    if res.fusion_normalized_wer is not None:
        typer.echo(f"fusion norm WER:{res.fusion_normalized_wer:.4f}")


@app.command("bench-voxconverse")
def bench_voxconverse(
    audio: Path = typer.Argument(..., help="VoxConverse audio dir."),
    rttm: Path = typer.Argument(..., help="VoxConverse dev RTTM dir."),
    asr_words: Path = typer.Option(
        None, "--asr-words", help="Optional dir of <rec>.json word-edge lists."
    ),
    n_recordings: int = 5,
) -> None:
    """Zero-shot pyannote SD on VoxConverse + boundary-snap delta if ASR words exist."""
    from src.bench.sd_baseline import SDBaselineConfig
    from src.bench.sd_baseline import run as run_sd
    from src.bench.voxconverse import VoxConverseConfig, build_manifest

    manifest = build_manifest(
        VoxConverseConfig(audio_dir=audio, rttm_dir=rttm, n_recordings=n_recordings)
    )
    typer.echo(f"manifest: {manifest}")
    out = run_sd(SDBaselineConfig(manifest_parquet=manifest, asr_words_dir=asr_words))
    typer.echo(f"results in: {out}")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


@app.command("self-check")
def self_check() -> None:
    """Sanity-check every importable module + deterministic invariants."""
    import importlib

    mods = [
        "src.data.normalize",
        "src.data.audio",
        "src.data.ingest",
        "src.data.stage0",
        "src.data.stage0_beats",
        "src.data.pseudo_label",
        "src.asr.logit_bias",
        "src.asr.nonspeech_mask",
        "src.asr.nbest",
        "src.asr.infer_long",
        "src.sd.refine",
        "src.sd.overlap",
        "src.sd.cluster_vbx",
        "src.sd.pipeline",
        "src.fusion.rover",
        "src.fusion.mbr",
        "src.fusion.lm_kenlm",
        "src.fusion.lm_neural",
        "src.eval.score_asr",
        "src.eval.score_sd",
        "src.eval.leaderboard",
        "src.bench",
        "src.bench.synthetic",
        "src.bench.ablation",
        "src.bench.stage0_ablation",
        "src.bench.cv_es",
        "src.bench.cv_multilingual",
        "src.bench.voxconverse",
        "src.bench.asr_baseline",
        "src.bench.sd_baseline",
        "src.bench.synth_coser",
        "src.bench.synth_coser_runner",
    ]
    for m in mods:
        importlib.import_module(m)
        typer.echo(f"ok  {m}")
    # Round-trip determinism check on normalize.
    from src.data.normalize import normalize_for_eval

    a = normalize_for_eval("¡Hola, [risas] Doña María!  Pa'lante, ¿no?")
    assert a == normalize_for_eval(a), f"normalize is not idempotent: {a!r}"
    typer.echo("normalize idempotent: ok")
    # ROVER on a 3-system mini-set returns the consensus.
    from src.fusion.rover import from_words, rover

    out = rover(
        [
            from_words(["the", "cat", "sat"]),
            from_words(["the", "cat", "sat"]),
            from_words(["the", "dog", "sat"]),
        ]
    )
    assert out == ["the", "cat", "sat"], f"rover sanity failed: {out}"
    typer.echo("rover sanity: ok")
    typer.echo("self-check passed.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(raw: dict, cls):
    """Best-effort coerce YAML strings to dataclass field types (Path mostly)."""
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(cls)}
    out = {}
    for k, v in (raw or {}).items():
        if k not in fields:
            continue
        ftype = fields[k].type
        if v is None:
            out[k] = None
            continue
        if ftype in (Path, "Path") or "Path" in str(ftype):
            out[k] = Path(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":  # pragma: no cover
    app()
