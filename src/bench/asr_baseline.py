"""Zero-shot Whisper baseline on a manifest, with optional fusion lift.

Validates the following hypotheses on public data:

  H1  Whisper-large-v3 zero-shot WER on Spanish read speech (CV ES) is in the
      single-digit percent range. Strong signal that the model is correctly
      configured before we hit the harder COSER corpus.
  H2  ROVER + MBR + KenLM fusion reduces WER vs Whisper 1-best on a multi-system
      bench. We need a 2nd voter (XLS-R or Whisper-medium) for this.
  H3  ``normalize_for_eval`` agrees with meeteval on this domain — verifies our
      dual-norm path before the COSER calibration step.

NOTE: This runs Whisper once per clip. faster-whisper is preferred (CPU-friendly);
we fall back to HF transformers if that's not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.normalize import normalize_for_eval, post_edit_raw
from src.fusion.mbr import wer as _wer


@dataclass
class ASRBaselineConfig:
    manifest_parquet: Path
    out_dir: Path = Path("experiments/bench/cv_es")
    backend: str = "faster-whisper"  # faster-whisper | hf
    model_id: str = "openai/whisper-large-v3"
    language: str = "es"
    compute_type: str = "int8"  # for faster-whisper CPU; ignored otherwise
    second_voter: str | None = None  # e.g. "facebook/wav2vec2-xls-r-1b"
    use_kenlm: Path | None = None  # path to ARPA if we want LM rescore


@dataclass
class ASRBaselineResult:
    n_records: int
    raw_wer: float
    normalized_wer: float
    fusion_normalized_wer: float | None
    per_record_path: Path


def run(cfg: ASRBaselineConfig) -> ASRBaselineResult:  # pragma: no cover (heavy)
    """Run zero-shot Whisper on every clip in the manifest, score, optionally fuse."""
    import pandas as pd

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.manifest_parquet)

    transcribe = _build_transcriber(cfg)

    rows = []
    for _, r in df.iterrows():
        hyp_raw = transcribe(r["audio_path"])
        rows.append(
            {
                "segment_id": r["segment_id"],
                "ref": r["transcript"],
                "hyp_whisper": hyp_raw,
            }
        )
    df_hyps = pd.DataFrame(rows)

    if cfg.second_voter:
        second = _build_transcriber(
            ASRBaselineConfig(
                manifest_parquet=cfg.manifest_parquet,
                out_dir=cfg.out_dir,
                backend="hf",
                model_id=cfg.second_voter,
                language=cfg.language,
            )
        )
        df_hyps["hyp_voter2"] = [second(r["audio_path"]) for _, r in df.iterrows()]

    # Score raw + normalized for the primary; ROVER+MBR if a 2nd voter exists.
    raw_wer = _corpus_wer(df_hyps["ref"], df_hyps["hyp_whisper"], normalize=False)
    norm_wer = _corpus_wer(df_hyps["ref"], df_hyps["hyp_whisper"], normalize=True)

    fusion_norm = None
    if "hyp_voter2" in df_hyps.columns:
        from src.fusion.rover import from_words, rover

        fused = []
        for _, r in df_hyps.iterrows():
            a = from_words(post_edit_raw(r["hyp_whisper"]).split())
            b = from_words(post_edit_raw(r["hyp_voter2"]).split())
            fused.append(" ".join(rover([a, b])))
        df_hyps["hyp_fused"] = fused
        fusion_norm = _corpus_wer(df_hyps["ref"], df_hyps["hyp_fused"], normalize=True)

    per_record = cfg.out_dir / "per_record.parquet"
    df_hyps.to_parquet(per_record, index=False)
    return ASRBaselineResult(
        n_records=len(df_hyps),
        raw_wer=raw_wer,
        normalized_wer=norm_wer,
        fusion_normalized_wer=fusion_norm,
        per_record_path=per_record,
    )


def _corpus_wer(refs, hyps, *, normalize: bool) -> float:
    transform = normalize_for_eval if normalize else post_edit_raw
    total_ref = 0
    total_err = 0
    for ref, hyp in zip(refs, hyps):
        ref_t = transform(ref).split()
        hyp_t = transform(hyp).split()
        total_ref += len(ref_t)
        total_err += int(round(_wer(ref_t, hyp_t) * max(1, len(ref_t))))
    return total_err / max(1, total_ref)


def _build_transcriber(cfg: ASRBaselineConfig):  # pragma: no cover (heavy)
    """Return a ``audio_path -> transcript`` callable for the chosen backend."""
    if cfg.backend == "faster-whisper":
        try:
            from faster_whisper import WhisperModel

            short_id = cfg.model_id.replace("openai/", "")
            model = WhisperModel(short_id, compute_type=cfg.compute_type)

            def _call(audio_path: str) -> str:
                segments, _info = model.transcribe(audio_path, language=cfg.language, beam_size=5)
                return " ".join(s.text.strip() for s in segments)

            return _call
        except ImportError:
            cfg.backend = "hf"

    if cfg.backend == "hf":
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(cfg.model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_id)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

        def _call(audio_path: str) -> str:
            from src.data.audio import load_mono_16k

            audio = load_mono_16k(audio_path)
            inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=224,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=cfg.language, task="transcribe"
                    ),
                )
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        return _call

    raise ValueError(f"unknown backend: {cfg.backend!r}")
