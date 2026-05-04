"""Common Voice multilingual + accent harness.

COSER is mostly Spanish but interleaves Galician / Catalan / Basque, and rural
Spanish has wide accent variation. To probe how Whisper degrades across these
shifts BEFORE COSER data lands, we pull tiny streamed slices from Mozilla
Common Voice for each of:

  - **Spanish** (``es``) — rural-dialect proxy via accent stratification.
  - **Catalan** (``ca``)
  - **Galician** (``gl``)
  - **Basque** (``eu``)

Within Spanish, we further stratify by the ``accents`` field (Castilian, Latin
American, Andalusian, Argentinian, Mexican, …) so we can measure WER drift
across accent groups.

**Eval-rule note**: Common Voice audio is ONLY used for pipeline validation
under ``data/bench/cv_multilingual/`` and MUST NOT be added to any FT manifest.
The ALBAYZIN Speech-COSER rules forbid external audio for training. This
module enforces that by writing to a separate directory and tagging every row
with ``source = "cv_multilingual"``; the FT recipes refuse rows with that tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUT = Path("data/bench/cv_multilingual")
DEFAULT_LANGUAGES = ("es", "ca", "gl", "eu")
SPANISH_ACCENTS_OF_INTEREST = (
    "castilian",
    "andalusian",
    "latin_american",
    "argentinian",
    "mexican",
    "chilean",
)


@dataclass
class CVMultilingualConfig:
    out_dir: Path = DEFAULT_OUT
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    n_samples_per_language: int = 20
    hf_dataset: str = "mozilla-foundation/common_voice_17_0"
    hf_split: str = "test"
    accent_stratify_es: bool = True
    samples_per_accent: int = 10
    accents: tuple[str, ...] = SPANISH_ACCENTS_OF_INTEREST
    extra: dict = field(default_factory=dict)


def _normalize_accent(raw: str | None) -> str:
    if not raw:
        return "_"
    s = raw.strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s


def from_hf(cfg: CVMultilingualConfig = CVMultilingualConfig()) -> Path:  # pragma: no cover
    """Stream slices from each language and emit a unified manifest.

    The streaming mode pulls one shard's worth of audio at a time and stops
    once we have enough samples per (language, accent) bucket. Total network
    transfer for a 20-sample-per-language run is typically <50 MB.
    """
    from datasets import load_dataset
    import pandas as pd
    import soundfile as sf

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = cfg.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for language in cfg.languages:
        per_lang_clips_dir = clips_dir / language
        per_lang_clips_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(
            cfg.hf_dataset,
            language,
            split=cfg.hf_split,
            streaming=True,
        )

        if language == "es" and cfg.accent_stratify_es:
            accent_counts = {a: 0 for a in cfg.accents}
            other_count = 0
            other_cap = max(
                1, cfg.n_samples_per_language - cfg.samples_per_accent * len(cfg.accents)
            )

            for sample in ds:
                accent = _normalize_accent(sample.get("accents") or sample.get("accent"))
                if accent in accent_counts and accent_counts[accent] < cfg.samples_per_accent:
                    accent_counts[accent] += 1
                elif other_count < other_cap:
                    accent = "other"
                    other_count += 1
                else:
                    if all(c >= cfg.samples_per_accent for c in accent_counts.values()):
                        break
                    continue

                wav_path = (
                    per_lang_clips_dir
                    / f"es_{accent}_{accent_counts.get(accent, other_count):04d}.wav"
                )
                _persist_clip(sample, wav_path, sf)
                rows.append(_row_from_sample(sample, wav_path, language=language, accent=accent))
        else:
            for i, sample in enumerate(ds):
                if i >= cfg.n_samples_per_language:
                    break
                wav_path = per_lang_clips_dir / f"{language}_{i:04d}.wav"
                _persist_clip(sample, wav_path, sf)
                rows.append(_row_from_sample(sample, wav_path, language=language, accent="_"))

    out = cfg.out_dir / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _persist_clip(sample, wav_path: Path, sf):  # pragma: no cover
    audio = sample["audio"]
    sf.write(str(wav_path), audio["array"], audio["sampling_rate"])


def _row_from_sample(sample, wav_path: Path, *, language: str, accent: str) -> dict:
    audio = sample["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return {
        "segment_id": wav_path.stem,
        "audio_path": str(wav_path),
        "transcript": sample.get("sentence", ""),
        "duration_s": float(duration),
        "language": language,
        "accent": accent,
        "source": "cv_multilingual",  # FT recipes refuse rows with this tag
        "split": "bench",
    }


# ---------------------------------------------------------------------------
# Per-language and per-accent ablation runner — wraps src.bench.asr_baseline.
# ---------------------------------------------------------------------------


@dataclass
class StratifiedReport:
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)
    by_accent: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_table(self) -> str:
        lines = ["# By language", f"{'language':<12} {'n':>5} {'raw WER':>10} {'norm WER':>10}"]
        for lang, stats in sorted(self.by_language.items()):
            lines.append(
                f"{lang:<12} {int(stats['n']):>5d} "
                f"{stats['raw_wer']:>10.4f} {stats['normalized_wer']:>10.4f}"
            )
        if self.by_accent:
            lines.append("")
            lines.append("# By accent (Spanish only)")
            lines.append(f"{'accent':<18} {'n':>5} {'raw WER':>10} {'norm WER':>10}")
            for acc, stats in sorted(self.by_accent.items()):
                lines.append(
                    f"{acc:<18} {int(stats['n']):>5d} "
                    f"{stats['raw_wer']:>10.4f} {stats['normalized_wer']:>10.4f}"
                )
        return "\n".join(lines)


def stratified_score(per_record_parquet: Path, manifest_parquet: Path) -> StratifiedReport:
    """Aggregate per-record WERs by language and (within Spanish) by accent.

    Inputs:
      ``per_record_parquet`` — output of ``src.bench.asr_baseline.run`` (columns:
      segment_id, ref, hyp_whisper, …).
      ``manifest_parquet`` — output of ``from_hf`` (carries language + accent).

    Returns a ``StratifiedReport`` with both raw and normalized corpus WER per
    bucket. Pure-Python WER (no meeteval dependency).
    """
    import pandas as pd

    from src.data.normalize import normalize_for_eval, post_edit_raw
    from src.fusion.mbr import wer as _wer

    per = pd.read_parquet(per_record_parquet)
    man = pd.read_parquet(manifest_parquet)
    df = per.merge(
        man[["segment_id", "language", "accent"]],
        on="segment_id",
        how="left",
    )

    def _agg(slice_df) -> dict[str, float]:
        total_ref = 0
        total_err_raw = 0
        total_err_norm = 0
        for _, r in slice_df.iterrows():
            ref_raw = post_edit_raw(r["ref"]).split()
            hyp_raw = post_edit_raw(r["hyp_whisper"]).split()
            ref_norm = normalize_for_eval(r["ref"]).split()
            hyp_norm = normalize_for_eval(r["hyp_whisper"]).split()
            total_ref += len(ref_norm)
            total_err_raw += int(round(_wer(ref_raw, hyp_raw) * max(1, len(ref_raw))))
            total_err_norm += int(round(_wer(ref_norm, hyp_norm) * max(1, len(ref_norm))))
        return {
            "n": float(len(slice_df)),
            "raw_wer": total_err_raw / max(1, total_ref),
            "normalized_wer": total_err_norm / max(1, total_ref),
        }

    by_language = {lang: _agg(df[df["language"] == lang]) for lang in df["language"].unique()}
    by_accent: dict[str, dict[str, float]] = {}
    es_only = df[df["language"] == "es"]
    for accent in es_only["accent"].dropna().unique():
        sub = es_only[es_only["accent"] == accent]
        if len(sub) == 0:
            continue
        by_accent[str(accent)] = _agg(sub)

    return StratifiedReport(by_language=by_language, by_accent=by_accent)
