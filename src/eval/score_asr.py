"""ASR scoring: dual raw + normalized WER.

Wraps ``meeteval`` for the canonical numbers and falls back to a pure-Python
WER (the same Levenshtein used in MBR) when meeteval is not installed — useful
for unit tests and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.normalize import normalize_for_eval, post_edit_raw
from src.fusion.mbr import wer as _wer


@dataclass
class ASRResult:
    raw_wer: float
    normalized_wer: float
    n_records: int
    per_record: dict[str, dict[str, float]]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def score_directory(
    hyp_dir: str | Path,
    ref_dir: str | Path,
    *,
    use_meeteval: bool = True,
) -> ASRResult:
    """Score every ``<rec>_fullaudio_transcrip.txt`` in ``hyp_dir`` against
    matching references in ``ref_dir`` (same filename or ``<rec>.txt``).
    """
    hyp_dir = Path(hyp_dir)
    ref_dir = Path(ref_dir)

    pairs = _pair_hyp_ref(hyp_dir, ref_dir)
    if not pairs:
        raise FileNotFoundError(f"No hypothesis/reference pairs found under {hyp_dir} / {ref_dir}")

    raw_pairs = []
    norm_pairs = []
    per_record = {}
    for rec_id, hyp_path, ref_path in pairs:
        hyp_raw = post_edit_raw(_read_text(hyp_path))
        ref_raw = post_edit_raw(_read_text(ref_path))
        hyp_norm = normalize_for_eval(hyp_raw)
        ref_norm = normalize_for_eval(ref_raw)
        raw_pairs.append((rec_id, ref_raw, hyp_raw))
        norm_pairs.append((rec_id, ref_norm, hyp_norm))
        per_record[rec_id] = {
            "raw_wer": _wer(ref_raw.split(), hyp_raw.split()),
            "normalized_wer": _wer(ref_norm.split(), hyp_norm.split()),
        }

    if use_meeteval:
        try:
            return ASRResult(
                raw_wer=_meeteval_corpus_wer(raw_pairs),
                normalized_wer=_meeteval_corpus_wer(norm_pairs),
                n_records=len(pairs),
                per_record=per_record,
            )
        except ImportError:
            pass

    # Pure-Python corpus WER: total edit distance / total ref words.
    return ASRResult(
        raw_wer=_corpus_wer(raw_pairs),
        normalized_wer=_corpus_wer(norm_pairs),
        n_records=len(pairs),
        per_record=per_record,
    )


def _pair_hyp_ref(hyp_dir: Path, ref_dir: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for hyp in sorted(hyp_dir.glob("*_fullaudio_transcrip.txt")):
        rec_id = hyp.name.replace("_fullaudio_transcrip.txt", "")
        for cand in (ref_dir / hyp.name, ref_dir / f"{rec_id}.txt"):
            if cand.exists():
                out.append((rec_id, hyp, cand))
                break
    return out


def _corpus_wer(pairs: list[tuple[str, str, str]]) -> float:
    total_ref = 0
    total_err = 0
    for _, ref, hyp in pairs:
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        total_ref += len(ref_tokens)
        # Levenshtein computed via _wer * len(ref). Reuse the function.
        total_err += int(round(_wer(ref_tokens, hyp_tokens) * max(1, len(ref_tokens))))
    return total_err / max(1, total_ref)


def _meeteval_corpus_wer(pairs: list[tuple[str, str, str]]) -> float:  # pragma: no cover (heavy)
    from meeteval.io.stm import STM, STMLine
    from meeteval.wer import wer as mtv_wer

    ref_lines = []
    hyp_lines = []
    for rec_id, ref, hyp in pairs:
        ref_lines.append(
            STMLine(
                filename=rec_id,
                channel="A",
                speaker_id="ref",
                begin_time=0.0,
                end_time=1.0,
                transcript=ref,
            )
        )
        hyp_lines.append(
            STMLine(
                filename=rec_id,
                channel="A",
                speaker_id="ref",
                begin_time=0.0,
                end_time=1.0,
                transcript=hyp,
            )
        )
    return float(mtv_wer.cpwer(STM(ref_lines), STM(hyp_lines)).error_rate)
