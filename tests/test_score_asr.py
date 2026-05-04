"""Deterministic tests for src.eval.score_asr (pure-Python WER path)."""

from __future__ import annotations

from pathlib import Path

from src.eval.score_asr import score_directory


def _write_pair(hyp: Path, ref: Path, rec: str, hyp_text: str, ref_text: str):
    (hyp / f"{rec}_fullaudio_transcrip.txt").write_text(hyp_text, encoding="utf-8")
    (ref / f"{rec}_fullaudio_transcrip.txt").write_text(ref_text, encoding="utf-8")


def test_score_directory_perfect_match(tmp_path: Path):
    hyp = tmp_path / "hyp"
    ref = tmp_path / "ref"
    hyp.mkdir()
    ref.mkdir()
    _write_pair(hyp, ref, "rec1", "hola mundo amigo", "hola mundo amigo")
    res = score_directory(hyp, ref, use_meeteval=False)
    assert res.raw_wer == 0.0
    assert res.normalized_wer == 0.0
    assert res.n_records == 1


def test_score_directory_punctuation_normalization(tmp_path: Path):
    hyp = tmp_path / "hyp"
    ref = tmp_path / "ref"
    hyp.mkdir()
    ref.mkdir()
    # Raw differs in punctuation, normalized should be identical.
    _write_pair(hyp, ref, "rec1", "Hola, mundo!", "Hola mundo")
    res = score_directory(hyp, ref, use_meeteval=False)
    # Raw WER may be > 0 because tokenization differs; normalized must be 0.
    assert res.normalized_wer == 0.0


def test_score_directory_handles_nonspeech(tmp_path: Path):
    hyp = tmp_path / "hyp"
    ref = tmp_path / "ref"
    hyp.mkdir()
    ref.mkdir()
    # Hyp incorrectly emits a non-speech token; the ref does not.
    # After post_edit_raw both should compare equal.
    _write_pair(hyp, ref, "rec1", "hola [risas] mundo", "hola mundo")
    res = score_directory(hyp, ref, use_meeteval=False)
    assert res.raw_wer == 0.0
