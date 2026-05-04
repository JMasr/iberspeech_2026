"""Tests for the CV multilingual harness.

Network-dependent paths are not exercised. We test:
  - the manifest schema is consistent,
  - the FT-rejection guard fires on bench-tagged rows,
  - the stratified scorer aggregates correctly on synthetic data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.bench import assert_no_bench_rows
from src.bench.cv_multilingual import (
    SPANISH_ACCENTS_OF_INTEREST,
    StratifiedReport,
    _normalize_accent,
    stratified_score,
)


def test_normalize_accent_lowercases_and_underscores():
    assert _normalize_accent("Latin American") == "latin_american"
    assert _normalize_accent("Castilian-Spanish") == "castilian_spanish"
    assert _normalize_accent("") == "_"
    assert _normalize_accent(None) == "_"


def test_assert_no_bench_rows_blocks_cv_multilingual():
    df = pd.DataFrame([{"audio_path": "a.wav", "transcript": "hola", "source": "cv_multilingual"}])
    with pytest.raises(ValueError, match="(?i)external audio is forbidden") as exc:
        assert_no_bench_rows(df)
    assert "cv_multilingual" in str(exc.value)


def test_assert_no_bench_rows_blocks_voxconverse():
    df = pd.DataFrame([{"source": "voxconverse"}])
    with pytest.raises(ValueError):
        assert_no_bench_rows(df)


def test_assert_no_bench_rows_allows_coser_rows():
    df = pd.DataFrame([{"source": "coser"}, {"source": "longform"}, {"source": None}])
    assert_no_bench_rows(df)  # does not raise


def test_assert_no_bench_rows_no_op_without_source_column():
    df = pd.DataFrame([{"audio_path": "x", "transcript": "y"}])
    assert_no_bench_rows(df)


def test_spanish_accents_of_interest_includes_castilian_and_andalusian():
    assert "castilian" in SPANISH_ACCENTS_OF_INTEREST
    assert "andalusian" in SPANISH_ACCENTS_OF_INTEREST


def test_stratified_score_aggregates_by_language(tmp_path):
    # Manifest has 2 ES, 2 CA. Hyps have known errors.
    manifest = pd.DataFrame(
        [
            {"segment_id": "es_0", "language": "es", "accent": "castilian"},
            {"segment_id": "es_1", "language": "es", "accent": "castilian"},
            {"segment_id": "ca_0", "language": "ca", "accent": "_"},
            {"segment_id": "ca_1", "language": "ca", "accent": "_"},
        ]
    )
    per_record = pd.DataFrame(
        [
            {"segment_id": "es_0", "ref": "hola amigo", "hyp_whisper": "hola amigo"},
            {"segment_id": "es_1", "ref": "hola amigo", "hyp_whisper": "hola enemigo"},
            {"segment_id": "ca_0", "ref": "bon dia", "hyp_whisper": "bon dia"},
            {"segment_id": "ca_1", "ref": "bon dia", "hyp_whisper": "bo dia"},
        ]
    )
    manifest_path = tmp_path / "manifest.parquet"
    per_path = tmp_path / "per_record.parquet"
    manifest.to_parquet(manifest_path)
    per_record.to_parquet(per_path)

    rep = stratified_score(per_path, manifest_path)
    # 1 substitution / 4 ref words for ES → 0.25 normalized WER.
    assert rep.by_language["es"]["normalized_wer"] == pytest.approx(0.25)
    # 1 substitution / 4 ref words for CA → 0.25.
    assert rep.by_language["ca"]["normalized_wer"] == pytest.approx(0.25)
    # By accent for ES: only "castilian" present.
    assert "castilian" in rep.by_accent
    assert rep.by_accent["castilian"]["normalized_wer"] == pytest.approx(0.25)


def test_stratified_report_table_has_both_sections():
    rep = StratifiedReport(
        by_language={"es": {"n": 2.0, "raw_wer": 0.1, "normalized_wer": 0.05}},
        by_accent={"castilian": {"n": 2.0, "raw_wer": 0.1, "normalized_wer": 0.05}},
    )
    text = rep.to_table()
    assert "By language" in text
    assert "By accent" in text
    assert "es" in text
    assert "castilian" in text
