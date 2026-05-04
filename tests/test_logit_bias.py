"""Deterministic tests for src.asr.logit_bias."""

from __future__ import annotations

from pathlib import Path

from src.asr.logit_bias import BiasLexicon, build_lexicons, load_lexicons, save_lexicons


def test_build_lexicons_minimum_doc_freq():
    rows = [
        {"transcript": "ganado vacuno pastoreo monte", "province": "LUG", "topic": "ganado"},
        {"transcript": "ganado vacuno pastoreo monte", "province": "LUG", "topic": "ganado"},
        {"transcript": "ganado vacuno pastoreo", "province": "LUG", "topic": "ganado"},
        {"transcript": "fiesta romería", "province": "LUG", "topic": "fiestas"},
        {"transcript": "fiesta romería", "province": "LUG", "topic": "fiestas"},
        {"transcript": "fiesta romería", "province": "LUG", "topic": "fiestas"},
    ]
    lex = build_lexicons(rows, top_n=4, min_doc_freq=2)
    assert ("LUG", "ganado") in lex.by_bucket
    assert ("LUG", "fiestas") in lex.by_bucket
    assert "ganado" in lex.by_bucket[("LUG", "ganado")]
    # 'fiesta' should appear in fiestas bucket; 'ganado' should not.
    assert "ganado" not in lex.by_bucket[("LUG", "fiestas")]


def test_words_for_returns_universal_plus_bucket():
    lex = BiasLexicon(
        by_bucket={("LUG", "ganado"): ["pastoreo", "monte"]},
        universal=["el", "la", "los"],
    )
    words = lex.words_for(province="LUG", topic="ganado")
    assert "pastoreo" in words
    assert "el" in words


def test_words_for_falls_back_to_universal_only():
    lex = BiasLexicon(by_bucket={("LUG", "ganado"): ["pastoreo"]}, universal=["el"])
    words = lex.words_for()
    assert words == ["el"]


def test_words_for_province_only_aggregates():
    lex = BiasLexicon(
        by_bucket={
            ("LUG", "ganado"): ["pastoreo"],
            ("LUG", "fiestas"): ["romería"],
        },
        universal=[],
    )
    words = lex.words_for(province="LUG")
    assert "pastoreo" in words and "romería" in words


def test_save_and_load_round_trip(tmp_path: Path):
    lex = BiasLexicon(
        by_bucket={("LUG", "ganado"): ["pastoreo", "monte"]},
        universal=["el", "la"],
    )
    out = save_lexicons(lex, tmp_path / "lex.json")
    loaded = load_lexicons(out)
    assert loaded.universal == ["el", "la"]
    assert loaded.by_bucket[("LUG", "ganado")] == ["pastoreo", "monte"]


def test_build_lexicons_empty_input_does_not_crash():
    lex = build_lexicons([])
    assert lex.universal == []
    assert lex.by_bucket == {}
