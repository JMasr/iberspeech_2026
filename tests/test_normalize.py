"""Deterministic tests for src.data.normalize."""

from __future__ import annotations

from pathlib import Path

from src.data.normalize import (
    DialectLexicon,
    normalize_for_eval,
    post_edit_raw,
    strip_nonspeech,
)


def test_normalize_strips_punct_and_lowercases():
    out = normalize_for_eval("¡Hola, Doña María! ¿Qué tal?")
    assert out == "hola doña maría qué tal"


def test_normalize_strips_nonspeech_tokens():
    out = normalize_for_eval("Hola [risas] amigo [susp] mío.")
    assert "risas" not in out
    assert "susp" not in out
    assert out == "hola amigo mío"


def test_normalize_collapses_whitespace():
    out = normalize_for_eval("hola    mundo\n\n\tamigo")
    assert out == "hola mundo amigo"


def test_normalize_preserves_word_internal_apostrophe():
    out = normalize_for_eval("pa'lante y pa'rriba")
    # word-internal apostrophe should survive
    assert "pa'lante" in out
    assert "pa'rriba" in out


def test_normalize_idempotent():
    a = normalize_for_eval("¡Hola, [risas] Doña María!  Pa'lante, ¿no?")
    b = normalize_for_eval(a)
    assert a == b


def test_post_edit_raw_keeps_punctuation():
    out = post_edit_raw("¡Hola, Doña María! ¿Qué tal?")
    assert "Hola" in out
    assert "?" in out
    assert "¡" in out


def test_post_edit_raw_normalizes_typographic_quotes():
    out = post_edit_raw("Dijo “hola” —pero no respondió—.")
    assert '"hola"' in out
    assert "“" not in out and "”" not in out
    assert "—" not in out


def test_strip_nonspeech_handles_inline_annotations():
    out = strip_nonspeech("hola {{nombre}} ((overlap)) <inaudible> mundo")
    assert "{{" not in out and "((" not in out and "<" not in out
    assert "hola" in out and "mundo" in out


def test_dialect_lexicon_to_canonical_word_whole(tmp_path: Path):
    lex_path = tmp_path / "lex.json"
    lex_path.write_text(
        '{"rules": [{"id": "r", "canonical": "para", "variants": ["pa"], '
        '"context": "word_whole", "applies_to": ["*"]}]}',
        encoding="utf-8",
    )
    lex = DialectLexicon.from_file(lex_path)
    assert lex.to_canonical("voy pa casa") == "voy para casa"


def test_dialect_lexicon_to_canonical_suffix(tmp_path: Path):
    lex_path = tmp_path / "lex.json"
    lex_path.write_text(
        '{"rules": [{"id": "r", "canonical": "ado", "variants": ["ao"], '
        '"context": "word_suffix", "applies_to": ["*"]}]}',
        encoding="utf-8",
    )
    lex = DialectLexicon.from_file(lex_path)
    assert lex.to_canonical("comío y pesao") == "comío y pesado"


def test_dialect_lexicon_province_filter(tmp_path: Path):
    lex_path = tmp_path / "lex.json"
    lex_path.write_text(
        '{"rules": [{"id": "r", "canonical": "donde", "variants": ["onde"], '
        '"context": "word_whole", "applies_to": ["AST"]}]}',
        encoding="utf-8",
    )
    lex = DialectLexicon.from_file(lex_path)
    # province match → applied
    assert lex.to_canonical("onde está", province="AST") == "donde está"
    # province miss → unchanged
    assert lex.to_canonical("onde está", province="MAD") == "onde está"


def test_dialect_lexicon_loads_real_file():
    """The committed lexicon at data/dialect_lexicon.json must parse cleanly."""
    lex = DialectLexicon.from_file(
        Path(__file__).resolve().parents[1] / "data" / "dialect_lexicon.json"
    )
    # Sanity: applying to canonical is non-destructive on text without variants.
    assert lex.to_canonical("buenas tardes") == "buenas tardes"
