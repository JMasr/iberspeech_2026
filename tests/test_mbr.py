"""Deterministic tests for src.fusion.mbr."""

from __future__ import annotations

from src.fusion.mbr import NBestEntry, mbr, wer


def test_wer_identity():
    assert wer(["a", "b", "c"], ["a", "b", "c"]) == 0.0


def test_wer_substitution():
    # 1 sub / 3 ref words = 1/3
    assert abs(wer(["a", "b", "c"], ["a", "x", "c"]) - 1 / 3) < 1e-9


def test_wer_empty_ref():
    assert wer([], []) == 0.0
    assert wer([], ["a"]) == 1.0


def test_mbr_picks_consensus():
    # Three hypotheses; two agree, one differs. The consensus should win
    # because it minimizes expected WER.
    nb = [
        NBestEntry("the cat sat", -0.05),
        NBestEntry("the cat sat", -0.10),
        NBestEntry("the dog sat", -0.20),
    ]
    chosen = mbr(nb)
    assert chosen.text == "the cat sat"


def test_mbr_single_passthrough():
    nb = [NBestEntry("hello", -0.5)]
    assert mbr(nb).text == "hello"


def test_mbr_handles_close_scores():
    # Three identical-score hypotheses, two consensus. Consensus wins.
    nb = [
        NBestEntry("hola amigo", -0.3),
        NBestEntry("hola amigo", -0.3),
        NBestEntry("hola amiga", -0.3),
    ]
    assert mbr(nb).text == "hola amigo"
