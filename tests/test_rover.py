"""Deterministic tests for src.fusion.rover."""

from __future__ import annotations

from src.fusion.rover import WordToken, from_words, rover


def test_rover_three_systems_majority():
    a = from_words(["the", "cat", "sat"])
    b = from_words(["the", "cat", "sat"])
    c = from_words(["the", "dog", "sat"])
    assert rover([a, b, c]) == ["the", "cat", "sat"]


def test_rover_single_hypothesis_passthrough():
    a = from_words(["hello", "world"])
    assert rover([a]) == ["hello", "world"]


def test_rover_empty_list():
    assert rover([]) == []


def test_rover_breaks_tie_with_confidence():
    # Two hypotheses, one slot disagrees. Higher-confidence wins.
    a = [WordToken("the", 1.0), WordToken("cat", 0.5), WordToken("sat", 1.0)]
    b = [WordToken("the", 1.0), WordToken("dog", 0.9), WordToken("sat", 1.0)]
    out = rover([a, b])
    assert out == ["the", "dog", "sat"]


def test_rover_handles_insertion():
    # Hyp B has an extra word relative to A.
    a = from_words(["hola", "mundo"])
    b = from_words(["hola", "muy", "mundo"])
    c = from_words(["hola", "muy", "mundo"])
    out = rover([a, b, c])
    assert out == ["hola", "muy", "mundo"]


def test_rover_handles_deletion():
    a = from_words(["the", "quick", "brown", "fox"])
    b = from_words(["the", "quick", "brown", "fox"])
    c = from_words(["the", "brown", "fox"])  # missing "quick"
    out = rover([a, b, c])
    assert out == ["the", "quick", "brown", "fox"]
