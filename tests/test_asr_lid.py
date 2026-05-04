"""Tests for src.asr.lid — confidence-thresholded language identification fallback.

The heavy ``transcribe_with_lid`` is exercised end-to-end in
``experiments/bench/synth_coser/real_run_004.json`` (real Whisper). Here we
test the deterministic ``resolve_language`` decision logic.
"""

from __future__ import annotations

import pytest

from src.asr.lid import LIDConfig, resolve_language


def test_high_confidence_allowed_language_is_used():
    cfg = LIDConfig(primary_language="es", allowed_languages=("es", "ca", "gl"), min_confidence=0.7)
    lang, reason = resolve_language("ca", 0.95, config=cfg)
    assert lang == "ca"
    assert reason == "detected"


def test_low_confidence_falls_back_to_primary():
    cfg = LIDConfig(primary_language="es", min_confidence=0.7)
    lang, reason = resolve_language("gl", 0.6, config=cfg)
    assert lang == "es"
    assert reason == "primary_low_confidence"


def test_disallowed_language_falls_back_to_primary():
    cfg = LIDConfig(primary_language="es", allowed_languages=("es", "ca"), min_confidence=0.5)
    lang, reason = resolve_language("pt", 0.99, config=cfg)
    assert lang == "es"
    assert reason == "primary_disallowed"


def test_at_threshold_is_accepted():
    """Boundary: confidence == threshold should be accepted (lt comparison)."""
    cfg = LIDConfig(primary_language="es", allowed_languages=("es", "gl"), min_confidence=0.7)
    lang, reason = resolve_language("gl", 0.7, config=cfg)
    assert lang == "gl"
    assert reason == "detected"


def test_empty_detection_falls_back():
    cfg = LIDConfig()
    lang, reason = resolve_language("", 0.99, config=cfg)
    assert lang == cfg.primary_language
    assert reason == "primary_low_confidence"


def test_primary_used_when_detection_is_primary():
    cfg = LIDConfig(primary_language="es")
    lang, reason = resolve_language("es", 0.95, config=cfg)
    assert lang == "es"
    assert reason == "detected"


def test_default_config_matches_coser_strategy():
    cfg = LIDConfig()
    assert cfg.primary_language == "es"
    assert "ca" in cfg.allowed_languages
    assert "gl" in cfg.allowed_languages
    assert "eu" in cfg.allowed_languages
    assert 0 < cfg.min_confidence < 1


@pytest.mark.parametrize(
    "detected,conf,expected",
    [
        ("es", 0.99, "es"),  # high-confidence primary
        ("ca", 0.85, "ca"),  # high-confidence allowed
        ("ca", 0.50, "es"),  # low-confidence → primary
        ("pt", 0.99, "es"),  # disallowed → primary
        ("eu", 0.71, "eu"),  # just over threshold
    ],
)
def test_resolve_language_table(detected, conf, expected):
    """Table-driven sanity check on the resolution rules."""
    cfg = LIDConfig()
    out, _ = resolve_language(detected, conf, config=cfg)
    assert out == expected
