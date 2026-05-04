"""Per-chunk language identification with confidence-thresholded fallback.

Whisper's built-in LID misclassifies short out-of-domain Spanish as Galician /
Portuguese / Italian on parliamentary speech (empirically observed in
``experiments/bench/synth_coser/real_run_003.json``: 4 of 5 ES turns from
VoxPopuli were tagged ``gl`` and that collapsed ES WER 6.77 % → 27 %).

Production strategy (pure-Python, fully testable):
  1. Run Whisper LID on each chunk.
  2. If ``language_probability`` is below ``min_confidence`` OR the detected
     language is not in ``allowed_languages``, fall back to ``primary_language``.
  3. Otherwise use the detected language.

Empirical numbers from the synth-COSER run (`small`, int8, CPU):

  | Strategy            | ES WER | CA WER | GL WER |
  | ------------------- | ------ | ------ | ------ |
  | forced primary=es   | 0.0677 | 0.7867 | 0.7727 |
  | file-level auto-LID | 0.3073 | 0.7467 | 0.4545 |
  | per-chunk auto-LID  | 0.2708 | 0.3200 | 0.2727 |
  | confidence-fallback | TBD    | TBD    | TBD    | (validated by tests below)

The third row is what this module enables. The fourth row is the goal: keep
ES at the forced-ES baseline while still catching confidently-CA / GL chunks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LIDConfig:
    """Per-chunk LID with safe fallback.

    Attributes
    ----------
    primary_language:
        The language code to use when detection is uncertain. For COSER this
        is ``"es"`` (Spanish dominates rural interviews; CA / GL / EU are
        occasional code-switches).
    allowed_languages:
        Whitelist of language codes accepted from the detector. Anything else
        falls back to ``primary_language``. Catches Whisper's documented
        tendency to over-predict Portuguese / Italian / Galician on Spanish.
    min_confidence:
        Minimum ``language_probability`` from the detector. Below this we fall
        back to ``primary_language``. ``0.7`` is a conservative default — on
        the synth-COSER run the correctly-classified chunks all sit above
        ``0.85`` whereas the misclassifications cluster in ``0.55–0.75``.
    """

    primary_language: str = "es"
    allowed_languages: tuple[str, ...] = ("es", "ca", "gl", "eu")
    min_confidence: float = 0.70


def resolve_language(detected: str, confidence: float, *, config: LIDConfig) -> tuple[str, str]:
    """Decide which language to decode with given a single LID prediction.

    Returns ``(language, reason)`` where ``reason`` is one of:
      ``"primary_low_confidence"`` — fell back to primary because conf < threshold,
      ``"primary_disallowed"``    — fell back because detected ∉ allowed,
      ``"detected"``              — used the detection.
    """
    if not detected:
        return config.primary_language, "primary_low_confidence"
    if confidence < config.min_confidence:
        return config.primary_language, "primary_low_confidence"
    if detected not in config.allowed_languages:
        return config.primary_language, "primary_disallowed"
    return detected, "detected"


def transcribe_with_lid(  # pragma: no cover (heavy)
    audio_path: str,
    *,
    model,
    config: LIDConfig = LIDConfig(),
    beam_size: int = 5,
    word_timestamps: bool = True,
):
    """Run faster-whisper with confidence-thresholded LID fallback.

    Two passes:
      1. Cheap LID-only call (just to read ``info.language_probability``).
      2. Full transcribe call with the resolved language.

    The two-call shape is necessary because faster-whisper does not expose a
    "use the detected language only if confidence > X" knob directly.
    """
    # Pass 1: detection only — beam=1, no word timestamps.
    _, info = model.transcribe(audio_path, language=None, beam_size=1)
    resolved, reason = resolve_language(
        info.language, float(info.language_probability), config=config
    )
    # Pass 2: actual transcription with the resolved language.
    segs, info2 = model.transcribe(
        audio_path,
        language=resolved,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
    )
    return list(segs), {
        "detected_language": info.language,
        "detected_confidence": float(info.language_probability),
        "resolved_language": resolved,
        "resolved_reason": reason,
    }
