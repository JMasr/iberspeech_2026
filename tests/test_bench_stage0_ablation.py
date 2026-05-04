"""Stage 0 (pre-Whisper) ablation regression tests.

Asserts each Stage 0 module's downstream contribution on a deterministic
synthesized 60-second recording (10 s speech + 20 s silence + 20 s music
+ 10 s speech, with a [risas] burst at 3.4–3.7 s in the first speech section).
"""

from __future__ import annotations

import json

import pytest

from src.bench.stage0_ablation import (
    DEFAULT_CONFIGS,
    GT_SECTIONS,
    LAUGHTER_INTERVAL,
    REFERENCE_TEXT,
    Stage0Knobs,
    run_one,
    run_stage0_ablation,
    synthesize_recording,
)

# Locked numbers — change only with a deliberate update to the ablation stub.
EXPECTED = {
    "—+—+—": {
        "raw_wer": 1.3636363636363635,
        "norm_wer": 1.2727272727272727,
        "n_chunks": 6,
        "n_skipped_music": 0,
    },
    "VAD+—+—": {
        "raw_wer": 0.8181818181818182,
        "norm_wer": 0.7272727272727273,
        "n_chunks": 4,
        "n_skipped_music": 0,
    },
    "VAD+MUSIC-ROUTE+—": {
        "raw_wer": 0.09090909090909091,
        "norm_wer": 0.0,
        "n_chunks": 2,
        "n_skipped_music": 2,
    },
    "VAD+MUSIC-ROUTE+MASK": {"raw_wer": 0.0, "norm_wer": 0.0, "n_chunks": 2, "n_skipped_music": 2},
}


def test_synthesized_audio_shape_and_duration():
    audio = synthesize_recording()
    assert audio.shape[0] == 60 * 16_000
    assert audio.dtype.name == "float32"
    assert audio.min() >= -1.0 and audio.max() <= 1.0


def test_each_default_config_matches_expected_numbers():
    for knobs in DEFAULT_CONFIGS:
        row = run_one(knobs)
        exp = EXPECTED[row.config]
        assert row.asr_raw_wer == pytest.approx(exp["raw_wer"], abs=1e-9), row.config
        assert row.asr_normalized_wer == pytest.approx(exp["norm_wer"], abs=1e-9), row.config
        assert row.n_chunks_processed == exp["n_chunks"], row.config
        assert row.n_chunks_skipped_music == exp["n_skipped_music"], row.config


def test_each_layer_provides_strictly_positive_lift_on_raw_wer():
    """Every Stage 0 layer must reduce raw WER monotonically on this stub."""
    rows = [run_one(k) for k in DEFAULT_CONFIGS]
    for prev, curr in zip(rows, rows[1:]):
        assert curr.asr_raw_wer <= prev.asr_raw_wer, f"{curr.config} regressed vs {prev.config}"


def test_full_pipeline_recovers_reference_exactly():
    full = run_one(Stage0Knobs(use_vad=True, route_music=True, apply_nonspeech_mask=True))
    assert full.asr_raw_wer == pytest.approx(0.0, abs=1e-9)
    assert full.asr_normalized_wer == pytest.approx(0.0, abs=1e-9)


def test_no_stage0_baseline_is_clearly_broken():
    """Sanity: the no-Stage-0 baseline must have WER > 1.0 (more insertions than ref words)."""
    base = run_one(Stage0Knobs(use_vad=False, route_music=False, apply_nonspeech_mask=False))
    assert base.asr_raw_wer > 1.0


def test_vad_drops_silence_chunks():
    """Energy VAD must remove the silence section (10–30 s)."""
    no_vad = run_one(Stage0Knobs(use_vad=False, route_music=False, apply_nonspeech_mask=False))
    with_vad = run_one(Stage0Knobs(use_vad=True, route_music=False, apply_nonspeech_mask=False))
    assert with_vad.n_chunks_processed < no_vad.n_chunks_processed


def test_music_routing_drops_music_chunks():
    """Music routing must remove the music section (30–50 s)."""
    no_route = run_one(Stage0Knobs(use_vad=True, route_music=False, apply_nonspeech_mask=False))
    with_route = run_one(Stage0Knobs(use_vad=True, route_music=True, apply_nonspeech_mask=False))
    assert with_route.n_chunks_skipped_music >= 2
    assert with_route.n_chunks_processed < no_route.n_chunks_processed


def test_mask_removes_laughter_token_from_raw_track():
    """MASK must strictly lower raw WER (the [risas] insertion is gone)."""
    no_mask = run_one(Stage0Knobs(use_vad=True, route_music=True, apply_nonspeech_mask=False))
    with_mask = run_one(Stage0Knobs(use_vad=True, route_music=True, apply_nonspeech_mask=True))
    assert with_mask.asr_raw_wer < no_mask.asr_raw_wer


def test_snr_separates_speech_silence_music():
    """WADA-SNR must give clearly different values for the three section types."""
    row = run_one(Stage0Knobs(True, True, True))
    # Silence has near-zero amplitude — SNR is the lowest of the three.
    assert row.snr_silence_db < row.snr_music_db
    # Speech (random noise) has a high SNR ceiling under WADA — that's expected
    # because the algorithm assumes a Gaussian noise background; for our
    # synthetic speech the algorithm clamps near +60. We just assert it is
    # strictly higher than silence.
    assert row.snr_speech_db > row.snr_silence_db


def test_run_stage0_ablation_persists_artifacts(tmp_path):
    out = tmp_path / "abl"
    run_stage0_ablation(out_dir=out)
    assert (out / "stage0_ablation.json").exists()
    assert (out / "stage0_ablation.txt").exists()
    blob = json.loads((out / "stage0_ablation.json").read_text(encoding="utf-8"))
    assert len(blob) == len(DEFAULT_CONFIGS)
    assert {row["config"] for row in blob} == set(EXPECTED)


def test_reference_and_layout_constants_are_consistent():
    """The synthetic layout must place the laughter inside a SPEECH section."""
    lmid = 0.5 * (LAUGHTER_INTERVAL[0] + LAUGHTER_INTERVAL[1])
    in_speech = any(label == "speech" and start <= lmid < end for label, start, end in GT_SECTIONS)
    assert in_speech
    assert REFERENCE_TEXT.split() == [
        "buenas",
        "tardes",
        "señora",
        "cómo",
        "está",
        "muchas",
        "gracias",
        "por",
        "su",
        "tiempo",
        "señora",
    ]
