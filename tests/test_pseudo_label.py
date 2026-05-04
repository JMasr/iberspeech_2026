"""Deterministic tests for src.data.pseudo_label gate logic."""

from __future__ import annotations

from src.data.pseudo_label import GateConfig, gate, snr_bucket, token_agreement


def test_token_agreement_perfect():
    assert token_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_token_agreement_disjoint():
    assert token_agreement(["a", "b"], ["c", "d"]) == 0.0


def test_token_agreement_partial_lcs():
    # LCS = 2 ("a", "c"); 2*2 / (3+3) = 0.666...
    score = token_agreement(["a", "b", "c"], ["a", "x", "c"])
    assert abs(score - 4 / 6) < 1e-9


def test_token_agreement_empty():
    assert token_agreement([], ["a"]) == 0.0
    assert token_agreement(["a"], []) == 0.0


def test_snr_bucket_assignment():
    buckets = [(-1e9, 5.0), (5.0, 15.0), (15.0, 1e9)]
    assert snr_bucket(-3.0, buckets) == 0
    assert snr_bucket(8.0, buckets) == 1
    assert snr_bucket(20.0, buckets) == 2


def test_gate_passes_clean_high_agreement():
    kept, diag = gate(
        whisper_words=["hola", "mundo", "amigo"],
        xlsr_words=["hola", "mundo", "amigo"],
        whisper_logprob=-0.20,
        snr_db=20.0,
        duration_s=2.0,
        mean_wps=1.5,
        std_wps=0.3,
        config=GateConfig(),
    )
    assert kept
    assert diag["agreement"] == 1.0


def test_gate_rejects_low_agreement():
    kept, diag = gate(
        whisper_words=["hola", "mundo"],
        xlsr_words=["adiós", "mundo"],
        whisper_logprob=-0.20,
        snr_db=20.0,
        duration_s=2.0,
        mean_wps=1.0,
        std_wps=0.2,
    )
    assert not kept
    assert diag["agreement"] < 0.85


def test_gate_rejects_low_logprob_in_low_snr_bucket():
    # SNR < 5 → bucket 0 → threshold -0.55. Logprob -0.6 is below.
    kept, _ = gate(
        whisper_words=["hola", "mundo"],
        xlsr_words=["hola", "mundo"],
        whisper_logprob=-0.6,
        snr_db=2.0,
        duration_s=2.0,
        mean_wps=1.0,
        std_wps=0.2,
    )
    assert not kept


def test_gate_rejects_outlier_word_rate():
    # A duration with an absurdly high words-per-second is rejected.
    kept, diag = gate(
        whisper_words=["a"] * 100,
        xlsr_words=["a"] * 100,
        whisper_logprob=-0.1,
        snr_db=20.0,
        duration_s=2.0,
        mean_wps=1.0,
        std_wps=0.5,
    )
    assert not kept
    assert diag["length_z"] > 2.0
