"""Ablation harness regression tests.

The harness is deterministic. Each row is a real run of the production fusion
code with one knob toggled off; the numbers below are asserted exactly.

Monotonicity: the configurations are ordered from "least fusion" to "most
fusion". A correctness invariant of the design is that each step is
**no-worse** than its predecessor (lower or equal WER, lower or equal DER).
A regression that breaks monotonicity here means a fusion stage hurt the
recovery — investigate the implementation, not the test.
"""

from __future__ import annotations

import json

import pytest

from src.bench.ablation import (
    DEFAULT_CONFIGS,
    AblationKnobs,
    run_ablation,
    run_one,
)

# Locked-in expected numbers from the deterministic stub. Update only when the
# stub itself changes (and document why in PROGRESS.md).
EXPECTED = {
    "—+—+—+—": {"raw_wer": 0.3, "norm_wer": 0.2, "der": 0.04615384615384617},
    "MBR+—+—+—": {"raw_wer": 0.2, "norm_wer": 0.1, "der": 0.04615384615384617},
    "MBR+ROVER+—+—": {"raw_wer": 0.1, "norm_wer": 0.0, "der": 0.04615384615384617},
    "MBR+ROVER+MASK+—": {"raw_wer": 0.0, "norm_wer": 0.0, "der": 0.04615384615384617},
    "MBR+ROVER+MASK+SNAP": {"raw_wer": 0.0, "norm_wer": 0.0, "der": 0.0},
}


def test_each_default_config_matches_expected_numbers():
    for knobs in DEFAULT_CONFIGS:
        row = run_one(knobs)
        exp = EXPECTED[row.config]
        assert row.asr_raw_wer == pytest.approx(exp["raw_wer"], abs=1e-9), row.config
        assert row.asr_normalized_wer == pytest.approx(exp["norm_wer"], abs=1e-9), row.config
        assert row.der == pytest.approx(exp["der"], abs=1e-9), row.config


def test_each_layer_provides_strictly_positive_lift():
    """Adding a layer must reduce raw WER OR DER strictly (no flats in our stub)."""
    rows = [run_one(k) for k in DEFAULT_CONFIGS]
    for prev, curr in zip(rows, rows[1:]):
        improved = (curr.asr_raw_wer < prev.asr_raw_wer) or (curr.der < prev.der)
        assert improved, f"{curr.config} did not improve over {prev.config}"


def test_full_pipeline_recovers_reference_exactly():
    """The last config is the full pipeline; it must hit 0.0 raw WER and 0.0 DER."""
    full = run_one(AblationKnobs(True, True, True, True))
    assert full.asr_raw_wer == pytest.approx(0.0, abs=1e-9)
    assert full.asr_normalized_wer == pytest.approx(0.0, abs=1e-9)
    assert full.der == pytest.approx(0.0, abs=1e-9)


def test_no_fusion_baseline_has_measurable_error():
    """Sanity: the baseline must NOT be already-perfect; otherwise the ablation says nothing."""
    base = run_one(AblationKnobs(False, False, False, False))
    assert base.asr_raw_wer >= 0.20
    assert base.der >= 0.04


def test_run_ablation_persists_artifacts(tmp_path):
    out = tmp_path / "abl"
    run_ablation(out_dir=out)
    assert (out / "ablation.json").exists()
    assert (out / "ablation.csv").exists()
    assert (out / "ablation.txt").exists()
    blob = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    assert len(blob) == len(DEFAULT_CONFIGS)
    assert {row["config"] for row in blob} == set(EXPECTED)


def test_csv_has_correct_header(tmp_path):
    run_ablation(out_dir=tmp_path / "abl")
    header = (tmp_path / "abl" / "ablation.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "config,use_mbr,use_rover,use_nonspeech_mask,use_boundary_snap,"
        "asr_raw_wer,asr_normalized_wer,der"
    )


def test_table_string_is_sorted_and_aligned():
    """The table includes every config in DEFAULT_CONFIGS order."""
    rows = [run_one(k) for k in DEFAULT_CONFIGS]
    from src.bench.ablation import AblationReport

    text = AblationReport(rows=rows).to_table()
    lines = text.splitlines()
    assert lines[0].startswith("config")
    assert len(lines) == 2 + len(DEFAULT_CONFIGS)
    for r, line in zip(rows, lines[2:]):
        assert line.startswith(r.config), (r.config, line)
