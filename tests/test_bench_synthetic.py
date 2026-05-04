"""End-to-end smoke test of the COSER pipeline using stub recognizers.

This test exercises every deterministic piece of the pipeline:
  - per-chunk MBR over an n-best list,
  - ROVER vs a 2nd voter,
  - long-form word-LCS overlap merge,
  - AudioSet non-speech mask (drops ``[risas]`` token in the laughter region),
  - dual-output normalization,
  - boundary snap to ASR word edges,
  - RTTM cleanup,
  - leaderboard submission packaging.

The numbers are stable because the recognizers are deterministic stubs:
  - ASR raw WER == 0.0 (fusion + non-speech mask recovers the reference exactly).
  - ASR normalized WER == 0.0.
  - DER baseline (misaligned boundaries) == 1/30 ≈ 0.0333.
  - DER after boundary snap == 0.0 (snap recovers all boundaries).
"""

from __future__ import annotations

import json
import zipfile

import pytest

from src.bench.synthetic import run_synthetic_e2e


def test_synthetic_pipeline_runs_end_to_end(tmp_path):
    report = run_synthetic_e2e(tmp_path / "syn")
    assert report.submission_zip.exists()


def test_synthetic_recovers_reference_after_fusion(tmp_path):
    report = run_synthetic_e2e(tmp_path / "syn")
    assert report.asr_raw_wer == pytest.approx(0.0, abs=1e-6)
    assert report.asr_normalized_wer == pytest.approx(0.0, abs=1e-6)


def test_synthetic_boundary_snap_reduces_no_collar_der(tmp_path):
    report = run_synthetic_e2e(tmp_path / "syn")
    # Snap MUST reduce DER on the deliberately misaligned input.
    assert report.snap_delta < 0.0
    # And the post-snap DER must be effectively zero (boundaries align with words).
    assert report.der_snapped == pytest.approx(0.0, abs=1e-6)


def test_synthetic_summary_is_persisted(tmp_path):
    out = tmp_path / "syn"
    run_synthetic_e2e(out)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {
        "asr_raw_wer",
        "asr_normalized_wer",
        "der_baseline_no_collar",
        "der_snapped_no_collar",
        "snap_delta",
        "submission_zip",
    } <= set(summary)


def test_synthetic_submission_has_correct_naming(tmp_path):
    report = run_synthetic_e2e(tmp_path / "syn")
    assert report.submission_zip.name == "UVigoBalideaBench_ASR_submission.zip"
    with zipfile.ZipFile(report.submission_zip) as zf:
        names = zf.namelist()
    assert names == ["synrec_001_fullaudio_transcrip.txt"]
    with zipfile.ZipFile(report.submission_zip) as zf:
        text = zf.read(names[0]).decode("utf-8").strip()
    # Non-speech token [risas] must be absent from the final transcript.
    assert "[risas]" not in text
    assert text == "buenas tardes señora cómo está usted hoy en el pueblo"
