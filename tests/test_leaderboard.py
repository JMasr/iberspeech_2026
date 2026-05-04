"""Deterministic tests for src.eval.leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.leaderboard import (
    DAILY_CAP,
    build_submission,
    record_submission,
    remaining_slots_today,
)


def _make_asr_dir(root: Path, record_ids: list[str]) -> Path:
    d = root / "asr"
    d.mkdir()
    for rec in record_ids:
        (d / f"{rec}_fullaudio_transcrip.txt").write_text("hola mundo\n", encoding="utf-8")
    return d


def _make_sd_dir(root: Path, record_ids: list[str]) -> Path:
    d = root / "sd"
    d.mkdir()
    for rec in record_ids:
        (d / f"{rec}.rttm").write_text(
            f"SPEAKER {rec} 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n",
            encoding="utf-8",
        )
    return d


def test_build_asr_submission_packages_zip(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["rec_001", "rec_002"])
    plan = build_submission(
        track="asr", hyp_dir=hyp, out_dir=tmp_path / "out", group_id="UVigoBalidea"
    )
    assert plan.out_zip.exists()
    assert plan.record_count == 2
    assert plan.record_ids == ["rec_001", "rec_002"]
    assert plan.digest  # hex


def test_build_sd_submission_packages_zip(tmp_path: Path):
    hyp = _make_sd_dir(tmp_path, ["rec_a", "rec_b"])
    plan = build_submission(
        track="sd", hyp_dir=hyp, out_dir=tmp_path / "out", group_id="UVigoBalidea"
    )
    assert plan.out_zip.name == "UVigoBalidea_SD_submission.zip"
    assert plan.record_count == 2


def test_invalid_track_rejected(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["a"])
    with pytest.raises(ValueError):
        build_submission(track="std", hyp_dir=hyp, out_dir=tmp_path, group_id="g")


def test_unexpected_record_id_rejected(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["rec_001", "rec_X"])
    with pytest.raises(ValueError):
        build_submission(
            track="asr",
            hyp_dir=hyp,
            out_dir=tmp_path / "out",
            group_id="g",
            expected_record_ids=["rec_001"],
        )


def test_missing_record_id_rejected(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["rec_001"])
    with pytest.raises(ValueError):
        build_submission(
            track="asr",
            hyp_dir=hyp,
            out_dir=tmp_path / "out",
            group_id="g",
            expected_record_ids=["rec_001", "rec_002"],
        )


def test_filename_pattern_enforced(tmp_path: Path):
    bad = tmp_path / "asr"
    bad.mkdir()
    (bad / "wrongname.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_submission(track="asr", hyp_dir=bad, out_dir=tmp_path / "out", group_id="g")


def test_record_submission_logs_and_decrements_slots(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["rec_1"])
    plan = build_submission(track="asr", hyp_dir=hyp, out_dir=tmp_path / "out", group_id="g")
    log = tmp_path / "log.jsonl"
    before = remaining_slots_today("asr", log_path=log)
    assert before == DAILY_CAP
    record_submission(plan, log_path=log)
    after = remaining_slots_today("asr", log_path=log)
    assert after == DAILY_CAP - 1
    line = log.read_text(encoding="utf-8").splitlines()[-1]
    entry = json.loads(line)
    assert entry["track"] == "asr"
    assert entry["record_count"] == 1


def test_daily_cap_blocks_third_record(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["rec_1"])
    plan = build_submission(track="asr", hyp_dir=hyp, out_dir=tmp_path / "out", group_id="g")
    log = tmp_path / "log.jsonl"
    for _ in range(DAILY_CAP):
        record_submission(plan, log_path=log)
    with pytest.raises(RuntimeError):
        record_submission(plan, log_path=log)


def test_dry_run_does_not_log(tmp_path: Path):
    hyp = _make_asr_dir(tmp_path, ["rec_1"])
    log = tmp_path / "log.jsonl"
    build_submission(track="asr", hyp_dir=hyp, out_dir=tmp_path / "out", group_id="g")
    assert not log.exists()
