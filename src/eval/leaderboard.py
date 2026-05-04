"""Submission packaging and per-day dedup tracker.

ALBAYZIN allows 2 submissions per track per day. This module:

- Validates the filename conventions (``<rec>_fullaudio_transcrip.txt`` for ASR,
  ``<rec>.rttm`` for SD).
- Builds the submission zip with the exact expected name
  ``<group_id>_<TRACK>_submission.zip``.
- Records every submission in ``experiments/submission_log.jsonl`` so we can
  see at a glance how many of today's slots are used.
- Refuses to record a submission unless ``--confirm`` is passed; the default
  is dry-run (build the zip, validate, but DO NOT count it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

VALID_TRACKS = {"asr", "sd"}
ASR_FILENAME_RE = re.compile(r"^(?P<rec>[A-Za-z0-9_\-]+)_fullaudio_transcrip\.txt$")
SD_FILENAME_RE = re.compile(r"^(?P<rec>[A-Za-z0-9_\-]+)\.rttm$")
DAILY_CAP = 2


@dataclass
class SubmissionPlan:
    track: str
    files: list[Path]
    record_ids: list[str]
    out_zip: Path
    digest: str = ""
    timestamp: str = ""
    record_count: int = 0
    metadata: dict = field(default_factory=dict)


def build_submission(
    *,
    track: str,
    hyp_dir: str | Path,
    out_dir: str | Path,
    group_id: str,
    expected_record_ids: list[str] | None = None,
) -> SubmissionPlan:
    """Validate hypotheses and stage the submission zip in ``out_dir``.

    Does NOT record the submission in the log; that's ``record_submission``'s job.
    """
    if track not in VALID_TRACKS:
        raise ValueError(f"track must be in {VALID_TRACKS}, got {track!r}")

    hyp_dir = Path(hyp_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = ASR_FILENAME_RE if track == "asr" else SD_FILENAME_RE
    glob_pat = "*_fullaudio_transcrip.txt" if track == "asr" else "*.rttm"

    files: list[Path] = []
    record_ids: list[str] = []
    for path in sorted(hyp_dir.glob(glob_pat)):
        m = pattern.match(path.name)
        if not m:
            raise ValueError(
                f"{path} does not match the expected {'ASR' if track == 'asr' else 'SD'} naming "
                f"pattern."
            )
        files.append(path)
        record_ids.append(m.group("rec"))

    if not files:
        raise FileNotFoundError(f"No {track} hypotheses found under {hyp_dir}")
    if expected_record_ids is not None:
        missing = sorted(set(expected_record_ids) - set(record_ids))
        if missing:
            raise ValueError(f"Missing record_ids in submission: {missing}")
        extra = sorted(set(record_ids) - set(expected_record_ids))
        if extra:
            raise ValueError(f"Unexpected record_ids in submission: {extra}")

    out_zip = out_dir / f"{group_id}_{track.upper()}_submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)

    digest = _sha256(out_zip)
    return SubmissionPlan(
        track=track,
        files=files,
        record_ids=record_ids,
        out_zip=out_zip,
        digest=digest,
        timestamp=datetime.now(timezone.utc).isoformat(),
        record_count=len(files),
    )


def record_submission(
    plan: SubmissionPlan, *, log_path: str | Path = "experiments/submission_log.jsonl"
) -> dict:
    """Append the plan to the log. Refuses if today's cap for this track is exceeded.

    The log is the source of truth for how many slots remain today.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used_today = 0
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("track") == plan.track and entry.get("date") == today:
                used_today += 1
    if used_today >= DAILY_CAP:
        raise RuntimeError(
            f"Daily cap reached for track {plan.track!r} on {today} ({DAILY_CAP} submissions). "
            "Wait until UTC tomorrow or remove the most recent log entry to retry."
        )

    entry = {
        "date": today,
        "timestamp": plan.timestamp,
        "track": plan.track,
        "zip": str(plan.out_zip),
        "digest": plan.digest,
        "record_count": plan.record_count,
        "record_ids": plan.record_ids,
        "metadata": plan.metadata,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def remaining_slots_today(
    track: str, *, log_path: str | Path = "experiments/submission_log.jsonl"
) -> int:
    log_path = Path(log_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used = 0
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("track") == track and entry.get("date") == today:
                used += 1
    return max(0, DAILY_CAP - used)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()
