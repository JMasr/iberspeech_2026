"""Convert COSER `.txt` soft labels to RTTM reference files.

Each `.txt` row has the form:

    START_TIME\tEND_TIME\tSPEAKER_ID: transcription

The output RTTM follows the standard NIST format consumed by
``src/eval/score_sd.py::score_directory``:

    SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker> <NA>

Run:
  python scripts/coser_labels_to_rttm.py \
      --labels data/raw/data_SD_track/train_dev/labels \
      --out    data/interim/sd_ref_rttm
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

LABEL_LINE = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([^:]+):\s*(.*)$")


def convert(label_path: Path, out_path: Path) -> int:
    """Convert one `.txt` soft-label file to RTTM. Returns the number of turns written."""
    rec_id = label_path.stem
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        m = LABEL_LINE.match(line)
        if not m:
            continue
        start = float(m.group(1))
        end = float(m.group(2))
        speaker = m.group(3).strip()
        dur = end - start
        if dur <= 0:
            continue
        rows.append(
            f"SPEAKER {rec_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory of COSER .txt soft labels.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for <rec>.rttm files.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    txt_paths = sorted(args.labels.glob("COSER-*.txt"))
    if not txt_paths:
        raise SystemExit(f"no COSER-*.txt found under {args.labels}")

    for txt in txt_paths:
        out_path = args.out / f"{txt.stem}.rttm"
        n = convert(txt, out_path)
        print(f"rttm: {txt.stem}  turns={n}")

    print(f"\n[done] {len(txt_paths)} rttm files → {args.out}")


if __name__ == "__main__":
    main()
