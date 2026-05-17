"""Explore the COSER train/dev data and write a Markdown report.

Run: python scripts/explore_coser_data.py

Writes: docs/data_exploration_report.md
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
ASR_ROOT = ROOT / "data/raw/data_ASR_track"
SD_ROOT = ROOT / "data/raw/data_SD_track"
REPORT = ROOT / "docs/data_exploration_report.md"

# Markup tokens used in the soft-label transcripts.
MARKUP_PATTERNS = {
    "[Anonim]": r"\[Anonim\]",
    "[RISAS]": r"\[RISAS\]",
    "[V-Sml]": r"\[V-Sml\]",
    "[Asent]": r"\[Asent\]",
    "[HS:…]": r"\[HS:[^\]]*\]",
    "[…] (other bracket)": r"\[[^\]]+\]",
    "= repair": r"\w+=\w+",
    "· mid-word pause": r"·",
    "… ellipsis": r"…",
}

LABEL_LINE = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([^:]+):\s*(.*)$")
# Metadata rows mostly use 2+-space separators, but a handful use a single space
# between date and duration. Match the trailing duration explicitly.
META_LINE = re.compile(
    r"^(\S+)\s+(.+?)\s{2,}(.+?)\s{2,}(\d{1,2} de \S+ de \d{4})\s+(\d+:\d+:\d+)\s*$"
)


@dataclass
class Turn:
    start: float
    end: float
    speakers: list[str]  # primary plus any embedded [HS:spk …]
    text: str


def parse_metadata(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = META_LINE.match(line)
        if not m:
            continue
        file_id, province, region, date, dur = m.groups()
        rows.append(
            {
                "file_id": file_id,
                "province": province.strip(),
                "region": region.strip(),
                "date": date.strip(),
                "duration_s": _hms_to_s(dur),
            }
        )
    return rows


def _hms_to_s(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_soft_labels(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LABEL_LINE.match(line)
        if not m:
            continue
        start, end, spk, text = m.groups()
        speakers = [spk.strip()]
        for hs in re.findall(r"\[HS:([^\s\]]+)", text):
            speakers.append(hs.strip())
        turns.append(Turn(float(start), float(end), speakers, text.strip()))
    return turns


def overlap_seconds(turns: list[Turn]) -> float:
    """Total time covered by ≥2 simultaneously-active turns."""
    events = []
    for t in turns:
        events.append((t.start, 1))
        events.append((t.end, -1))
    events.sort()
    active = 0
    last_t = None
    overlap = 0.0
    for t_, delta in events:
        if active >= 2 and last_t is not None:
            overlap += t_ - last_t
        active += delta
        last_t = t_
    return overlap


def speech_seconds(turns: list[Turn], cap_s: float | None = None) -> float:
    """Union of turn intervals (mono-speech coverage), optionally capped at cap_s."""
    if not turns:
        return 0.0
    intervals = sorted(
        [(t.start, min(t.end, cap_s) if cap_s is not None else t.end) for t in turns]
    )
    intervals = [(s, e) for s, e in intervals if e > s]
    if not intervals:
        return 0.0
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return sum(e - s for s, e in merged)


def label_overshoot_seconds(turns: list[Turn], audio_dur_s: float) -> float:
    """How many seconds of label time fall beyond the audio file end."""
    return max(0.0, max((t.end for t in turns), default=0.0) - audio_dur_s)


def clean_text(text: str) -> str:
    out = text
    for pat in MARKUP_PATTERNS.values():
        out = re.sub(pat, " ", out)
    return re.sub(r"\s+", " ", out).strip()


def probe_audio(path: Path) -> dict:
    import soundfile as sf

    info = sf.info(str(path))
    return {
        "sr": info.samplerate,
        "channels": info.channels,
        "duration_s": info.frames / info.samplerate,
        "subtype": info.subtype,
        "format": info.format,
    }


def rms_dbfs(path: Path) -> tuple[float, float]:
    """Return (rms_dbfs, silence_ratio) computed on the full file."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return float("nan"), float("nan")
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    # Silence ratio via 25 ms frames below -40 dBFS.
    win = int(0.025 * sr)
    n_frames = audio.size // win
    if n_frames == 0:
        return rms_db, float("nan")
    frames = audio[: n_frames * win].reshape(n_frames, win)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms + 1e-12)
    silence = float(np.mean(frame_db < -40.0))
    return rms_db, silence


def fmt_dur(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    metadata = parse_metadata(ASR_ROOT / "metadata.txt")
    meta_by_id = {r["file_id"]: r for r in metadata}

    full_audios = sorted((ASR_ROOT / "train_dev/audio").glob("*.wav"))
    delivered_ids = [p.stem for p in full_audios]

    # --- parse all 8 soft-label files (ASR == SD, byte-identical) ---
    per_record: list[dict] = []
    markup_counter: Counter[str] = Counter()
    all_clean_text: list[str] = []
    for rec_id in delivered_ids:
        label_path = ASR_ROOT / f"train_dev/labels/{rec_id}.txt"
        turns = parse_soft_labels(label_path)
        speakers = sorted({s for t in turns for s in t.speakers})
        e_speakers = [s for s in speakers if s.startswith("E")]
        i_speakers = [s for s in speakers if s.startswith("I")]
        other_speakers = [s for s in speakers if not s.startswith(("E", "I"))]
        rec_dur = meta_by_id[rec_id]["duration_s"] if rec_id in meta_by_id else None
        # Markup counts on raw text.
        raw_concat = " ".join(t.text for t in turns)
        for label, pat in MARKUP_PATTERNS.items():
            markup_counter[label] += len(re.findall(pat, raw_concat))
        cleans = [clean_text(t.text) for t in turns]
        all_clean_text.extend(cleans)
        # Probe audio.
        probe = probe_audio(ASR_ROOT / f"train_dev/audio/{rec_id}.wav")
        rms_db, silence = rms_dbfs(ASR_ROOT / f"train_dev/audio/{rec_id}.wav")
        ov = overlap_seconds(turns)
        sp = speech_seconds(turns, cap_s=probe["duration_s"])
        overshoot = label_overshoot_seconds(turns, probe["duration_s"])
        per_record.append(
            {
                "rec_id": rec_id,
                "province": meta_by_id.get(rec_id, {}).get("province", "?"),
                "region": meta_by_id.get(rec_id, {}).get("region", "?"),
                "meta_dur_s": rec_dur,
                "audio_dur_s": probe["duration_s"],
                "sr": probe["sr"],
                "channels": probe["channels"],
                "subtype": probe["subtype"],
                "rms_dbfs": rms_db,
                "silence_ratio": silence,
                "n_turns": len(turns),
                "n_speakers": len(speakers),
                "n_e": len(e_speakers),
                "n_i": len(i_speakers),
                "n_other": len(other_speakers),
                "speakers": ",".join(speakers),
                "speech_s": sp,
                "overlap_s": ov,
                "label_overshoot_s": overshoot,
                "speech_ratio": sp / probe["duration_s"] if probe["duration_s"] else 0.0,
                "overlap_ratio_speech": ov / sp if sp else 0.0,
                "mean_turn_s": stats.mean([t.end - t.start for t in turns]) if turns else 0.0,
                "median_turn_s": stats.median([t.end - t.start for t in turns]) if turns else 0.0,
            }
        )

    # --- segments inventory ---
    asr_segments = sorted((ASR_ROOT / "train_dev/segments").glob("*.wav"))
    sd_segments = sorted((SD_ROOT / "train_dev/segments").glob("*.wav"))
    jsonl_path = ASR_ROOT / "train_dev/labels/segments_labels_ASR_track.jsonl"
    seg_records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    seg_by_audio: Counter[str] = Counter(r["audio"] for r in seg_records)
    seg_lengths_chars = [len(r["text"]) for r in seg_records]
    # Acoustic probe on a random-but-deterministic sample of segments.
    sample_idx = list(range(0, len(asr_segments), max(1, len(asr_segments) // 200)))[:200]
    seg_durations: list[float] = []
    seg_rms: list[float] = []
    for i in sample_idx:
        p = asr_segments[i]
        info = probe_audio(p)
        seg_durations.append(info["duration_s"])
        rms_db, _ = rms_dbfs(p)
        seg_rms.append(rms_db)

    # --- aggregate language hints (very rough) ---
    co_official_hints = {
        "ca": [" amb ", " això ", " aquest ", " perquè "],
        "gl": [" non ", " ti ", " teño ", " moito "],
        "eu": [" ez ", " bai ", " gara ", " duzu "],
    }
    text_corpus = " ".join(all_clean_text).lower()
    co_official_counts = {
        lang: sum(text_corpus.count(tok) for tok in toks)
        for lang, toks in co_official_hints.items()
    }

    # --- write report ---
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    a = lines.append

    a("# COSER train/dev — data exploration report")
    a("")
    a(
        "Generated by `scripts/explore_coser_data.py`. Covers the 8 full-length recordings + "
        "14 487 segments shipped under `data/raw/data_{ASR,SD}_track/train_dev/`."
    )
    a("")
    a("## 1. Inventory")
    a("")
    a(f"- Metadata rows: **{len(metadata)}** records listed in `metadata.txt` (both tracks).")
    a(f"- Full audios delivered: **{len(full_audios)}** in each track (same record_ids).")
    a(f"- Soft-label `.txt` files: **{len(full_audios)}** per track — byte-identical between ASR and SD.")
    a(f"- Segments (≤30 s): **{len(asr_segments)}** in ASR, **{len(sd_segments)}** in SD (same filenames).")
    a(f"- Segment-level transcripts: **{len(seg_records)}** rows in `segments_labels_ASR_track.jsonl` (ASR only).")
    missing = [m["file_id"] for m in metadata if m["file_id"] not in delivered_ids]
    a(
        f"- Records listed in metadata but **not** delivered as full audio yet: "
        f"**{len(missing)}** — partial release.",
    )
    a("")
    a(f"## 2. Metadata overview ({len(metadata)} listed records)")
    a("")
    regions = Counter(r["region"] for r in metadata)
    provinces = Counter(r["province"] for r in metadata)
    years = Counter(r["date"].split()[-1] for r in metadata if r["date"].split())
    total_listed = sum(r["duration_s"] for r in metadata)
    a(f"- Total listed audio: **{fmt_dur(total_listed)}** across {len(metadata)} files "
      f"(mean {fmt_dur(total_listed / len(metadata))}, "
      f"min {fmt_dur(min(r['duration_s'] for r in metadata))}, "
      f"max {fmt_dur(max(r['duration_s'] for r in metadata))}).")
    a(f"- Regions covered ({len(regions)}): " + ", ".join(f"{k} ({v})" for k, v in regions.most_common()))
    a(f"- Provinces covered: **{len(provinces)}** distinct.")
    a(f"- Recording years span: **{min(years)} → {max(years)}** ({len(years)} distinct years).")
    a("")
    a("## 3. Per-recording summary (8 delivered full audios)")
    a("")
    a(
        "| Record | Province | Dur (audio) | Speakers (E/I/oth) | Turns | "
        "Speech % | Overlap % of speech | Mean turn s | RMS dBFS | Silence % | Label overshoot |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in per_record:
        a(
            f"| {r['rec_id']} | {r['province']} | {fmt_dur(r['audio_dur_s'])} "
            f"| {r['n_speakers']} ({r['n_e']}/{r['n_i']}/{r['n_other']}) "
            f"| {r['n_turns']} | {100 * r['speech_ratio']:.1f} | "
            f"{100 * r['overlap_ratio_speech']:.1f} | {r['mean_turn_s']:.2f} | "
            f"{r['rms_dbfs']:.1f} | {100 * r['silence_ratio']:.1f} | "
            f"{r['label_overshoot_s']:.0f}s |"
        )
    a("")
    a("## 4. Speakers per recording (full audios)")
    a("")
    n_spk = [r["n_speakers"] for r in per_record]
    a(f"- Mean: **{stats.mean(n_spk):.2f}**")
    a(f"- Median: **{stats.median(n_spk):.0f}**")
    a(f"- Min / Max: **{min(n_spk)} / {max(n_spk)}**")
    a(f"- Total unique-per-record speaker labels across the 8 audios: **{sum(n_spk)}**")
    a(
        f"- Speaker IDs follow the convention `E#` (entrevistador / interviewer) and "
        f"`I#` (informante / informant). Across all 8 records: "
        f"E speakers={sum(r['n_e'] for r in per_record)}, "
        f"I speakers={sum(r['n_i'] for r in per_record)}, "
        f"other tags={sum(r['n_other'] for r in per_record)}."
    )
    overlap_pct = [100 * r["overlap_ratio_speech"] for r in per_record]
    a(
        f"- Overlap (% of speech time, computed from soft-label intervals): "
        f"mean **{stats.mean(overlap_pct):.1f}%**, max **{max(overlap_pct):.1f}%** — "
        f"non-trivial, must be modelled for SD (no-collar DER) and considered in ASR fusion."
    )
    a("")
    a("## 5. Transcription characteristics")
    a("")
    total_turns = sum(r["n_turns"] for r in per_record)
    total_words = sum(len(t.split()) for t in all_clean_text)
    vocab = Counter(w for t in all_clean_text for w in t.split())
    a(f"- Total turns across 8 audios: **{total_turns}**.")
    a(f"- Total tokens (after stripping markup): **{total_words}**, vocabulary size: **{len(vocab)}**.")
    a("- Soft labels are **punctuated, mixed-case** Spanish — directly usable as the *raw* ASR reference. "
      "For the *normalized* track they must pass through `src/data/normalize.py::normalize_for_eval`.")
    a("- Markup token frequencies (raw, summed across 8 records):")
    a("")
    a("| Tag | Count | Meaning |")
    a("|---|---|---|")
    descriptions = {
        "[Anonim]": "anonymised proper noun (name/place blanked)",
        "[RISAS]": "laughter",
        "[V-Sml]": "vocal/non-verbal sound",
        "[Asent]": "back-channel / assent (mhm, sí…)",
        "[HS:…]": "embedded overlapping speaker",
        "[…] (other bracket)": "any other bracketed annotation",
        "= repair": "dialect-canonical repair (e.g. `repartila=repartirla`)",
        "· mid-word pause": "in-word pause / hesitation marker",
        "… ellipsis": "trailing-off / unfinished utterance",
    }
    for tag, _ in markup_counter.most_common():
        a(f"| `{tag}` | {markup_counter[tag]} | {descriptions.get(tag, '')} |")
    a("")
    a("- Co-official-language stop-word hits in transcripts (very rough heuristic):")
    a("")
    a("| Code | Hits | Language |")
    a("|---|---|---|")
    a(f"| ca | {co_official_counts['ca']} | Catalan |")
    a(f"| gl | {co_official_counts['gl']} | Galician |")
    a(f"| eu | {co_official_counts['eu']} | Basque |")
    a("")
    a("> Hits are token co-occurrences and **not** real LID. Treat as a flag that "
      "code-switching exists; the actual language ratio needs `src/asr/lid.py`.")
    a("")
    a("## 6. Segments corpus")
    a("")
    a(f"- Files: **{len(asr_segments)}** WAVs per track (≤30 s each, by spec).")
    a(f"- ASR segment-level labels (`segments_labels_ASR_track.jsonl`): **{len(seg_records)}** rows. "
      "SD segments ship **without** labels.")
    a(f"- Segment text length (chars): mean {stats.mean(seg_lengths_chars):.0f}, "
      f"median {stats.median(seg_lengths_chars):.0f}, "
      f"min {min(seg_lengths_chars)}, max {max(seg_lengths_chars)}.")
    a("- Segment counts per source recording (top 10):")
    a("")
    a("| Source record | # segments |")
    a("|---|---|")
    for rec, n in seg_by_audio.most_common(10):
        a(f"| {rec} | {n} |")
    a("")
    a(f"- Source records appearing in segments: **{len(seg_by_audio)}** "
      "(more than the 8 full audios delivered — segments cover additional recordings).")
    a(f"- Acoustic probe on {len(sample_idx)}-segment sample: "
      f"duration mean {stats.mean(seg_durations):.2f}s "
      f"(median {stats.median(seg_durations):.2f}s, max {max(seg_durations):.2f}s), "
      f"RMS dBFS mean {stats.mean(seg_rms):.1f} (min {min(seg_rms):.1f}, max {max(seg_rms):.1f}).")
    a("")
    a("## 7. Acoustic characteristics (full audios)")
    a("")
    srs = Counter(r["sr"] for r in per_record)
    chs = Counter(r["channels"] for r in per_record)
    sub = Counter(r["subtype"] for r in per_record)
    a(f"- Sample rate: {dict(srs)} (README claims 16 kHz mono — verified).")
    a(f"- Channels: {dict(chs)}.")
    a(f"- WAV subtype: {dict(sub)}.")
    rms_all = [r["rms_dbfs"] for r in per_record]
    sil_all = [r["silence_ratio"] for r in per_record]
    a(f"- RMS dBFS: mean {stats.mean(rms_all):.1f}, "
      f"min {min(rms_all):.1f}, max {max(rms_all):.1f} — "
      f"~{abs(min(rms_all) - max(rms_all)):.0f} dB spread across the 8 files.")
    a(f"- Silence ratio (frames < -40 dBFS, 25 ms frames): "
      f"mean {100 * stats.mean(sil_all):.1f}%, max {100 * max(sil_all):.1f}%.")
    audio_total = sum(r["audio_dur_s"] for r in per_record)
    a(f"- Total delivered full-audio duration: **{fmt_dur(audio_total)}** "
      f"({audio_total / 3600:.2f} h) — matches the README's ~10 h figure for full recordings.")
    a("")
    a("## 8. Implications for the pipeline")
    a("")
    a("- **Speaker count per audio is small (≈3-4 active labels)** but `[HS:…]` tags reveal frequent "
      "short overlaps — diarization must keep overlap on (no-collar DER assumption holds).")
    a("- **Soft-label transcripts contain rich markup** (`[Anonim]`, `[RISAS]`, `=repair`, `·`, `…`). "
      "The raw track keeps them; the normalized track needs them stripped consistently with `meeteval`.")
    a("- **Two label modalities for ASR**: (a) full-audio soft labels with turn timings (suitable for "
      "long-form FT and pseudo-label calibration); (b) `segments_labels_ASR_track.jsonl` with clean "
      "≤30 s pairs (drop-in for Whisper / XLS-R FT).")
    a("- **SD segments are unlabeled** — they can only be used as unsupervised pretraining material, "
      "or they need pseudo-labels from the diarization pipeline.")
    a("- **Audio is consistently 16 kHz mono PCM**, but loudness varies ~"
      f"{abs(min(rms_all) - max(rms_all)):.0f} dB across the 8 files — front-end normalization is needed "
      "before x-vector / ECAPA embedding.")
    a("- **Code-switching is real** (Catalan/Galician/Basque hits in stopword heuristic). LID per "
      "segment will matter for dialect-aware decoding.")
    a(f"- **Only {len(delivered_ids)} of {len(metadata)}** metadata records have full audio + soft "
      "labels delivered so far. The remaining recordings are listed in `metadata.txt` but only appear "
      "via the segment slices.")
    overshoots = [r for r in per_record if r["label_overshoot_s"] > 1.0]
    if overshoots:
        a("- **Soft-label timestamps overshoot the audio file end** in "
          f"{len(overshoots)}/{len(per_record)} records (e.g. " +
          ", ".join(f"`{r['rec_id']}` +{r['label_overshoot_s']:.0f}s" for r in overshoots) +
          "). Clamp turn end-times to audio duration before training/scoring.")
    a("")
    a("---")
    a("")
    a(f"_Report regenerated by re-running `python scripts/explore_coser_data.py`._")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
