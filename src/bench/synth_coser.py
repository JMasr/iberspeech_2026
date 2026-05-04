"""Synth-COSER — a custom HF-sourced dataset that mimics COSER structure.

Each recording is a multi-turn, multi-speaker, optionally multilingual
"interview-like" audio file built by concatenating Common Voice clips.
Mirrors the COSER characteristics that matter for validating our pipeline
hypotheses BEFORE the official corpus drops:

  - **2 speakers** alternating turns, like the COSER interviewer / interviewee
  - **Spanish primary** + optional Galician / Catalan / Basque code-switching
  - **Spanish accent stratification** via Common Voice ``accents`` (castilian /
    andalusian / latin-american / argentinian / mexican / chilean)
  - **Silence padding** between turns (probes VAD lift)
  - **Optional synthetic music interlude** mid-recording (probes music routing)
  - **5-minute default duration** (fast iteration; configurable up to ~50 min)
  - **Reference transcript** (one line per turn) and **RTTM** diarization

Outputs per recording (under ``data/bench/synth_coser/<rec>/``):
  - ``<rec>.wav`` — 16 kHz mono float32 PCM, concatenated turns + silence
  - ``<rec>.rttm`` — speaker turns (one segment per turn, no overlap by default)
  - ``<rec>.txt`` — reference transcript
  - parent ``manifest.parquet`` rows tagged ``source = "synth_coser"``

**Eval-rule compliance:** every manifest row carries ``source = "synth_coser"``
which `src/bench/__init__.py:assert_no_bench_rows` rejects from FT. This
dataset is for **validation only** and cannot leak into COSER fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

DEFAULT_OUT = Path("data/bench/synth_coser")
DEFAULT_PRIMARY_LANG = "es"
DEFAULT_SECONDARY_LANGS: tuple[str, ...] = ()
SAMPLE_RATE = 16_000


# Open-license multilingual sources, per language. Each entry is a recipe:
#   dataset, config (or None), split, transcript_key, speaker_key (or None)
# We deliberately avoid CV 11.0 because it became gated. These are all open
# parquet-format datasets streamable via HF without auth.
SOURCE_REGISTRY: dict[str, dict] = {
    "es": {
        "dataset": "facebook/voxpopuli",
        "config": "es",
        "split": "test",
        "transcript_key": "raw_text",
        "speaker_key": "speaker_id",
        "accent_key": "gender",  # voxpopuli has no accent; we reuse gender as a coarse stratifier
    },
    "ca": {
        "dataset": "shunyalabs/catalan-speech-dataset",
        "config": None,
        "split": "train",
        "transcript_key": "transcript",
        "speaker_key": None,  # no speaker metadata; secondary-language clips don't need it
        "accent_key": None,
    },
    "gl": {
        "dataset": "shunyalabs/galician-speech-dataset",
        "config": None,
        "split": "train",
        "transcript_key": "transcript",
        "speaker_key": None,
        "accent_key": None,
    },
    # Basque (eu) — no open-license HF dataset found. Add when one is available.
}
DEFAULT_CV_DATASET = SOURCE_REGISTRY["es"]["dataset"]  # back-compat for older configs


@dataclass
class SynthCOSERConfig:
    # What to build
    target_duration_min: float = 5.0
    n_speakers: int = 2
    n_recordings: int = 3

    # Language mix
    primary_language: str = DEFAULT_PRIMARY_LANG
    secondary_languages: tuple[str, ...] = DEFAULT_SECONDARY_LANGS
    code_switch_probability: float = 0.0
    primary_accent_filter: tuple[str, ...] | None = None  # e.g. ("castilian",) for ES-only

    # Audio shaping
    silence_between_turns_s: tuple[float, float] = (0.3, 1.5)
    music_interlude: bool = False
    music_position_s: float = 60.0
    music_duration_s: float = 30.0
    music_source_path: str | None = None  # 16k mono WAV; None → synthetic sines

    # HF source
    cv_dataset: str = DEFAULT_CV_DATASET
    cv_split: str = "test"
    pool_size_per_language: int = 200  # how many clips to stream per language
    min_clips_per_speaker: int = 4  # speakers with fewer clips are dropped

    # Output
    out_dir: Path = DEFAULT_OUT
    seed: int = 20260503
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clip pool loading
# ---------------------------------------------------------------------------


def _load_clip_pool(
    language: str,
    pool_size: int,
    *,
    cv_dataset: str | None = None,
    cv_split: str | None = None,
    cache_dir: Path,
    accent_filter: tuple[str, ...] | None = None,
) -> list[dict]:  # pragma: no cover (network)
    """Stream ``pool_size`` clips from the registry source for ``language``.

    Materializes WAVs to ``cache_dir`` and writes a ``pool.parquet`` manifest
    so subsequent calls reuse the on-disk audio. ``cv_dataset`` / ``cv_split``
    overrides are for back-compat with older configs that point at a specific
    Common Voice variant.

    Uses ``Audio(decode=False)`` + ``soundfile`` to bypass the torchcodec
    requirement (we only need numpy float32 arrays).
    """
    import io

    from datasets import Audio, load_dataset
    import pandas as pd
    import soundfile as sf

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "pool.parquet"
    if manifest.exists():
        cached = pd.read_parquet(manifest)
        if len(cached) >= pool_size:
            rows = cached.head(pool_size).to_dict("records")
            if accent_filter:
                af = set(accent_filter)
                rows = [r for r in rows if r["accent"] in af]
            return rows

    src = SOURCE_REGISTRY.get(language)
    if src is None:
        raise ValueError(
            f"No open-license HF source registered for language {language!r}. "
            "Add an entry to SOURCE_REGISTRY in src/bench/synth_coser.py."
        )
    dataset_name = cv_dataset or src["dataset"]
    cfg = src["config"]
    split = cv_split or src["split"]
    transcript_key = src["transcript_key"]
    speaker_key = src.get("speaker_key")
    accent_key = src.get("accent_key")

    ds = (
        load_dataset(dataset_name, cfg, split=split, streaming=True)
        if cfg
        else load_dataset(dataset_name, split=split, streaming=True)
    )
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))

    rows: list[dict] = []
    for i, sample in enumerate(ds):
        if i >= pool_size:
            break
        audio_field = sample["audio"]
        if not (isinstance(audio_field, dict) and audio_field.get("bytes")):
            continue
        try:
            wav, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            continue
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        # Resample to 16k if needed (rare — most sources are already 16k).
        if sr != SAMPLE_RATE:
            try:
                import librosa  # noqa: F401

                wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE).astype("float32")
            except ImportError:
                # Without librosa, accept the native rate and resample later.
                pass
            else:
                sr = SAMPLE_RATE
        wav_path = cache_dir / f"{language}_{i:05d}.wav"
        sf.write(str(wav_path), wav, sr)
        accent_raw = sample.get(accent_key) if accent_key else None
        accent = (str(accent_raw).lower().strip().replace(" ", "_") if accent_raw else "_")
        if accent_filter and accent not in set(accent_filter):
            wav_path.unlink(missing_ok=True)
            continue
        duration = len(wav) / sr
        speaker_id = (
            str(sample.get(speaker_key))
            if speaker_key and sample.get(speaker_key) is not None
            else f"{language}_speaker_{i}"  # synthetic per-clip id when source has no speaker meta
        )
        rows.append(
            {
                "segment_id": wav_path.stem,
                "audio_path": str(wav_path),
                "transcript": str(sample.get(transcript_key, "")),
                "duration_s": float(duration),
                "speaker_id": speaker_id,
                "accent": accent,
                "language": language,
                "source_dataset": dataset_name,
            }
        )

    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return rows


# ---------------------------------------------------------------------------
# Speaker pool / turn sequencing — pure Python, fully testable
# ---------------------------------------------------------------------------


def group_by_speaker(clips: list[dict], min_clips: int) -> dict[str, list[dict]]:
    """Group clips by ``speaker_id``, drop speakers with fewer than ``min_clips``."""
    by: dict[str, list[dict]] = {}
    for c in clips:
        by.setdefault(c["speaker_id"], []).append(c)
    return {sid: cs for sid, cs in by.items() if len(cs) >= min_clips}


def pick_speakers(
    speaker_pool: dict[str, list[dict]],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Pick ``n`` distinct speaker_ids uniformly at random."""
    if len(speaker_pool) < n:
        raise ValueError(
            f"Need {n} speakers but only {len(speaker_pool)} have ≥min_clips. "
            "Increase pool_size_per_language or lower min_clips_per_speaker."
        )
    return rng.sample(sorted(speaker_pool.keys()), n)


@dataclass
class Turn:
    speaker_id: str
    language: str
    audio_path: str
    transcript: str
    duration_s: float


def build_turn_sequence(
    speakers: list[str],
    primary_pool: dict[str, list[dict]],
    secondary_pools: dict[str, list[dict]],
    *,
    target_duration_s: float,
    silence_range: tuple[float, float],
    code_switch_prob: float,
    rng: random.Random,
) -> list[Turn]:
    """Generate alternating-speaker turns until target duration is reached.

    Code-switch turns: with probability ``code_switch_prob`` and a non-empty
    ``secondary_pools``, the next turn picks a clip from a randomly chosen
    secondary language. The speaker_id is preserved (i.e., the same person
    "switches language" — typical of bilingual rural speakers).

    Returns the list of turns. The total duration including silence padding is
    approximately ``target_duration_s`` (slightly over by one turn).
    """
    used_per_speaker: dict[str, set[str]] = {sid: set() for sid in speakers}
    turns: list[Turn] = []
    accumulated = 0.0
    speaker_idx = 0
    silence_lo, silence_hi = silence_range
    secondary_keys = list(secondary_pools.keys())

    while True:
        speaker = speakers[speaker_idx % len(speakers)]
        speaker_idx += 1

        # Add the silence-between-turns BEFORE this turn (only after turn 1).
        if turns:
            accumulated += rng.uniform(silence_lo, silence_hi)

        # Decide language for this turn.
        if secondary_keys and rng.random() < code_switch_prob:
            lang = rng.choice(secondary_keys)
            pool = secondary_pools[lang]
            # Pick any clip from the secondary language pool.
            clip = rng.choice(pool)
        else:
            lang = "primary"
            available = [
                c
                for c in primary_pool[speaker]
                if c["segment_id"] not in used_per_speaker[speaker]
            ]
            if not available:
                # Refill: this speaker exhausted unique clips. Allow reuse.
                available = primary_pool[speaker]
            clip = rng.choice(available)
            used_per_speaker[speaker].add(clip["segment_id"])
            lang = clip["language"]

        turns.append(
            Turn(
                speaker_id=speaker,
                language=lang,
                audio_path=clip["audio_path"],
                transcript=clip["transcript"],
                duration_s=clip["duration_s"],
            )
        )
        accumulated += clip["duration_s"]

        # Exit once the assembled-audio duration meets the target.
        if accumulated >= target_duration_s:
            break

    return turns


# ---------------------------------------------------------------------------
# Audio assembly + RTTM + transcript
# ---------------------------------------------------------------------------


def assemble_recording(
    turns: list[Turn],
    *,
    silence_range: tuple[float, float],
    rng: random.Random,
    sr: int = SAMPLE_RATE,
):
    """Concatenate turn audios with silence padding. Return audio + RTTM segments + transcript lines.

    Heavy imports are local. Returns:
      audio (np.ndarray float32),
      rttm_segments (list[dict] with start_s/end_s/speaker),
      transcript_lines (list[str], aligned with turns),
      timeline (list[dict] with t_start, t_end, kind ∈ {speech, silence}).
    """
    import numpy as np
    import soundfile as sf

    chunks: list = []
    rttm_segments: list[dict] = []
    transcript_lines: list[str] = []
    timeline: list[dict] = []
    cursor_s = 0.0
    silence_lo, silence_hi = silence_range
    for i, turn in enumerate(turns):
        audio, file_sr = sf.read(turn.audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            import librosa  # noqa: F401  (will rarely hit; CV is 16k or 48k)

            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr).astype("float32")
        # Append a silence pad before this turn (except the first).
        if i > 0:
            silence_dur = rng.uniform(silence_lo, silence_hi)
            n_silence = int(round(silence_dur * sr))
            chunks.append(np.zeros(n_silence, dtype="float32"))
            timeline.append(
                {"t_start": cursor_s, "t_end": cursor_s + silence_dur, "kind": "silence"}
            )
            cursor_s += silence_dur
        chunks.append(audio)
        turn_dur = audio.shape[0] / sr
        rttm_segments.append(
            {"start_s": cursor_s, "end_s": cursor_s + turn_dur, "speaker": turn.speaker_id}
        )
        transcript_lines.append(turn.transcript.strip())
        timeline.append({"t_start": cursor_s, "t_end": cursor_s + turn_dur, "kind": "speech"})
        cursor_s += turn_dur

    if not chunks:
        audio_out = np.zeros(0, dtype="float32")
    else:
        audio_out = np.concatenate(chunks)

    return audio_out, rttm_segments, transcript_lines, timeline


def inject_music_interlude(
    audio,
    *,
    position_s: float,
    duration_s: float,
    sr: int = SAMPLE_RATE,
    music_source: str | Path | None = None,
):
    """Replace ``[position_s, position_s + duration_s]`` of ``audio`` with music.

    If ``music_source`` is a path to a 16 kHz mono WAV (e.g. an AudioSet music
    clip cached under ``data/bench/synth_coser/_music_pool/``), that audio is
    spliced in (looped if too short, truncated if too long).

    Otherwise we fall back to a 4-tone harmonic synthesis — useful for unit
    tests that must run without external music files. The H5 hypothesis test
    only fires when ``music_source`` is real.

    Returns ``(modified_audio, music_interval)``.
    """
    import numpy as np

    audio = np.asarray(audio, dtype="float32").copy()
    a = int(round(position_s * sr))
    b = int(round((position_s + duration_s) * sr))
    b = min(b, audio.shape[0])
    if b <= a:
        return audio, (position_s, position_s)
    n = b - a

    if music_source is not None:
        import soundfile as sf

        clip, clip_sr = sf.read(str(music_source), dtype="float32", always_2d=False)
        if clip.ndim == 2:
            clip = clip.mean(axis=1)
        if clip_sr != sr:
            raise ValueError(
                f"music_source sample rate {clip_sr} != target {sr}; "
                "resample before injection."
            )
        if clip.size == 0:
            return audio, (position_s, position_s)
        # Loop or truncate the clip to fill exactly n samples.
        if clip.size >= n:
            music = clip[:n].astype("float32")
        else:
            reps = (n + clip.size - 1) // clip.size
            music = np.tile(clip, reps)[:n].astype("float32")
    else:
        t = np.arange(n) / sr
        music = np.zeros(n, dtype="float32")
        for f, amp in [(220.0, 0.10), (330.0, 0.07), (440.0, 0.05), (550.0, 0.04)]:
            music += (amp * np.sin(2 * np.pi * f * t)).astype("float32")

    audio[a:b] = music
    return audio, (a / sr, b / sr)


def write_rttm(record_id: str, segments: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in segments:
        dur = float(s["end_s"]) - float(s["start_s"])
        if dur <= 0:
            continue
        lines.append(
            f"SPEAKER {record_id} 1 {float(s['start_s']):.3f} {dur:.3f} "
            f"<NA> <NA> {s['speaker']} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_recording(
    cfg: SynthCOSERConfig,
    primary_pool: dict[str, list[dict]],
    secondary_pools: dict[str, list[dict]],
    *,
    record_id: str,
    rng: random.Random,
) -> dict:
    """Build one synth-COSER recording. Returns a manifest row."""
    import soundfile as sf

    speakers = pick_speakers(primary_pool, cfg.n_speakers, rng)
    target_s = cfg.target_duration_min * 60.0
    turns = build_turn_sequence(
        speakers,
        primary_pool,
        secondary_pools,
        target_duration_s=target_s,
        silence_range=cfg.silence_between_turns_s,
        code_switch_prob=cfg.code_switch_probability,
        rng=rng,
    )
    audio, rttm_segments, transcript_lines, timeline = assemble_recording(
        turns, silence_range=cfg.silence_between_turns_s, rng=rng
    )

    music_interval = None
    if cfg.music_interlude:
        audio, music_interval = inject_music_interlude(
            audio,
            position_s=cfg.music_position_s,
            duration_s=cfg.music_duration_s,
            music_source=cfg.music_source_path,
        )

    out_dir = cfg.out_dir / record_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{record_id}.wav"
    rttm_path = out_dir / f"{record_id}.rttm"
    txt_path = out_dir / f"{record_id}.txt"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    write_rttm(record_id, rttm_segments, rttm_path)
    txt_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

    code_switch_count = sum(
        1 for t in turns if t.language not in {cfg.primary_language, "primary"}
    )
    return {
        "record_id": record_id,
        "audio_path": str(wav_path),
        "rttm_path": str(rttm_path),
        "transcript_path": str(txt_path),
        "duration_s": float(audio.shape[0]) / SAMPLE_RATE,
        "n_speakers": len(speakers),
        "n_turns": len(turns),
        "primary_language": cfg.primary_language,
        "n_languages": int(len({t.language for t in turns} | {cfg.primary_language})),
        "code_switch_count": code_switch_count,
        "has_music_interlude": cfg.music_interlude,
        "music_interval_s": list(music_interval) if music_interval else None,
        "source": "synth_coser",  # FT recipes refuse rows with this tag
        "split": "bench",
    }


def build_dataset(cfg: SynthCOSERConfig) -> Path:  # pragma: no cover (network)
    """Build N synth-COSER recordings. Returns the manifest parquet path."""
    import pandas as pd

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = cfg.out_dir / "_clip_pools"
    primary_clips = _load_clip_pool(
        cfg.primary_language,
        cfg.pool_size_per_language,
        cv_dataset=cfg.cv_dataset,
        cv_split=cfg.cv_split,
        cache_dir=cache_root / cfg.primary_language,
        accent_filter=cfg.primary_accent_filter,
    )
    primary_pool = group_by_speaker(primary_clips, cfg.min_clips_per_speaker)

    secondary_pools: dict[str, list[dict]] = {}
    for lang in cfg.secondary_languages:
        secondary_clips = _load_clip_pool(
            lang,
            cfg.pool_size_per_language,
            cv_dataset=cfg.cv_dataset,
            cv_split=cfg.cv_split,
            cache_dir=cache_root / lang,
        )
        # For secondary languages we don't need speaker grouping (code-switch
        # turns just need *any* clip in that language).
        secondary_pools[lang] = secondary_clips

    rows = []
    for i in range(cfg.n_recordings):
        record_id = f"synth_coser_{i:03d}"
        rows.append(
            build_recording(
                cfg,
                primary_pool,
                secondary_pools,
                record_id=record_id,
                rng=random.Random(cfg.seed + i),
            )
        )

    out = cfg.out_dir / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out
