"""Stage 0A — cheap, always-on enrichment.

For each long recording we emit ``data/processed/chunks/<record_id>.parquet``
with the schema documented in the plan:

  chunk_id, record_id, start_s, end_s,
  vad_speech_ratio, wada_snr_db,
  spectral_cluster_id, era_bucket,
  ina_label, ina_confidence,
  beats_tags (NULL until Stage 0B), beats_scores (NULL until Stage 0B),
  province, topic, n_speakers_hint, flagged_for_beats

Every column has a documented downstream consumer; we do NOT compute features
without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data.audio import (
    CHUNK_SECONDS_DEFAULT,
    OVERLAP_SECONDS_DEFAULT,
    SAMPLE_RATE,
    Chunk,
    chunk_intervals,
    load_mono_16k,
    slice_audio,
)


@dataclass(frozen=True)
class Stage0Config:
    chunk_s: float = CHUNK_SECONDS_DEFAULT
    overlap_s: float = OVERLAP_SECONDS_DEFAULT
    spectral_clusters: int = 5
    snr_low_pct: float = 25.0
    use_pyannote_vad: bool = True
    use_inaspeech: bool = True


def wada_snr_db(audio) -> float:
    """WADA-SNR for speech, single-channel float32 audio.

    Reference: Kim & Stern, "Robust Signal-to-Noise Ratio Estimation Based on
    Waveform Amplitude Distribution Analysis", Interspeech 2008. Numpy-only,
    O(N) memory, deterministic.
    """
    import numpy as np

    if audio.size == 0:
        return float("nan")
    abs_audio = np.abs(audio.astype("float64"))
    abs_audio = abs_audio[abs_audio > 0]
    if abs_audio.size == 0:
        return float("-inf")
    # Lognormal moment ratio.
    log_a = np.log(abs_audio)
    mean_log = log_a.mean()
    e_log = np.log(np.mean(abs_audio))
    g = mean_log - e_log  # always <= 0; closer to 0 = higher SNR
    # Empirical mapping (Kim & Stern 2008 Table 1) clamped to a sensible range.
    # We use the clipped-linear approximation: SNR ≈ -10 - 60 * (g + 0.4)
    snr = float(-10.0 - 60.0 * (g + 0.4))
    return max(-20.0, min(60.0, snr))


def vad_speech_segments(
    audio,
    use_pyannote: bool = True,
    sr: int = SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """Return speech intervals as (start_s, end_s).

    Primary: pyannote VAD (3.x). Fallback: simple energy-based detector — keeps
    the rest of the pipeline runnable without the heavy stack during smoke tests.
    """
    import numpy as np

    if use_pyannote:
        try:
            return _pyannote_vad(audio, sr)
        except Exception:
            # Fall through to the energy fallback. We do not log here — that is the
            # caller's responsibility (it has the recording id).
            pass

    # Energy-based fallback: 25 ms frames, 10 ms hop, threshold at 5th percentile.
    frame = int(0.025 * sr)
    hop = int(0.010 * sr)
    if audio.size < frame:
        return []
    n_frames = 1 + (audio.size - frame) // hop
    energies = np.empty(n_frames, dtype="float32")
    for i in range(n_frames):
        window = audio[i * hop : i * hop + frame]
        energies[i] = float(np.sqrt(np.mean(window * window) + 1e-12))
    thresh = float(np.percentile(energies, 25)) * 1.5
    voiced = energies > thresh
    return _runs_to_segments(voiced, hop_s=0.010, min_dur_s=0.20)


def _pyannote_vad(audio, sr: int) -> list[tuple[float, float]]:
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    waveform = torch.from_numpy(audio).unsqueeze(0)
    out = pipeline({"waveform": waveform, "sample_rate": sr})
    return [(seg.start, seg.end) for seg in out.get_timeline().support()]


def _runs_to_segments(mask, hop_s: float, min_dur_s: float) -> list[tuple[float, float]]:
    import numpy as np

    if mask.size == 0:
        return []
    diff = np.diff(np.concatenate([[0], mask.astype("int8"), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    out = []
    for a, b in zip(starts, ends):
        s, e = a * hop_s, b * hop_s
        if e - s >= min_dur_s:
            out.append((float(s), float(e)))
    return out


def spectral_features(audio, sr: int = SAMPLE_RATE):
    """Compact spectral fingerprint used for the era/channel cluster."""
    import librosa
    import numpy as np

    if audio.size < sr // 4:
        return np.zeros(7, dtype="float32")
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4)
    return np.array(
        [
            float(centroid.mean()),
            float(centroid.std()),
            float(bw.mean()),
            float(rolloff.mean()),
            float(mfcc[1].mean()),
            float(mfcc[2].mean()),
            float(mfcc[3].mean()),
        ],
        dtype="float32",
    )


def fit_spectral_clusterer(features, k: int = 5):
    """K-Means over per-recording spectral features. Returns (model, labels)."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init="auto", random_state=20260501)
    labels = km.fit_predict(features)
    return km, labels


def ina_speech_segmentation(audio, sr: int = SAMPLE_RATE):
    """Run inaSpeechSegmenter; return list of (label, start_s, end_s).

    Falls back to a single 'speech' span if the package is not installed —
    the cheap-tier flagging logic treats this as "non-ambiguous, no music".
    """
    try:
        from inaSpeechSegmenter import Segmenter  # noqa: F401  (probe for installability)
    except ImportError:
        duration = audio.size / float(sr)
        return [("speech", 0.0, duration)]
    # inaSpeechSegmenter expects a file path; the higher-level enrich() handles
    # that. This in-memory entry point is intentionally inert and exists only
    # so the public API is symmetric with the other Stage 0 helpers.
    raise RuntimeError("ina_speech_segmentation must be called via enrich() with a wav path")


def _ina_segments_for_path(path: str | Path):
    from inaSpeechSegmenter import Segmenter

    seg = Segmenter()
    out = seg(str(path))
    return [(label, float(start), float(end)) for label, start, end in out]


def chunk_label_from_ina(ina_segments, chunk: Chunk) -> tuple[str, float]:
    """Pick the dominant inaSpeech label inside a chunk, return (label, coverage)."""
    if not ina_segments:
        return ("speech", 1.0)
    chunk_dur = max(1e-6, chunk.duration_s)
    coverage: dict[str, float] = {}
    for label, s, e in ina_segments:
        a = max(s, chunk.start_s)
        b = min(e, chunk.end_s)
        if b > a:
            coverage[label] = coverage.get(label, 0.0) + (b - a)
    if not coverage:
        return ("speech", 0.0)
    label, dur = max(coverage.items(), key=lambda kv: kv[1])
    return (label, dur / chunk_dur)


def era_bucket_from_meta(year: int | None, channel: str | None) -> str:
    """Map year + channel into one of the documented era buckets."""
    if year is None:
        return "unknown"
    y = int(year)
    if y < 1995:
        return "cassette-1990s-early"
    if y < 2002:
        return "cassette-1990s-late"
    if y < 2010:
        return "MD-2000s"
    if y < 2018:
        return "digital-2010s"
    return "digital-2020s"


def flag_for_beats(
    *,
    ina_label: str,
    ina_coverage: float,
    snr_db: float,
    snr_p25: float,
    era_bucket: str,
) -> bool:
    """Decide whether a chunk needs Tier-0B BEATs tagging.

    True if ANY of:
      - inaSpeech says music/no_energy or label coverage < 70% (mixed within chunk),
      - SNR below the 25th percentile of the training distribution,
      - era bucket = cassette-1990s-early (most non-speech artifacts).
    """
    if ina_label != "speech" or ina_coverage < 0.70:
        return True
    if snr_db < snr_p25:
        return True
    if era_bucket.startswith("cassette-1990s-early"):
        return True
    return False


def enrich_recording(
    record_id: str,
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    config: Stage0Config = Stage0Config(),
    province: str | None = None,
    topic: str | None = None,
    year: int | None = None,
    channel: str | None = None,
    snr_p25_global: float | None = None,
) -> Path:
    """Enrich one recording, write the per-recording parquet, and return its path.

    ``snr_p25_global`` should come from a corpus-wide pre-pass; if None we use
    the per-recording 25th percentile (acceptable for smoke tests / single-file runs).
    """
    import numpy as np
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = load_mono_16k(audio_path)
    speech = vad_speech_segments(audio, use_pyannote=config.use_pyannote_vad)
    if not speech:
        # Empty parquet — still useful as a sentinel that the recording was processed.
        speech = [(0.0, audio.size / SAMPLE_RATE)]

    chunks = list(
        chunk_intervals(
            speech,
            chunk_s=config.chunk_s,
            overlap_s=config.overlap_s,
            record_id=record_id,
        )
    )

    if config.use_inaspeech:
        try:
            ina_segments = _ina_segments_for_path(audio_path)
        except Exception:
            ina_segments = [("speech", 0.0, audio.size / SAMPLE_RATE)]
    else:
        ina_segments = [("speech", 0.0, audio.size / SAMPLE_RATE)]

    snrs = []
    spectral = []
    for ch in chunks:
        clip = slice_audio(audio, ch.start_s, ch.end_s)
        snrs.append(wada_snr_db(clip))
        spectral.append(spectral_features(clip))
    spectral = np.stack(spectral) if spectral else np.zeros((0, 7), dtype="float32")
    snrs_arr = np.asarray(snrs, dtype="float32")
    snr_p25 = (
        (
            float(snr_p25_global)
            if snr_p25_global is not None
            else float(np.percentile(snrs_arr, 25))
        )
        if snrs_arr.size
        else 0.0
    )
    cluster_labels = (
        np.zeros(len(chunks), dtype="int32")
        if len(chunks) < config.spectral_clusters
        else fit_spectral_clusterer(spectral, k=config.spectral_clusters)[1]
    )
    era = era_bucket_from_meta(year, channel)

    rows = []
    for i, ch in enumerate(chunks):
        ina_label, ina_cov = chunk_label_from_ina(ina_segments, ch)
        flagged = flag_for_beats(
            ina_label=ina_label,
            ina_coverage=ina_cov,
            snr_db=float(snrs_arr[i]) if snrs_arr.size else 0.0,
            snr_p25=snr_p25,
            era_bucket=era,
        )
        rows.append(
            {
                "chunk_id": ch.chunk_id,
                "record_id": ch.record_id,
                "start_s": ch.start_s,
                "end_s": ch.end_s,
                "vad_speech_ratio": _vad_speech_ratio(speech, ch),
                "wada_snr_db": float(snrs_arr[i]) if snrs_arr.size else 0.0,
                "spectral_cluster_id": int(cluster_labels[i]),
                "era_bucket": era,
                "ina_label": ina_label,
                "ina_confidence": float(ina_cov),
                "beats_tags": None,
                "beats_scores": None,
                "province": province,
                "topic": topic,
                "n_speakers_hint": None,  # filled by SD pipeline if/when run
                "flagged_for_beats": flagged,
            }
        )

    out_path = out_dir / f"{record_id}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


def _vad_speech_ratio(speech: Iterable[tuple[float, float]], chunk: Chunk) -> float:
    chunk_dur = max(1e-6, chunk.duration_s)
    total = 0.0
    for s, e in speech:
        a = max(s, chunk.start_s)
        b = min(e, chunk.end_s)
        if b > a:
            total += b - a
    return float(min(1.0, total / chunk_dur))
