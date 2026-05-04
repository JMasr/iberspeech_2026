# iberspeech_2026 — UVigo–Balidea ALBAYZIN Speech-COSER 2026

Submission code for the IberSPEECH 2026 ALBAYZIN Speech-COSER evaluation. We compete on **ASR** (WER, dual scoring: raw-with-punct + normalized lowercase no-punct) and **SD** (DER, no collar, includes overlap). STD is out of scope.

The full system design is in [`/home/jmramirez/.claude/plans/nested-nibbling-puppy.md`](/home/jmramirez/.claude/plans/nested-nibbling-puppy.md). Phase-by-phase progress is tracked in [`PROGRESS.md`](PROGRESS.md).

## Quickstart

```bash
make create_environment      # uv venv 3.10
source .venv/bin/activate
make requirements            # uv sync (heavy stack: torch, pyannote, transformers, …)
make lint test               # ruff + pytest
```

## Pipeline (one make target per stage)

```bash
make ingest             # build COSER manifest + stratified train/dev split
make stage0             # tier-0A enrichment (VAD/SNR/spectral/inaSpeechSegmenter)
make stage0-beats       # tier-0B (BEATs AudioSet tagger) on flagged chunks
make pseudo-label       # confidence-filtered set from the 10h dirty pool

make train-asr-whisper  # FT Whisper-large-v3 (full)
make train-asr-xlsr     # FT XLS-R-1B (CTC)
make train-sd-seg       # FT pyannote segmentation-3.0
make build-lm           # KenLM 5-gram on in-domain text

make infer-asr          # full ASR pipeline (chunked + ROVER + MBR + LM)
make infer-sd           # full SD pipeline (VBx + ASR-anchored boundary snap)
make score-asr          # meeteval, raw + normalized
make score-sd           # dscore-style, no-collar, overlap-included

make submit-asr SUBMIT=1   # build zip and (with SUBMIT=1) record the submission
make submit-sd  SUBMIT=1
make verify                # lint + tests + CLI self-check
```

The `coser` console script (registered via `[project.scripts]`) wraps the same CLI: `coser stage0 …`, `coser infer-asr …`, etc.

## Layout

```
src/
├── cli.py                  # typer entry, one subcommand per Make target
├── data/                   # ingest, audio, normalize, stage0, stage0_beats, pseudo_label
├── asr/                    # whisper_ft, w2v2_ft, infer_long, nbest, logit_bias, nonspeech_mask
├── sd/                     # seg_ft, embed, cluster_vbx, overlap, refine, pipeline
├── fusion/                 # rover, mbr, lm_kenlm, lm_neural
├── eval/                   # score_asr, score_sd, leaderboard
└── configs/                # YAML per experiment

data/dialect_lexicon.json   # curated bidirectional lexicon (province → elision rules)
experiments/<date>_<id>/    # one directory per leaderboard submission, immutable
references/                 # COSER_EvalPlan.pdf, initial_idea.md
PROGRESS.md                 # phase tracker
```

## Bench (public-data pipeline validation)

Before COSER data lands, validate the pipeline on public datasets. **Bench data never enters COSER FT** — it lives under `data/bench/` and the FT recipes refuse it (eval rule compliance).

```bash
make bench-synthetic            # E2E with stub recognizers — runs in <50ms, no GPU
make bench-ablation             # per-component lift table over MBR/ROVER/MASK/SNAP
make bench-cv-es                # Whisper-large-v3 zero-shot on 50 CV ES clips
make bench-cv-es \
    SECOND_VOTER=facebook/wav2vec2-xls-r-1b   # adds ROVER lift measurement

# VoxConverse SD baseline + boundary-snap delta:
VC_AUDIO=/path/to/voxconverse/audio \
VC_RTTM=/path/to/voxconverse/dev   \
VC_ASR_WORDS=experiments/bench/cv_es/words \
make bench-voxconverse
```

`make verify` runs lint + tests + the synthetic bench. The CV ES and VoxConverse benches need network and the heavy stack and are run on demand.

## Notes on the heavy stack

`torch`, `transformers`, `pyannote.audio`, etc. are pinned in `pyproject.toml` but only resolved when you actually run training/inference. The deterministic library (normalization, ROVER, MBR, RTTM cleanup, leaderboard packaging) imports nothing heavy and is fully tested via `make test`. Heavy imports happen inside the function bodies that use them so that `make lint test` works on a bare environment.

Optional extras (install only when needed; some have native build deps):

```bash
uv pip install -e '.[ina]'        # inaSpeechSegmenter
uv pip install -e '.[kenlm]'      # KenLM Python bindings
uv pip install -e '.[wespeaker]'  # WeSpeaker
uv pip install -e '.[nemo]'       # NeMo (only for the Canary-1B P4 stretch goal)
```

## License

MIT — see [LICENSE](LICENSE).
