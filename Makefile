#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME    = iberspeech_2026
PYTHON_VERSION  = 3.10
PYTHON          = python
RAW             = data/raw
INTERIM         = data/interim
PROCESSED       = data/processed
MODELS          = models
EXPERIMENTS     = experiments
GROUP_ID       ?= UVigoBalidea

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run unit tests (deterministic, no GPU/data required)
.PHONY: test
test:
	$(PYTHON) -m pytest tests

#################################################################################
# DATA                                                                          #
#################################################################################

## Build COSER manifest + stratified train/dev split (province × age × topic)
.PHONY: ingest
ingest:
	$(PYTHON) -m src.cli ingest --raw $(RAW) --out $(INTERIM)/manifest.parquet

## Stage 0A enrichment (VAD + WADA-SNR + spectral cluster + inaSpeechSegmenter)
.PHONY: stage0
stage0:
	$(PYTHON) -m src.cli stage0 --manifest $(INTERIM)/manifest.parquet --out $(PROCESSED)/chunks

## Stage 0B BEATs tagger (only on flagged chunks)
.PHONY: stage0-beats
stage0-beats:
	$(PYTHON) -m src.cli stage0-beats --chunks $(PROCESSED)/chunks

## Build pseudo-label set from the 10h dirty long-form audio
.PHONY: pseudo-label
pseudo-label:
	$(PYTHON) -m src.cli pseudo-label --manifest $(INTERIM)/manifest.parquet --out $(INTERIM)/pseudo_labels.parquet

#################################################################################
# TRAIN                                                                         #
#################################################################################

## Fine-tune Whisper-large-v3 on COSER (round 1 or 2 depending on config)
.PHONY: train-asr-whisper
train-asr-whisper:
	$(PYTHON) -m src.cli train-asr-whisper --config src/configs/whisper_ft.yaml

## Fine-tune Wav2Vec2-XLS-R-1B (CTC) on COSER
.PHONY: train-asr-xlsr
train-asr-xlsr:
	$(PYTHON) -m src.cli train-asr-xlsr --config src/configs/w2v2_ft.yaml

## Fine-tune pyannote segmentation-3.0
.PHONY: train-sd-seg
train-sd-seg:
	$(PYTHON) -m src.cli train-sd-seg --config src/configs/sd_seg_ft.yaml

## Build KenLM 5-gram on cleaned in-domain text
.PHONY: build-lm
build-lm:
	$(PYTHON) -m src.cli build-lm --text $(INTERIM)/lm_text.txt --out $(MODELS)/coser_5gram.arpa

#################################################################################
# INFER + SCORE                                                                 #
#################################################################################

## Run full ASR pipeline on a manifest split (default: dev)
.PHONY: infer-asr
infer-asr:
	$(PYTHON) -m src.cli infer-asr --config src/configs/infer_asr.yaml --split dev

## Run full SD pipeline on a manifest split (default: dev)
.PHONY: infer-sd
infer-sd:
	$(PYTHON) -m src.cli infer-sd --config src/configs/infer_sd.yaml --split dev

## Score ASR (raw + normalized WER via meeteval)
.PHONY: score-asr
score-asr:
	$(PYTHON) -m src.cli score-asr --hyp $(EXPERIMENTS)/latest/asr --ref $(INTERIM)/dev_refs

## Score SD (no-collar overlap-included DER)
.PHONY: score-sd
score-sd:
	$(PYTHON) -m src.cli score-sd --hyp $(EXPERIMENTS)/latest/sd --ref $(INTERIM)/dev_rttm

#################################################################################
# SUBMIT                                                                        #
#################################################################################

## Build ASR submission zip (dry-run only by default; use SUBMIT=1 to confirm)
.PHONY: submit-asr
submit-asr:
	$(PYTHON) -m src.cli submit --track asr --hyp $(EXPERIMENTS)/latest/asr --group $(GROUP_ID) $(if $(SUBMIT),--confirm,)

## Build SD submission zip
.PHONY: submit-sd
submit-sd:
	$(PYTHON) -m src.cli submit --track sd --hyp $(EXPERIMENTS)/latest/sd --group $(GROUP_ID) $(if $(SUBMIT),--confirm,)

#################################################################################
# BENCH (public-data pipeline validation; NEVER mixed into FT)                  #
#################################################################################

## Synthetic E2E (no GPU/internet) — exercises full ASR+SD wiring with stub recognizers
.PHONY: bench-synthetic
bench-synthetic:
	$(PYTHON) -m src.cli bench-synthetic

## Ablation table over MBR / ROVER / non-speech mask / boundary snap
.PHONY: bench-ablation
bench-ablation:
	$(PYTHON) -m src.cli bench-ablation

## Pre-Whisper Stage 0 ablation (VAD / music routing / non-speech mask) on real audio
.PHONY: bench-stage0
bench-stage0:
	$(PYTHON) -m src.cli bench-stage0

## Multilingual + accent stratified CV pull → zero-shot Whisper → per-bucket WER
.PHONY: bench-cv-multilingual
bench-cv-multilingual:
	$(PYTHON) -m src.cli bench-cv-multilingual --n-per-language 20

## Build a custom HF-sourced "synth-COSER" dataset for hypothesis validation
##   Override defaults with: make synth-coser-build DURATION=10 SECONDARY=ca,gl
.PHONY: synth-coser-build
synth-coser-build:
	$(PYTHON) -m src.cli synth-coser-build \
	    --duration-min $(if $(DURATION),$(DURATION),5) \
	    --n-recordings $(if $(N_RECORDINGS),$(N_RECORDINGS),3) \
	    --primary $(if $(PRIMARY),$(PRIMARY),es) \
	    --secondary $(if $(SECONDARY),$(SECONDARY),) \
	    --code-switch-prob $(if $(CODE_SWITCH),$(CODE_SWITCH),0.0) \
	    $(if $(MUSIC),--music,)

## Run zero-shot Whisper over a synth-COSER manifest and grade hypotheses
.PHONY: synth-coser-run
synth-coser-run:
	$(PYTHON) -m src.cli synth-coser-run \
	    --whisper-model $(if $(MODEL),$(MODEL),small)

## Common Voice ES — zero-shot Whisper baseline, optional 2nd voter for ROVER lift
.PHONY: bench-cv-es
bench-cv-es:
	$(PYTHON) -m src.cli bench-cv-es --n-samples 50

## VoxConverse — pyannote zero-shot SD + boundary-snap delta
.PHONY: bench-voxconverse
bench-voxconverse:
	@if [ -z "$$VC_AUDIO" ] || [ -z "$$VC_RTTM" ]; then \
	    echo "Set VC_AUDIO=/path/to/voxconverse/audio and VC_RTTM=/path/to/voxconverse/dev"; exit 2; \
	fi
	$(PYTHON) -m src.cli bench-voxconverse $$VC_AUDIO $$VC_RTTM --asr-words $$VC_ASR_WORDS --n-recordings 5

#################################################################################
# COMPOSITE / VERIFY                                                            #
#################################################################################

## End-to-end verification used at end of P0 (does not need real data)
.PHONY: verify
verify: lint test bench-synthetic bench-ablation bench-stage0
	$(PYTHON) -m src.cli self-check
	@echo ">>> Verification passed."

#################################################################################
# Self-Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
