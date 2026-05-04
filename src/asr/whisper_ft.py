"""Whisper-large-v3 (full) fine-tuning.

Round-1 trains on the 23h validated segments. Round-2 mixes in the retained
pseudo-labels. The same script handles both via the YAML config.

DDP is wired through ``accelerate``; reserve 1 of the 7 RTX 6000 Ada GPUs for
inference/eval (set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 for the trainer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WhisperFTConfig:
    base_model: str = "openai/whisper-large-v3"
    train_manifest: Path = Path("data/interim/manifest.parquet")
    pseudo_label_parquet: Path | None = None
    output_dir: Path = Path("models/whisper_ft_r1")
    language: str = "es"
    task: str = "transcribe"
    learning_rate: float = 1e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 4
    warmup_ratio: float = 0.05
    bf16: bool = True
    eval_steps: int = 500
    save_steps: int = 500
    extra: dict = field(default_factory=dict)


def train(cfg: WhisperFTConfig) -> Path:  # pragma: no cover (heavy + GPU only)
    """Fine-tune Whisper. Returns the output dir.

    Inputs:
      - ``cfg.train_manifest`` — output of ``src.data.ingest.build_manifest``
        (columns: split, audio_path, transcript, …).
      - ``cfg.pseudo_label_parquet`` — optional output of
        ``src.data.pseudo_label.build_pseudo_labels``.
    """
    from datasets import Audio, Dataset
    import pandas as pd
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.train_manifest)
    from src.bench import assert_no_bench_rows

    assert_no_bench_rows(df)
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    if cfg.pseudo_label_parquet is not None and Path(cfg.pseudo_label_parquet).exists():
        plf = pd.read_parquet(cfg.pseudo_label_parquet)
        assert_no_bench_rows(plf)
        plf["split"] = "train"
        plf = plf.rename(columns={})
        plf["transcript"] = plf["transcript"]
        train_df = pd.concat([train_df, plf[["audio_path", "transcript"]]], ignore_index=True)

    train_ds = Dataset.from_pandas(train_df[["audio_path", "transcript"]]).cast_column(
        "audio_path", Audio(sampling_rate=16_000)
    )
    dev_ds = Dataset.from_pandas(dev_df[["audio_path", "transcript"]]).cast_column(
        "audio_path", Audio(sampling_rate=16_000)
    )

    processor = WhisperProcessor.from_pretrained(
        cfg.base_model, language=cfg.language, task=cfg.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg.language, task=cfg.task
    )
    model.config.suppress_tokens = []

    def _prepare(batch):
        audio = batch["audio_path"]
        feat = processor.feature_extractor(audio["array"], sampling_rate=16_000).input_features[0]
        labels = processor.tokenizer(batch["transcript"]).input_ids
        return {"input_features": feat, "labels": labels}

    train_ds = train_ds.map(_prepare, remove_columns=train_ds.column_names, num_proc=4)
    dev_ds = dev_ds.map(_prepare, remove_columns=dev_ds.column_names, num_proc=4)

    args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        bf16=cfg.bf16,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["wandb"],
        logging_steps=25,
        save_total_limit=2,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor.feature_extractor,
        data_collator=_DataCollatorWhisper(processor),
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    processor.save_pretrained(str(cfg.output_dir))
    return cfg.output_dir


class _DataCollatorWhisper:  # pragma: no cover (heavy)
    """Pads input_features and labels separately."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
