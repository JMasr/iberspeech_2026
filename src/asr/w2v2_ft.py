"""Wav2Vec2-XLS-R-1B (CTC) fine-tuning.

Same data plumbing as Whisper FT, but the head is character-level CTC.
We don't push CTC outputs to the leaderboard directly — XLS-R's role is to
provide a CTC lattice / 1-best for ROVER and confidence info for the
pseudo-label gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class XLSRFTConfig:
    base_model: str = "facebook/wav2vec2-xls-r-1b"
    train_manifest: Path = Path("data/interim/manifest.parquet")
    pseudo_label_parquet: Path | None = None
    output_dir: Path = Path("models/xlsr_ft_r1")
    learning_rate: float = 5e-5
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 8
    warmup_ratio: float = 0.10
    bf16: bool = True
    extra: dict = field(default_factory=dict)


def train(cfg: XLSRFTConfig) -> Path:  # pragma: no cover (heavy + GPU only)
    from datasets import Audio, Dataset
    import pandas as pd
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
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
        train_df = pd.concat([train_df, plf[["audio_path", "transcript"]]], ignore_index=True)

    train_ds = Dataset.from_pandas(train_df[["audio_path", "transcript"]]).cast_column(
        "audio_path", Audio(sampling_rate=16_000)
    )
    dev_ds = Dataset.from_pandas(dev_df[["audio_path", "transcript"]]).cast_column(
        "audio_path", Audio(sampling_rate=16_000)
    )

    # Build a char vocab from the training transcripts. Standard XLS-R recipe.
    chars = sorted({c for t in train_df["transcript"] for c in t.lower() if c.strip()})
    vocab = {c: i + 3 for i, c in enumerate(chars)}
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    vocab["|"] = 2  # word boundary
    import json

    vocab_path = cfg.output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True
    )
    processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)
    processor.save_pretrained(str(cfg.output_dir))
    model = Wav2Vec2ForCTC.from_pretrained(
        cfg.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(vocab),
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    def _prepare(batch):
        audio = batch["audio_path"]
        feats = processor(audio["array"], sampling_rate=16_000).input_values[0]
        text = batch["transcript"].lower().replace(" ", "|")
        labels = processor.tokenizer(text).input_ids
        return {"input_values": feats, "labels": labels}

    train_ds = train_ds.map(_prepare, remove_columns=train_ds.column_names, num_proc=4)
    dev_ds = dev_ds.map(_prepare, remove_columns=dev_ds.column_names, num_proc=4)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        bf16=cfg.bf16,
        report_to=["wandb"],
        logging_steps=25,
        save_total_limit=2,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor.feature_extractor,
        data_collator=_DataCollatorXLSR(processor),
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    return cfg.output_dir


class _DataCollatorXLSR:  # pragma: no cover (heavy)
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
