"""pyannote segmentation-3.0 fine-tuning.

Light FT (5 epochs, LR 1e-5). Even a small adaptation materially improves
boundary timing on COSER's interview style. We do NOT fine-tune the embedding
model — see the plan's "What we are intentionally NOT doing" section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SegFTConfig:
    base_model: str = "pyannote/segmentation-3.0"
    rttm_dir: Path = Path("data/raw/rttm")
    audio_dir: Path = Path("data/raw/longform")
    output_dir: Path = Path("models/pyannote_seg_ft")
    learning_rate: float = 1e-5
    num_epochs: int = 5
    batch_size: int = 32
    duration_s: float = 5.0
    extra: dict = field(default_factory=dict)


def train(cfg: SegFTConfig) -> Path:  # pragma: no cover (heavy + GPU only)
    """Fine-tune pyannote segmentation-3.0 on the COSER RTTMs.

    Heavy imports are local. The training loop uses pyannote.audio's
    ``SpeakerSegmentation`` task class.
    """
    from pyannote.audio import Model
    from pyannote.audio.tasks import SpeakerDiarization
    from pyannote.database import FileFinder, get_protocol
    from pytorch_lightning import Trainer

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Expect the user to have registered a ``database.yml`` describing COSER as
    # ``COSER.SpeakerDiarization.Default``.
    protocol = get_protocol(
        "COSER.SpeakerDiarization.Default",
        preprocessors={"audio": FileFinder()},
    )
    task = SpeakerDiarization(
        protocol,
        duration=cfg.duration_s,
        max_num_speakers=4,
        batch_size=cfg.batch_size,
    )
    model = Model.from_pretrained(cfg.base_model)
    model.task = task

    trainer = Trainer(
        max_epochs=cfg.num_epochs,
        accelerator="gpu",
        devices="auto",
        default_root_dir=str(cfg.output_dir),
        precision="bf16-mixed",
    )
    trainer.fit(model)
    out = cfg.output_dir / "model.ckpt"
    trainer.save_checkpoint(str(out))
    return out
