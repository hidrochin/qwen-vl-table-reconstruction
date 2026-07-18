"""LoRA fine-tuning for Qwen3-VL via Unsloth.

Runs on the GPU box only; every heavy import is deferred so the module still
imports on a Mac.

Two settings here exist specifically to survive a free-tier environment:

* Checkpoints are written every ``save_steps`` and training resumes from the last
  one. Colab disconnects, and a run that cannot resume is a run that has to start
  over -- which on a four-day clock can cost the whole experiment.
* ``max_pixels`` caps visual tokens. A full-page table can exceed 4,000 image
  tokens on its own; combined with a long HTML target that is enough to OOM a
  16 GB card partway through an epoch.

Default target is structure-only HTML. It cuts sequence length 5-10x, matches
TEDS-Struct, and is what makes an 8B-class model trainable on a small GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.html_utils import to_structure_only
from src.data.loader import TableRecord
from src.model.inference import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, MODEL_4B
from src.model.prompts import resolve_instruction


@dataclass
class TrainConfig:
    model_id: str = MODEL_4B
    output_dir: str = "outputs/lora"
    mode: str = "structure"

    # Must match the prompt used at inference. If Day 2's prompt iteration lands
    # on a better instruction, set it here too -- training on one prompt and
    # generating with another wastes most of what the adapter learned.
    instruction: str | None = None

    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = False  # structure is a language-side task

    batch_size: int = 1
    grad_accum: int = 4
    learning_rate: float = 2e-4
    epochs: int = 2
    warmup_steps: int = 10
    max_seq_length: int = 4096

    max_pixels: int = DEFAULT_MAX_PIXELS
    min_pixels: int = DEFAULT_MIN_PIXELS

    save_steps: int = 50
    logging_steps: int = 5
    seed: int = 3407


def build_dataset(
    records: list[TableRecord], mode: str = "structure", instruction: str | None = None
) -> list[dict]:
    """Convert records into Unsloth vision-chat format.

    Targets are normalized through the same code path the metric uses, so the
    model is trained on exactly the representation it is scored against.
    """
    from PIL import Image

    instruction = resolve_instruction(mode, instruction)
    dataset = []
    for rec in records:
        target = to_structure_only(rec.html) if mode == "structure" else rec.html
        if not target:
            continue
        dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.open(rec.image_path).convert("RGB")},
                            {"type": "text", "text": instruction},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ]
            }
        )
    return dataset


def filter_by_length(
    records: list[TableRecord], processor, cfg: TrainConfig, margin: int = 512
) -> list[TableRecord]:
    """Drop samples whose target would overflow ``max_seq_length``.

    One pathological table is enough to OOM a run partway through. Losing a few
    outliers costs less than losing the run.
    """
    from src.model.inference import visual_token_estimate
    from PIL import Image

    kept, dropped = [], 0
    for rec in records:
        target = to_structure_only(rec.html) if cfg.mode == "structure" else rec.html
        text_tokens = len(processor.tokenizer(target).input_ids)
        with Image.open(rec.image_path) as im:
            image_tokens = visual_token_estimate(*im.size, cfg.max_pixels)
        if text_tokens + image_tokens + margin <= cfg.max_seq_length:
            kept.append(rec)
        else:
            dropped += 1

    if dropped:
        print(f"  length filter dropped {dropped}/{len(records)} over {cfg.max_seq_length} tokens")
    return kept


def train(records: list[TableRecord], cfg: TrainConfig | None = None, resume: bool = True):
    """Fine-tune a LoRA adapter. Returns (model, processor, trainer)."""
    cfg = cfg or TrainConfig()

    from unsloth import FastVisionModel  # must precede transformers imports
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    model, processor = FastVisionModel.from_pretrained(
        cfg.model_id,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=cfg.max_seq_length,
    )
    for attr, value in (("max_pixels", cfg.max_pixels), ("min_pixels", cfg.min_pixels)):
        if hasattr(processor, "image_processor"):
            setattr(processor.image_processor, attr, value)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        random_state=cfg.seed,
    )
    FastVisionModel.for_training(model)

    records = filter_by_length(records, processor, cfg)
    dataset = build_dataset(records, cfg.mode, cfg.instruction)
    print(f"  training on {len(dataset)} samples, mode={cfg.mode}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_strategy="steps",
            save_total_limit=2,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            seed=cfg.seed,
            max_length=cfg.max_seq_length,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="none",
        ),
    )

    checkpoint = _latest_checkpoint(out_dir) if resume else None
    if checkpoint:
        print(f"  resuming from {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"  adapter saved -> {adapter_dir}")

    return model, processor, trainer


def _latest_checkpoint(out_dir: Path) -> str | None:
    checkpoints = sorted(
        out_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(checkpoints[-1]) if checkpoints else None
