"""
src/training/train.py

QLoRA fine-tuning of Qwen2.5-7B-Instruct on your WhatsApp data.

Usage:
    python src/training/train.py
    python src/training/train.py --epochs 5 --lr 1e-4
    python src/training/train.py --resume models/checkpoints/checkpoint-1000
"""

import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import sys
import trl
import transformers

print("Python:", sys.executable)
print("TRL:", trl.__version__, trl.__file__)
print("Transformers:", transformers.__version__, transformers.__file__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import *


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',    type=int,   default=None)
    p.add_argument('--lr',        type=float, default=None)
    p.add_argument('--resume',    type=str,   default=None)
    p.add_argument('--base-model',type=str,   default=None)
    return p.parse_args()


def load_model_and_tokenizer(model_name: str):
    print(f"\n[INFO] Loading base model: {model_name}")
    print(f"[INFO] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=getattr(torch, COMPUTE_DTYPE),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Model loaded. VRAM used: "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


def setup_lora(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()

    # Apply CLI overrides
    epochs    = args.epochs    or NUM_EPOCHS
    lr        = args.lr        or LEARNING_RATE
    base_model = args.base_model or BASE_MODEL

    # Validate data files
    for path, name in [(TRAIN_FILE, "train.jsonl"), (VAL_FILE, "val.jsonl")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} not found at {path}\n"
                "Run: python scripts/run_pipeline.py first"
            )

    # Load dataset
    print("\n[INFO] Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
        }
    )
    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Val:   {len(dataset['validation']):,} samples")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.resume or base_model
    )
    model = setup_lora(model)

    # Training config
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    import inspect

    print(SFTConfig)
    print(inspect.signature(SFTConfig))

    training_args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        eval_strategy=EVAL_STRATEGY,
        load_best_model_at_end=LOAD_BEST_MODEL,
        metric_for_best_model="eval_loss",
        greater_is_better=False, 
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",    # Change to "wandb" if you want experiment tracking
        run_name="whatsapp-style-model",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["validation"],
    #     args=training_args,
    # )

    # Train
    print("\n[INFO] Starting training...")
    print(f"  Epochs:            {epochs}")
    print(f"  Learning rate:     {lr}")
    print(f"  Effective batch:   {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Max seq length:    {MAX_SEQ_LENGTH}")
    print(f"  LoRA rank:         {LORA_R}\n")

    trainer.train()

    # Save final checkpoint
    print("\n[INFO] Saving final checkpoint...")
    trainer.save_model(str(CHECKPOINT_DIR / "checkpoint-final"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "checkpoint-final"))

    print("\n✅ Training complete!")
    print(f"   Checkpoint saved to: {CHECKPOINT_DIR / 'checkpoint-final'}")
    print(f"\nNext step: python src/training/merge.py")


if __name__ == "__main__":
    main()
