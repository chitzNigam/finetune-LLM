"""
src/training/merge.py

Merge LoRA adapter into the base model.

Usage:
    python src/training/merge.py
    python src/training/merge.py --checkpoint models/checkpoints/checkpoint-final
"""

import argparse
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import BASE_MODEL, CHECKPOINT_DIR, FINAL_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "checkpoint-final"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(FINAL_DIR),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n[INFO] Loading base model: {BASE_MODEL}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)

    # Load entirely on CPU to avoid accelerate offloading issues.
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     dtype=torch.float16,
    #     device_map={"": "cpu"},
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    print(f"[INFO] Loading LoRA adapter from: {checkpoint_path}")

    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
    )

    print("[INFO] Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    print(f"[INFO] Saving merged model to: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(output_path)

    readme = f"""# WhatsApp Style Model

Merged LoRA model.

Base model:
{BASE_MODEL}

Adapter:
{checkpoint_path}
"""

    (output_path / "README.md").write_text(readme)

    print("\n✅ Merge complete!")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()