"""
src/inference/chat.py

Interactive CLI for testing your fine-tuned WhatsApp style model.

Usage:
    python src/inference/chat.py
    python src/inference/chat.py --model models/final
    python src/inference/chat.py --relationship best_friend
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    FINAL_DIR, YOUR_NAME,
    GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P, GEN_TOP_K, GEN_REPEAT_PENALTY
)
from src.features.context import SYSTEM_PROMPT


VALID_RELATIONSHIPS = [
    "best_friend", "close_friend", "girlfriend",
    "family", "colleague", "acquaintance", "stranger",
    "group_close", "group_formal"
]

BANNER = """
╔══════════════════════════════════════════╗
║     WhatsApp Style Model  —  CLI         ║
╚══════════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str, default=str(FINAL_DIR))
    p.add_argument('--relationship', type=str, default=None)
    p.add_argument('--temp',         type=float, default=GEN_TEMPERATURE)
    p.add_argument('--max-tokens',   type=int,   default=GEN_MAX_NEW_TOKENS)
    p.add_argument('--quantize',     action='store_true',
                   help='Load in 4-bit (saves VRAM, slightly slower)')
    return p.parse_args()


def load_model(model_path: str, quantize: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[INFO] Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if quantize:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[INFO] Model ready. VRAM: {vram:.1f} GB")
    return model, tokenizer


def build_prompt(
    context_lines: list[str],
    relationship: str,
    your_name: str = YOUR_NAME,
) -> str:
    now = datetime.now()
    hour = now.hour
    day_name = now.strftime('%A')
    context_text = "\n".join(context_lines) if context_lines else "(start of conversation)"

    system = SYSTEM_PROMPT.format(your_name=your_name)
    metadata = (
        f"[Relationship: {relationship} | Hour: {hour} | Day: {day_name}]"
    )
    user_content = f"{context_text}\n{metadata}"

    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REPEAT_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply.strip()


def choose_relationship() -> str:
    print("\nRelationship type:")
    for i, rel in enumerate(VALID_RELATIONSHIPS, 1):
        print(f"  {i:2d}. {rel}")
    while True:
        choice = input("\nEnter number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(VALID_RELATIONSHIPS):
            return VALID_RELATIONSHIPS[int(choice) - 1]
        if choice in VALID_RELATIONSHIPS:
            return choice
        print("Invalid choice, try again.")


def main():
    args = parse_args()
    model_path = args.model

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("Run: python src/training/merge.py first")
        sys.exit(1)

    print(BANNER)
    model, tokenizer = load_model(model_path, args.quantize)

    relationship = args.relationship
    if not relationship:
        relationship = choose_relationship()

    recipient = input("\nRecipient name (for display): ").strip() or "Them"

    print(f"\n{'─'*44}")
    print(f"  Chat with: {recipient}  [{relationship}]")
    print(f"  Type messages as {recipient}. Empty line = generate reply.")
    print(f"  Type 'quit' to exit, 'clear' to reset context.")
    print(f"{'─'*44}\n")

    context_lines = []

    while True:
        line = input(f"[{recipient}]: ").strip()

        if line.lower() == 'quit':
            break

        if line.lower() == 'clear':
            context_lines = []
            print("  [Context cleared]\n")
            continue

        if line:
            context_lines.append(f"{recipient}: {line}")
            # Keep context window bounded
            if len(context_lines) > 10:
                context_lines = context_lines[-10:]
            continue

        # Empty line → generate
        if not context_lines:
            print("  [No context yet — type some messages first]\n")
            continue

        print(f"[{YOUR_NAME}]: ", end='', flush=True)
        prompt = build_prompt(context_lines, relationship)
        reply = generate(model, tokenizer, prompt, args)
        print(reply)

        # Add to context
        context_lines.append(f"{YOUR_NAME}: {reply}")
        print()


if __name__ == "__main__":
    main()
