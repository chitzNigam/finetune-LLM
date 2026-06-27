"""
src/inference/evaluate.py

Evaluates your trained model on the held-out test set.
Generates replies and saves them alongside ground truth for comparison.

Usage:
    python src/inference/evaluate.py
    python src/inference/evaluate.py --n 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import DATA_DIR, FINAL_DIR, YOUR_NAME


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, default=str(FINAL_DIR))
    p.add_argument('--n',       type=int, default=50,
                   help='Number of test samples to evaluate')
    p.add_argument('--output',  type=str, default='results/eval_output.json')
    p.add_argument('--include-raw-text', action='store_true',
                   help='Store full context and true reply text in the output JSON')
    return p.parse_args()


def extract_context_and_reply(chatml_text: str):
    """Parse a ChatML string back into context and expected reply."""
    try:
        user_start   = chatml_text.find('<|im_start|>user\n') + len('<|im_start|>user\n')
        user_end     = chatml_text.find('\n<|im_end|>\n<|im_start|>assistant\n')
        asst_start   = chatml_text.find('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')
        asst_end     = chatml_text.rfind('<|im_end|>')

        user_content = chatml_text[user_start:user_end].strip()
        reply        = chatml_text[asst_start:asst_end].strip()
        return user_content, reply
    except Exception:
        return "", ""


def redact_text(text: str, keep: int = 12) -> str:
    """Return a short preview without storing the full sensitive text."""
    text = (text or "").strip()
    if len(text) <= keep:
        return text
    return text[:keep] + "..."


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args = parse_args()

    test_path = DATA_DIR / "exports" / "test.jsonl"
    if not test_path.exists():
        print(f"[ERROR] test.jsonl not found at {test_path}")
        print("Run: python scripts/run_pipeline.py first")
        sys.exit(1)

    # Load test samples
    samples = []
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line)['text'])

    samples = samples[:args.n]
    print(f"\n[INFO] Evaluating {len(samples)} samples...")

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    results = []
    for i, chatml in enumerate(samples):
        user_content, true_reply = extract_context_and_reply(chatml)
        if not user_content:
            continue

        # Build prompt (everything except the assistant's reply)
        prompt = chatml[:chatml.find('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.85,
                top_p=0.92,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        record = {
            'id': i,
            'generated': generated,
            'context_preview': redact_text(user_content),
            'true_preview': redact_text(true_reply),
        }
        if args.include_raw_text:
            record['context'] = user_content
            record['true'] = true_reply
        results.append(record)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Evaluation complete — {len(results)} samples")
    print(f"   Results: {output_path}")
    if not args.include_raw_text:
        print("   Saved in redacted mode (use --include-raw-text to store full text)")
    print(f"\nNext: python scripts/show_eval.py {output_path}")


if __name__ == "__main__":
    main()
