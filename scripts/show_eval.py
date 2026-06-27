"""
scripts/show_eval.py

Displays evaluation results in a readable format.
Shows true replies vs model-generated replies side by side.

Usage:
    python scripts/show_eval.py results/eval_output.json
    python scripts/show_eval.py results/eval_output.json --n 20
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('file', type=str)
    p.add_argument('--n', type=int, default=20)
    return p.parse_args()


def avg_len(texts):
    return sum(len(t) for t in texts) / max(len(texts), 1)


def get_value(record, raw_key, preview_key):
    return record.get(raw_key) or record.get(preview_key, "")


def main():
    args = parse_args()

    with open(args.file, encoding='utf-8') as f:
        results = json.load(f)

    print(f"\n{'═'*60}")
    print(f"  Evaluation Results — {len(results)} samples")
    print(f"{'═'*60}")

    true_texts = [get_value(r, 'true', 'true_preview') for r in results]
    gen_texts  = [r['generated'] for r in results]

    print(f"\n📊 Aggregate Stats:")
    print(f"   Avg true reply length:      {avg_len(true_texts):.1f} chars")
    print(f"   Avg generated reply length: {avg_len(gen_texts):.1f} chars")

    # Length ratio (style check — should be close to 1.0)
    ratio = avg_len(gen_texts) / max(avg_len(true_texts), 1)
    status = "✅" if 0.6 <= ratio <= 1.6 else "⚠️"
    print(f"   Length ratio (gen/true):    {ratio:.2f}  {status}")

    # Emoji usage
    import re
    emoji_re = re.compile(r'[\U0001F300-\U0001FFFF\U00002600-\U000027FF]')
    true_emoji  = sum(1 for t in true_texts if emoji_re.search(t)) / max(len(true_texts), 1)
    gen_emoji   = sum(1 for t in gen_texts  if emoji_re.search(t)) / max(len(gen_texts), 1)
    print(f"\n   Emoji usage (true):         {true_emoji:.1%}")
    print(f"   Emoji usage (generated):    {gen_emoji:.1%}")

    # Side-by-side samples
    print(f"\n{'─'*60}")
    print(f"  Sample Comparisons (showing {min(args.n, len(results))})")
    print(f"{'─'*60}")

    for r in results[:args.n]:
        # Show last line of context
        context_value = get_value(r, 'context', 'context_preview')
        context_lines = context_value.strip().split('\n')
        last_ctx = context_lines[-2] if len(context_lines) >= 2 else context_lines[-1]
        # Strip metadata line
        if last_ctx.startswith('['):
            last_ctx = context_lines[-3] if len(context_lines) >= 3 else "..."

        print(f"\n  Context : {last_ctx[:70]}")
        print(f"  True    : {get_value(r, 'true', 'true_preview')[:80]}")
        print(f"  Model   : {r['generated'][:80]}")
        print(f"  {'─'*56}")

    print(f"\n💡 Manual eval tip:")
    print(f"   Share a mix of true/generated replies with a friend")
    print(f"   Target: they can't reliably tell which is you (~40%+ confusion rate)\n")


if __name__ == "__main__":
    main()
