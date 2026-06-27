"""
scripts/preview_data.py

Quick preview of raw exports before running the full pipeline.
Useful to check parsing and see your data distribution.

Usage:
    python scripts/preview_data.py
    python scripts/preview_data.py --file data/raw/rahul.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.config import DATA_DIR, YOUR_NAME


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--file', type=str, default=None,
                   help='Preview a specific file (default: all)')
    p.add_argument('--n', type=int, default=10,
                   help='Number of sample messages to show')
    p.add_argument('--show-raw-text', action='store_true',
                   help='Print raw message text instead of a redacted preview')
    return p.parse_args()


def redact_text(text: str, keep: int = 24) -> str:
    text = (text or "").strip()
    if len(text) <= keep:
        return text
    return text[:keep] + '...'


def main():
    args = parse_args()

    from src.parser.whatsapp_parser import parse_file, parse_all

    if args.file:
        df = parse_file(args.file)
        print(f"\nFile: {args.file}")
    else:
        raw_dir = DATA_DIR / "raw"
        files = list(raw_dir.glob("*.txt"))
        if not files:
            print(f"\n[ERROR] No .txt files found in {raw_dir}")
            print("Export your WhatsApp chats and drop them there.")
            sys.exit(1)
        df = parse_all(raw_dir)

    if df.empty:
        print("[ERROR] No messages parsed.")
        sys.exit(1)

    print(f"\n{'═'*52}")
    print(f"  Preview — {len(df):,} total messages")
    print(f"{'═'*52}")

    # Senders
    senders = df['sender'].value_counts()
    print(f"\n👥 Top senders:")
    for sender, count in senders.head(8).items():
        you_marker = " ← YOU" if sender == YOUR_NAME else ""
        print(f"   {sender:<30s} {count:>6,}{you_marker}")

    # Your messages
    your_count = (df['sender'] == YOUR_NAME).sum()
    pct = your_count / len(df) * 100
    print(f"\n   Your messages: {your_count:,} ({pct:.1f}% of total)")

    if your_count == 0:
        print(f"\n   ⚠️  No messages from '{YOUR_NAME}' found!")
        print(f"      Check YOUR_NAME in src/training/config.py")
        print(f"      It must match exactly how you appear in exports.")
        print(f"      Try one of: {list(senders.head(5).index)}")

    # Sample messages
    print(f"\n📝 Sample messages (last {args.n}):")
    sample = df.tail(args.n)[['datetime', 'sender', 'text']]
    for _, row in sample.iterrows():
        dt  = row['datetime'].strftime('%d %b %H:%M')
        sndr = row['sender'][:20]
        source_text = row['text'] if args.show_raw_text else redact_text(row['text'])
        txt  = source_text[:60] + ('...' if len(source_text) > 60 else '')
        you = "→" if row['sender'] == YOUR_NAME else " "
        print(f"  {you} [{dt}] {sndr:<20s}: {txt}")

    if not args.show_raw_text:
        print("\n   Raw text is redacted by default. Use --show-raw-text to reveal it.")

    print()


if __name__ == "__main__":
    main()
