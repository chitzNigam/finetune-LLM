"""
scripts/preview_cleaning.py

Preview how the cleaner transforms WhatsApp messages before running
the full pipeline.

Usage:
    python scripts/preview_cleaning.py
    python scripts/preview_cleaning.py --file data/raw/rahul.txt
    python scripts/preview_cleaning.py --n 20 --keep-hindi --filter-hindi-nsfw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import (
    DATA_DIR,
    REMOVE_HINDI,
    REMOVE_ENGLISH,
    FILTER_NSFW_ENGLISH,
    FILTER_NSFW_HINGLISH,
    FILTER_NSFW_HINDI,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default=None,
                   help="Preview a specific export file instead of all raw files")
    p.add_argument("--n", type=int, default=10,
                   help="Number of messages to preview")
    p.add_argument("--keep-hindi", action="store_true",
                   help="Do not strip Devanagari text for this preview")
    p.add_argument("--remove-english", action="store_true",
                   help="Remove conservative English filler words for this preview")
    p.add_argument("--filter-english-nsfw", action="store_true",
                   help="Replace English NSFW words with [NSFW] for this preview")
    p.add_argument("--filter-hinglish-nsfw", action="store_true",
                   help="Replace Hinglish NSFW words with [NSFW] for this preview")
    p.add_argument("--filter-hindi-nsfw", action="store_true",
                   help="Replace Hindi NSFW words with [NSFW] for this preview")
    p.add_argument("--show-unchanged", action="store_true",
                   help="Also show messages whose cleaned text did not change")
    p.add_argument("--show-raw-text", action="store_true",
                   help="Print full raw and cleaned text instead of redacted previews")
    return p.parse_args()


def redact_text(text: str | None, keep: int = 24) -> str:
    if text is None:
        return "[DROPPED]"
    text = text.strip()
    if len(text) <= keep:
        return text
    return text[:keep] + "..."


def load_messages(file_arg: str | None):
    from src.parser.whatsapp_parser import parse_all, parse_file

    if file_arg:
        df = parse_file(file_arg)
        label = file_arg
    else:
        raw_dir = DATA_DIR / "raw"
        files = list(raw_dir.glob("*.txt"))
        if not files:
            print(f"\n[ERROR] No .txt files found in {raw_dir}")
            print("Export your WhatsApp chats and drop them there first.")
            sys.exit(1)
        df = parse_all(raw_dir)
        label = str(raw_dir)

    if df.empty:
        print("\n[ERROR] No messages parsed.")
        sys.exit(1)

    return df, label


def main():
    from src.features.cleaner import clean_text

    args = parse_args()
    df, label = load_messages(args.file)

    remove_hindi = False if args.keep_hindi else REMOVE_HINDI
    remove_english = True if args.remove_english else REMOVE_ENGLISH
    filter_english_nsfw = True if args.filter_english_nsfw else FILTER_NSFW_ENGLISH
    filter_hinglish_nsfw = True if args.filter_hinglish_nsfw else FILTER_NSFW_HINGLISH
    filter_hindi_nsfw = True if args.filter_hindi_nsfw else FILTER_NSFW_HINDI

    preview = df[["datetime", "sender", "text"]].copy()
    preview["cleaned"] = preview["text"].apply(
        lambda text: clean_text(
            text,
            remove_hindi=remove_hindi,
            remove_english=remove_english,
            filter_english_nsfw=filter_english_nsfw,
            filter_hinglish_nsfw=filter_hinglish_nsfw,
            filter_hindi_nsfw=filter_hindi_nsfw,
        )
    )
    preview["changed"] = preview["text"] != preview["cleaned"]

    if not args.show_unchanged:
        changed = preview[preview["changed"] | preview["cleaned"].isna()]
        if not changed.empty:
            preview = changed

    preview = preview.tail(args.n)

    print("\n" + "═" * 60)
    print("  Cleaning Preview")
    print("═" * 60)
    print(f"\nSource: {label}")
    print(f"Messages parsed: {len(df):,}")
    print(f"Showing: {len(preview):,}")

    print("\nSettings:")
    print(f"  remove_hindi         = {remove_hindi}")
    print(f"  remove_english       = {remove_english}")
    print(f"  filter_english_nsfw  = {filter_english_nsfw}")
    print(f"  filter_hinglish_nsfw = {filter_hinglish_nsfw}")
    print(f"  filter_hindi_nsfw    = {filter_hindi_nsfw}")
    if not args.show_raw_text:
        print("  output_mode          = redacted")

    if preview.empty:
        print("\nNo changed rows to show with the current filters.")
        print("Try --show-unchanged or enable different filter flags.")
        print()
        return

    print("\nExamples:")
    for _, row in preview.iterrows():
        stamp = row["datetime"].strftime("%d %b %H:%M")
        raw_text = row['text'] if args.show_raw_text else redact_text(row['text'])
        cleaned_text = row['cleaned'] if args.show_raw_text else redact_text(row['cleaned'])
        print(f"\n[{stamp}] {row['sender']}")
        print(f"  Raw    : {raw_text}")
        print(f"  Cleaned: {cleaned_text}")

    dropped = preview["cleaned"].isna().sum()
    changed = preview["changed"].sum()
    print(f"\nSummary: {changed:,} changed, {dropped:,} dropped in shown rows\n")


if __name__ == "__main__":
    main()
