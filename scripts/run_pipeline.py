"""
scripts/run_pipeline.py

End-to-end data pipeline:
  raw .txt exports → cleaned dataset → formatted .jsonl files

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --stage parse
    python scripts/run_pipeline.py --stage clean
    python scripts/run_pipeline.py --stage features
    python scripts/run_pipeline.py --stage format
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import (
    DATA_DIR, CONTACTS_FILE, YOUR_NAME,
    REMOVE_HINDI, REMOVE_ENGLISH,
    FILTER_NSFW_ENGLISH, FILTER_NSFW_HINGLISH, FILTER_NSFW_HINDI,
    CONTEXT_WINDOW_SIZE, CUTOFF_MONTHS,
    MIN_REPLY_LENGTH, MAX_REPLY_LENGTH,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['parse', 'clean', 'features', 'format', 'all'],
                   default='all')
    return p.parse_args()


def stage_parse():
    print("\n" + "═"*50)
    print("  STAGE 1: Parsing WhatsApp exports")
    print("═"*50)

    from src.parser.whatsapp_parser import parse_all

    raw_dir = DATA_DIR / "raw"
    out_path = DATA_DIR / "processed" / "parsed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_all(raw_dir)

    # Apply date cutoff
    if CUTOFF_MONTHS > 0:
        cutoff = datetime.now() - timedelta(days=CUTOFF_MONTHS * 30)
        before = len(df)
        df = df[df['datetime'] >= cutoff]
        print(f"[INFO] Date cutoff ({CUTOFF_MONTHS}mo): {before:,} → {len(df):,} messages")

    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    return df


def stage_clean(df=None):
    print("\n" + "═"*50)
    print("  STAGE 2: Cleaning text")
    print("═"*50)
    import pandas as pd
    from src.features.cleaner import clean_dataframe

    if df is None:
        df = pd.read_parquet(DATA_DIR / "processed" / "parsed.parquet")

    out_path = DATA_DIR / "processed" / "cleaned.parquet"
    df = clean_dataframe(
        df,
        remove_hindi=REMOVE_HINDI,
        remove_english=REMOVE_ENGLISH,
        filter_english_nsfw=FILTER_NSFW_ENGLISH,
        filter_hinglish_nsfw=FILTER_NSFW_HINGLISH,
        filter_hindi_nsfw=FILTER_NSFW_HINDI,
    )
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    return df


def stage_features(df=None):
    print("\n" + "═"*50)
    print("  STAGE 3: Feature engineering")
    print("═"*50)
    import pandas as pd
    from src.features.engineer import run_all

    if df is None:
        df = pd.read_parquet(DATA_DIR / "processed" / "cleaned.parquet")

    out_path = DATA_DIR / "processed" / "enriched.parquet"
    contacts_path = CONTACTS_FILE if CONTACTS_FILE.exists() else None

    if not contacts_path:
        print("[WARN] contacts.json not found — relationships will be 'acquaintance'")
        print(f"       Copy data/contacts.example.json to data/contacts.json and fill it in")

    df = run_all(df, YOUR_NAME, contacts_path)
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    return df


def stage_format(df=None):
    print("\n" + "═"*50)
    print("  STAGE 4: Building context windows & formatting")
    print("═"*50)
    import pandas as pd
    from src.features.context import build_context_windows, format_dataset

    if df is None:
        df = pd.read_parquet(DATA_DIR / "processed" / "enriched.parquet")

    # Only keep YOUR replies as training targets
    your_messages_count = (df['sender'] == YOUR_NAME).sum()
    print(f"[INFO] Your messages (training targets): {your_messages_count:,}")

    contexts_df = build_context_windows(df, YOUR_NAME, CONTEXT_WINDOW_SIZE)

    export_dir = DATA_DIR / "exports"
    paths = format_dataset(
        contexts_df,
        YOUR_NAME,
        export_dir,
        min_reply_length=MIN_REPLY_LENGTH,
        max_reply_length=MAX_REPLY_LENGTH,
    )

    # Save contexts too (useful for analysis)
    contexts_df.to_parquet(DATA_DIR / "processed" / "contexts.parquet", index=False)

    print(f"\n✓ Dataset ready in {export_dir}")
    return paths


def print_summary():
    import json

    train_path = DATA_DIR / "exports" / "train.jsonl"
    if not train_path.exists():
        return

    count = sum(1 for _ in open(train_path))
    print("\n" + "═"*50)
    print("  PIPELINE COMPLETE")
    print("═"*50)
    print(f"\n  Training samples: {count:,}")
    print(f"\n  Next step:")
    print(f"    python src/training/train.py\n")


def main():
    args = parse_args()
    stage = args.stage

    df = None

    if stage in ('parse', 'all'):
        df = stage_parse()

    if stage in ('clean', 'all'):
        df = stage_clean(df)

    if stage in ('features', 'all'):
        df = stage_features(df)

    if stage in ('format', 'all'):
        stage_format(df)
        print_summary()


if __name__ == "__main__":
    main()
