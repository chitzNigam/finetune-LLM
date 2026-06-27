"""
scripts/dataset_stats.py

Prints statistics about your processed dataset.
Run after: python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.config import DATA_DIR


def main():
    import pandas as pd

    print("\n" + "═"*52)
    print("  Dataset Statistics")
    print("═"*52)

    # ── Raw parse stats ───────────────────────────────────
    parsed_path = DATA_DIR / "processed" / "parsed.parquet"
    if parsed_path.exists():
        df = pd.read_parquet(parsed_path)
        print(f"\n📁 Parsed messages:       {len(df):>10,}")
        print(f"   Date range:            {df['datetime'].min().date()} → {df['datetime'].max().date()}")
        print(f"   Unique chats:          {df['source_file'].nunique():>10,}")
        print(f"   Media messages:        {df['is_media'].sum():>10,}")

    # ── Enriched stats ────────────────────────────────────
    enriched_path = DATA_DIR / "processed" / "enriched.parquet"
    if enriched_path.exists():
        df = pd.read_parquet(enriched_path)
        from src.training.config import YOUR_NAME
        your_df = df[df['sender'] == YOUR_NAME]

        print(f"\n👤 Your messages:         {len(your_df):>10,}")
        print(f"   Avg reply length:      {your_df['message_length'].mean():>10.1f} chars")
        print(f"   Avg word count:        {your_df['word_count'].mean():>10.1f} words")
        print(f"   Messages with emoji:   {your_df['has_emoji'].sum():>10,}")

        if 'relationship' in df.columns:
            print(f"\n📊 By relationship:")
            counts = your_df['relationship'].value_counts()
            for rel, count in counts.items():
                bar = "█" * (count * 30 // counts.max())
                print(f"   {rel:<20s} {count:>6,}  {bar}")

        if 'hour' in df.columns:
            print(f"\n🕐 Your reply hours (top 5):")
            top_hours = your_df['hour'].value_counts().head(5)
            for hour, count in top_hours.items():
                period = "AM" if hour < 12 else "PM"
                h12 = hour % 12 or 12
                print(f"   {h12:2d}:00 {period}  →  {count:,} messages")

    # ── Training dataset stats ────────────────────────────
    print(f"\n📦 Training splits:")
    total = 0
    for split in ['train', 'val', 'test']:
        path = DATA_DIR / "exports" / f"{split}.jsonl"
        if path.exists():
            count = sum(1 for _ in open(path, encoding='utf-8'))
            total += count
            print(f"   {split:<8s}  {count:>8,} samples")

    if total:
        print(f"   {'TOTAL':<8s}  {total:>8,} samples")

        if total < 3000:
            print(f"\n   ⚠️  Low sample count. Export more chats for better results.")
        elif total < 8000:
            print(f"\n   ✅ Decent dataset. Should produce reasonable results.")
        else:
            print(f"\n   ✅ Good dataset size. Expect solid style capture.")

    print()


if __name__ == "__main__":
    main()
