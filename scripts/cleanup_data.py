"""
scripts/cleanup_data.py

Deletes raw and processed chat data after training is complete.
Keeps model weights. Run this for privacy once you're satisfied with the model.

Usage:
    python scripts/cleanup_data.py
    python scripts/cleanup_data.py --yes   (skip confirmation)
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.config import DATA_DIR, MODELS_DIR


SENSITIVE_PATHS = [
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    DATA_DIR / "exports",
]

KEEP_PATHS = [
    MODELS_DIR / "final",
    DATA_DIR / "contacts.json",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--yes', action='store_true', help='Skip confirmation')
    p.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    return p.parse_args()


def main():
    args = parse_args()

    print("\n⚠️  Privacy Cleanup")
    print("═"*44)
    print("\nThis will permanently delete:")
    total_size = 0
    for path in SENSITIVE_PATHS:
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            total_size += size
            print(f"  🗑️  {path}  ({size / 1e6:.1f} MB)")

    print(f"\nTotal: {total_size / 1e6:.1f} MB")
    print("\nThis will KEEP:")
    for path in KEEP_PATHS:
        if path.exists():
            print(f"  ✅ {path}")

    if args.dry_run:
        print("\n[DRY RUN] Nothing deleted.")
        return

    if not args.yes:
        confirm = input("\nType 'delete' to confirm: ").strip()
        if confirm != 'delete':
            print("Cancelled.")
            return

    for path in SENSITIVE_PATHS:
        if path.exists():
            shutil.rmtree(path)
            # Recreate empty dirs so the project still works
            path.mkdir(parents=True, exist_ok=True)
            (path / ".gitkeep").touch()
            print(f"  ✓ Deleted {path}")

    print("\n✅ Raw data deleted. Model weights preserved.")
    print("   Your trained model is at: models/final/\n")


if __name__ == "__main__":
    main()
