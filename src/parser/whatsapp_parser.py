"""
src/parser/whatsapp_parser.py

Parses WhatsApp .txt export files into a structured DataFrame.
Handles multi-line messages, system messages, and date format variations.
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


# Matches: "15/03/2024, 22:14 - Name: message"
# Also handles AM/PM format used in some iOS exports
WHATSAPP_PATTERN = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s*'
    r'(\d{1,2}:\d{2}(?:[\s\u00A0\u202F]*[AaPp][Mm])?)\s*-\s*'
    r'([^:]+):\s(.*)$'
)

SYSTEM_MESSAGES = [
    "Messages and calls are end-to-end encrypted",
    "This message was deleted",
    "You deleted this message",
    "missed voice call",
    "missed video call",
    "created group",
    "added you",
    "left",
    "changed the group",
    "changed their phone number",
    "Your security code with",
]


def _is_system_message(sender: str, text: str) -> bool:
    if sender.strip() in ("", "~"):
        return True
    for marker in SYSTEM_MESSAGES:
        if marker.lower() in text.lower():
            return True
    return False


def _parse_datetime(date_str: str, time_str: str):
    time_str = (
        time_str.replace("\u202F", " ")
                .replace("\u00A0", " ")
                .strip()
                .upper()
    )

    s = f"{date_str} {time_str}"

    for fmt in (
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%m/%d/%y %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%d/%m/%y %I:%M %p",
        "%m/%d/%y %I:%M %p",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    return None


def parse_file(filepath: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Parse a single WhatsApp .txt export file.

    Returns a DataFrame with columns:
        datetime, sender, text, source_file, is_media
    """
    filepath = Path(filepath)
    rows = []
    current_row = None

    try:
        lines = filepath.read_text(encoding=encoding, errors="replace").splitlines()
    except Exception as e:
        print(f"[WARN] Could not read {filepath.name}: {e}")
        return pd.DataFrame()

    for line in lines:
        line = (
            line.replace("\u202F", " ")
                .replace("\u00A0", " ")
        )
        match = WHATSAPP_PATTERN.match(line)
        if match:
            # Save previous row
            if current_row:
                rows.append(current_row)

            date_str, time_str, sender, text = match.groups()
            dt = _parse_datetime(date_str, time_str)

            if dt is None:
                current_row = None
                continue

            if _is_system_message(sender, text):
                current_row = None
                continue

            is_media = "<Media omitted>" in text or "image omitted" in text.lower()

            current_row = {
                "datetime": dt,
                "sender": sender.strip(),
                "text": text.strip(),
                "source_file": filepath.stem,
                "is_media": is_media,
            }
        else:
            # Continuation of previous message (multi-line)
            if current_row and line.strip():
                current_row["text"] += " " + line.strip()

    if current_row:
        rows.append(current_row)

    if not rows:
        print(f"[WARN] No messages parsed from {filepath.name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def parse_all(raw_dir: str | Path) -> pd.DataFrame:
    """
    Parse all .txt files in raw_dir and concatenate into one DataFrame.
    Deduplicates exact (datetime, sender, text) triples across files.
    """
    raw_dir = Path(raw_dir)
    txt_files = list(raw_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    print(f"[INFO] Found {len(txt_files)} export files")

    dfs = []
    for f in txt_files:
        df = parse_file(f)
        if not df.empty:
            print(f"  ✓ {f.name:40s} → {len(df):,} messages")
            dfs.append(df)

    if not dfs:
        raise ValueError("No messages could be parsed from any file")

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["datetime", "sender", "text"])
    after = len(combined)

    print(f"\n[INFO] Total: {after:,} messages ({before - after:,} duplicates removed)")
    return combined.sort_values("datetime").reset_index(drop=True)
