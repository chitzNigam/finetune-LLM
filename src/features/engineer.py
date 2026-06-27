"""
src/features/engineer.py

Adds derived features to the parsed + cleaned DataFrame.
All features that don't require the contact map go here.
"""

import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional


SESSION_GAP_SECONDS = 3600  # 1 hour gap = new conversation session
GHOST_THRESHOLD_SECONDS = 86400  # 24 hours = likely ghosted


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timedelta_prev'] = (
        df['datetime'].diff().dt.total_seconds().fillna(0).clip(lower=0)
    )
    df['hour']        = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek      # 0=Mon, 6=Sun
    df['day_name']    = df['datetime'].dt.day_name()
    df['is_weekend']  = df['day_of_week'].isin([5, 6])
    df['month']       = df['datetime'].dt.month
    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Session break = gap > SESSION_GAP_SECONDS within the same chat
    df['session_break'] = (
        (df['timedelta_prev'] > SESSION_GAP_SECONDS) |
        (df['source_file'] != df['source_file'].shift(1))
    )
    df['session_id'] = df.groupby('source_file')['session_break'].cumsum()

    # Position in session: opener / mid / closer
    def tag_positions(group):
        positions = ['mid'] * len(group)
        positions[0] = 'opener'
        # Closer = last message in session before a long gap or end
        positions[-1] = 'closer'
        return positions

    df['position'] = 'mid'
    for (file, session), group in df.groupby(['source_file', 'session_id']):
        if len(group) >= 2:
            df.loc[group.index[0], 'position']  = 'opener'
            df.loc[group.index[-1], 'position'] = 'closer'

    return df


def add_conversation_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['conversation_id'] = df['source_file'].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()[:8]
    )
    return df


def add_message_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['message_length']   = df['text'].str.len()
    df['word_count']       = df['text'].str.split().str.len()
    import emoji

    df["emoji_count"] = df["text"].apply(
        lambda s: emoji.emoji_count(s) if isinstance(s, str) else 0
    )
    df["has_emoji"] = df["emoji_count"] > 0
    df['has_emoji']        = df['emoji_count'] > 0
    df['ends_ellipsis']    = df['text'].str.rstrip().str.endswith('...')
    df['ends_question']    = df['text'].str.rstrip().str.endswith('?')
    df['has_caps_word']    = df['text'].str.contains(r'\b[A-Z]{2,}\b', regex=True)
    df['is_single_word']   = df['word_count'] == 1
    df['is_very_short']    = df['word_count'] <= 3
    return df


def add_reply_chain_depth(df: pd.DataFrame, your_name: str) -> pd.DataFrame:
    df = df.copy()
    
    df['reply_chain_depth'] = 0
    depth = 0
    last_sender = None

    for i, row in df.iterrows():
        if row['sender'] != last_sender:
            depth = 0 if row['sender'] == your_name else depth + 1
        df.at[i, 'reply_chain_depth'] = depth
        last_sender = row['sender']

    return df


def add_ghosted_flag(df: pd.DataFrame, your_name: str) -> pd.DataFrame:
    """Flag messages that were never replied to (likely ghosted)."""
    df = df.copy()
    df['ghosted'] = False
    df['next_timedelta'] = df['timedelta_prev'].shift(-1).fillna(0)

    mask = (
        (df['sender'] != your_name) &
        (df['next_timedelta'] > GHOST_THRESHOLD_SECONDS)
    )
    df.loc[mask, 'ghosted'] = True
    df = df.drop(columns=['next_timedelta'])
    return df


def tag_relationships(
    df: pd.DataFrame,
    contacts_path: str | Path,
    your_name: str,
) -> pd.DataFrame:
    """
    Add recipient_phone and relationship columns using contacts.json.
    Falls back to 'acquaintance' for untagged contacts.
    """
    contacts_path = Path(contacts_path)
    if not contacts_path.exists():
        print(f"[WARN] contacts.json not found at {contacts_path}. "
              "All contacts will be tagged as 'acquaintance'.")
        df['recipient_phone'] = 'unknown'
        df['relationship']    = 'acquaintance'
        return df

    with open(contacts_path) as f:
        config = json.load(f)

    contacts     = config.get('contacts', {})
    group_chats  = config.get('group_chats', {})
    default_rel  = config.get('default_relationship', 'acquaintance')

    # Build name → (phone, relationship) lookup
    name_map = {}
    for phone, info in contacts.items():
        name_map[info['name'].lower()] = (phone, info['relationship'])

    def lookup(sender: str, source_file: str):
        # Check group chats first
        for group_name, rel in group_chats.items():
            if group_name.lower() in source_file.lower():
                return ('group', rel)

        candidates = []

        # For incoming messages, the sender is usually the contact name.
        sender_key = sender.lower().strip()
        if sender_key and sender_key != your_name.lower().strip():
            candidates.append(sender_key)

        # For your own replies, use the export filename as a fallback signal.
        source_key = source_file.lower().strip()
        if source_key:
            candidates.append(source_key)

        for key in candidates:
            if key in name_map:
                return name_map[key]
            for name, val in name_map.items():
                if name in key or key in name:
                    return val

        return ('unknown', default_rel)

    df = df.copy()
    results = df.apply(
        lambda row: lookup(row['sender'], row['source_file']), axis=1
    )
    df['recipient_phone'] = results.apply(lambda x: x[0])
    df['relationship']    = results.apply(lambda x: x[1])
    return df


def run_all(df: pd.DataFrame, your_name: str, contacts_path: Optional[Path] = None) -> pd.DataFrame:
    print("[INFO] Adding temporal features...")
    df = add_temporal_features(df)
    print("[INFO] Adding session features...")
    df = add_session_features(df)
    print("[INFO] Adding conversation IDs...")
    df = add_conversation_id(df)
    print("[INFO] Adding message features...")
    df = add_message_features(df)
    print("[INFO] Adding reply chain depth...")
    df = add_reply_chain_depth(df, your_name)
    print("[INFO] Adding ghost flags...")
    df = add_ghosted_flag(df, your_name)
    if contacts_path:
        print("[INFO] Tagging relationships...")
        df = tag_relationships(df, contacts_path, your_name)
    return df
