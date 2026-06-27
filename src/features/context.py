"""
src/features/context.py

Builds context windows around your replies and formats
them into ChatML instruction format for training.
"""

import pandas as pd
from typing import Optional


CONTEXT_WINDOW = 5  # Number of messages before your reply to include as context

SYSTEM_PROMPT = """You are replicating {your_name}'s WhatsApp texting style exactly.
You will receive recent conversation messages and contextual metadata.
Generate a reply that matches {your_name}'s natural texting style for this relationship.
Keep it authentic — match their typical length, tone, vocabulary, and emoji usage.
This is texting, not formal writing. Be natural."""


def build_context_windows(
    df: pd.DataFrame,
    your_name: str,
    window_size: int = CONTEXT_WINDOW,
) -> pd.DataFrame:
    """
    For each of your messages, extract the preceding window_size messages
    as context. Returns a new DataFrame of (context, reply, metadata) rows.
    """
    samples = []

    # Process per conversation/session to avoid bleed across chats or long gaps.
    group_cols = ['conversation_id', 'source_file']
    if 'session_id' in df.columns:
        group_cols.append('session_id')

    for _, conv_df in df.groupby(group_cols):
        conv_df = conv_df.sort_values('datetime').reset_index(drop=True)
        your_indices = conv_df[conv_df['sender'] == your_name].index.tolist()

        for idx in your_indices:
            # Get context window (messages before this one)
            start = max(0, idx - window_size)
            context_rows = conv_df.iloc[start:idx]
            reply_row    = conv_df.iloc[idx]

            if context_rows.empty:
                # No context — skip or use empty context
                context_text = "(start of conversation)"
            else:
                context_lines = []
                for _, row in context_rows.iterrows():
                    prefix = your_name if row['sender'] == your_name else row['sender']
                    context_lines.append(f"{prefix}: {row['text']}")
                context_text = "\n".join(context_lines)

            # Determine who we're replying to (last non-you sender)
            other_senders = context_rows[context_rows['sender'] != your_name]
            recipient = other_senders.iloc[-1]['sender'] if not other_senders.empty else "unknown"

            samples.append({
                'context':           context_text,
                'reply':             reply_row['text'],
                'recipient_name':    recipient,
                'recipient_phone':   reply_row.get('recipient_phone', 'unknown'),
                'relationship':      reply_row.get('relationship', 'acquaintance'),
                'datetime':          reply_row['datetime'],
                'hour':              reply_row.get('hour', 12),
                'day_name':          reply_row.get('day_name', 'Monday'),
                'is_weekend':        reply_row.get('is_weekend', False),
                'timedelta_prev':    int(reply_row.get('timedelta_prev', 0)),
                'position':          reply_row.get('position', 'mid'),
                'reply_chain_depth': reply_row.get('reply_chain_depth', 0),
                'conversation_id':   reply_row.get('conversation_id', ''),
                'source_file':       reply_row.get('source_file', ''),
            })

    result = pd.DataFrame(samples)
    print(f"[INFO] Built {len(result):,} context-reply pairs")
    return result


def format_as_chatml(row: pd.Series, your_name: str) -> str:
    """
    Format a single context-reply pair into ChatML format.

    Example output:
    <|im_start|>system
    You are replicating Rohan's WhatsApp texting style...
    <|im_end|>
    <|im_start|>user
    Rahul: kal milte hain kya
    Rahul: bata na yaar
    [Relationship: best_friend | Hour: 22 | Day: Friday | Since last: 3600s | Position: mid]
    <|im_end|>
    <|im_start|>assistant
    haan bhai, evening chalega<|im_end|>
    """
    system = SYSTEM_PROMPT.format(your_name=your_name)

    metadata = (
        f"[Relationship: {row['relationship']} | "
        f"Hour: {row['hour']} | "
        f"Day: {row['day_name']} | "
        f"Since last msg: {row['timedelta_prev']}s | "
        f"Position: {row['position']}]"
    )

    user_content = f"{row['context']}\n{metadata}"

    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{row['reply']}<|im_end|>"
    )


def format_dataset(
    contexts_df: pd.DataFrame,
    your_name: str,
    output_dir: str,
    train_ratio: float = 0.85,
    val_ratio:   float = 0.10,
    # test gets the rest
    min_reply_length: int = 2,
    max_reply_length: int = 500,
    random_seed: int = 42,
) -> dict:
    """
    Format contexts into ChatML and split into train/val/test .jsonl files.
    Stratifies by relationship to ensure all categories are represented.
    """
    import json
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by reply length
    df = contexts_df[
        (contexts_df['reply'].str.len() >= min_reply_length) &
        (contexts_df['reply'].str.len() <= max_reply_length)
    ].copy()

    print(f"[INFO] {len(df):,} samples after length filtering")

    # Format all samples
    df['text'] = df.apply(lambda row: format_as_chatml(row, your_name), axis=1)

    # Stratified split by relationship
    train_dfs, val_dfs, test_dfs = [], [], []

    for rel, group in df.groupby('relationship'):
        n = len(group)
        if n < 10:
            train_dfs.append(group)
            continue

        n_test  = max(1, int(n * (1 - train_ratio - val_ratio)))
        n_val   = max(1, int(n * val_ratio))

        train_g, temp = train_test_split(group, test_size=n_test + n_val,
                                         random_state=random_seed)
        val_g, test_g = train_test_split(temp, test_size=n_test,
                                          random_state=random_seed)

        train_dfs.append(train_g)
        val_dfs.append(val_g)
        test_dfs.append(test_g)

    splits = {
        'train': pd.concat(train_dfs).sample(frac=1, random_state=random_seed),
        'val':   pd.concat(val_dfs).sample(frac=1, random_state=random_seed) if val_dfs else pd.DataFrame(),
        'test':  pd.concat(test_dfs).sample(frac=1, random_state=random_seed) if test_dfs else pd.DataFrame(),
    }

    paths = {}
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for text in split_df['text']:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
        paths[split_name] = out_path
        print(f"  ✓ {split_name:6s}: {len(split_df):,} samples → {out_path}")

    return paths
