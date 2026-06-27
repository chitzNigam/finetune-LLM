"""
src/features/cleaner.py

Cleans message text for Hinglish WhatsApp data.
- Removes Devanagari script (pure Hindi)
- Optionally removes pure English words
- Optionally filters NSFW words in English, Hinglish, and Hindi
- Strips PII (phone numbers, emails, UPI IDs)
- Preserves: emojis, typos, capitalization, punctuation style
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ── Regex Patterns ────────────────────────────────────────────────────────────

RE_DEVANAGARI   = re.compile(r'[\u0900-\u097F]+')
RE_URL          = re.compile(r'https?://\S+|www\.\S+')
RE_PHONE        = re.compile(r'(\+91|0)?[6-9]\d{9}')
RE_EMAIL        = re.compile(r'\b[\w.+-]+@[\w-]+\.\w+\b')
RE_UPI          = re.compile(r'\b[\w.]+@(okaxis|oksbi|okicici|ybl|paytm|upi)\b')
RE_AADHAAR_LIKE = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
RE_WHITESPACE   = re.compile(r'\s{2,}')
RE_TOKEN_CORE   = re.compile(r'(^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$)')


# ── English Wordlist ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_english_wordlist() -> set[str]:
    """
    Load a set of common English words.
    We use a curated list — NOT a full dictionary — to avoid
    stripping Hinglish words that are spelled like English
    (e.g., 'but', 'so', 'ok' are fine to keep).
    """
    # Minimal set — words that are clearly English-only
    # and unlikely to be used as Hinglish
    wordlist_path = Path(__file__).parent.parent.parent / "data" / "english_wordlist.txt"
    if wordlist_path.exists():
        return set(wordlist_path.read_text().lower().split())

    # Fallback: very conservative built-in list
    # Expand this file if you want more aggressive English filtering
    return {
        "the", "a", "an", "and", "or", "but", "with", "this",
        "that", "these", "those", "from", "have", "has", "had",
        "will", "would", "should", "could", "shall", "may",
        "might", "must", "been", "being", "there", "their", "they",
        "them", "then", "than", "when", "where", "while", "which",
        "who", "what", "how", "why", "each", "both", "some",
        "into", "onto", "upon", "about", "above", "below",
        "between", "through", "during", "before", "after",
        "because", "therefore", "however", "although", "though",
    }


@lru_cache(maxsize=1)
def _load_nsfw_wordlists() -> dict[str, set[str]]:
    """
    Load customizable NSFW wordlists grouped by language bucket.
    Falls back to a compact built-in set when no data file is present.
    """
    wordlist_path = Path(__file__).parent.parent.parent / "data" / "nsfw_words.json"
    if wordlist_path.exists():
        with open(wordlist_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            key: {str(word).strip().lower() for word in words if str(word).strip()}
            for key, words in data.items()
        }

    return {
        "english": {
            "fuck", "fucking", "fucked", "motherfucker", "motherfucking",
            "bitch", "bitches", "asshole", "bastard", "dick", "cock",
            "pussy", "slut", "whore", "cum", "cumming",
        },
        "hinglish": {
            "bc", "bkl", "bhenchod", "behenchod", "madarchod", "mc",
            "chutiya", "chutiya", "chuitya", "lund", "lawda", "lauda",
            "gaand", "gand", "randi", "harami", "kamina", "kaminey",
        },
        "hindi": {
            "भोसड़ी", "भोसड़ी", "भोसडी", "बहनचोद", "बेहनचोद", "मादरचोद",
            "चूतिया", "चुतिया", "लौड़ा", "लौंडा", "लंड", "गांड", "रंडी",
            "हरामी", "कमीना",
        },
    }


def _normalize_token(token: str) -> str:
    token = RE_TOKEN_CORE.sub("", token.strip().lower())
    return token


def remove_devanagari(text: str) -> str:
    return RE_DEVANAGARI.sub('', text)


def remove_pure_english(text: str) -> str:
    """
    Remove tokens that are in the English wordlist.
    Conservative — only removes unambiguous English function words.
    """
    wordlist = _load_english_wordlist()
    tokens = text.split()
    kept = [t for t in tokens if t.lower() not in wordlist]
    return ' '.join(kept)


def strip_pii(text: str) -> str:
    text = RE_URL.sub('[URL]', text)
    text = RE_PHONE.sub('[PHONE]', text)
    text = RE_EMAIL.sub('[EMAIL]', text)
    text = RE_UPI.sub('[UPI]', text)
    text = RE_AADHAAR_LIKE.sub('[ID]', text)
    return text


def normalize_whitespace(text: str) -> str:
    return RE_WHITESPACE.sub(' ', text).strip()


def filter_nsfw_words(
    text: str,
    remove_english_nsfw: bool = True,
    remove_hinglish_nsfw: bool = True,
    remove_hindi_nsfw: bool = True,
    replacement: str = "[NSFW]",
) -> str:
    """
    Replace tokens matching the configured NSFW lists while preserving
    surrounding punctuation where possible.
    """
    lists = _load_nsfw_wordlists()
    active_words = set()
    if remove_english_nsfw:
        active_words.update(lists.get("english", set()))
    if remove_hinglish_nsfw:
        active_words.update(lists.get("hinglish", set()))
    if remove_hindi_nsfw:
        active_words.update(lists.get("hindi", set()))

    if not active_words:
        return text

    filtered_tokens = []
    for token in text.split():
        normalized = _normalize_token(token)
        filtered_tokens.append(replacement if normalized in active_words else token)

    return " ".join(filtered_tokens)


def clean_text(
    text: str,
    remove_hindi: bool = True,
    remove_english: bool = False,  # Conservative default — change to True if needed
    filter_english_nsfw: bool = False,
    filter_hinglish_nsfw: bool = False,
    filter_hindi_nsfw: bool = False,
    strip_pii_flag: bool = True,
) -> Optional[str]:
    """
    Full cleaning pipeline for a single message.

    Returns None if the message is empty after cleaning
    (caller should drop these rows).
    """
    if not text or not isinstance(text, str):
        return None

    # Skip media placeholders
    if '<Media omitted>' in text or 'image omitted' in text.lower():
        return None

    if remove_hindi:
        text = remove_devanagari(text)

    if strip_pii_flag:
        text = strip_pii(text)

    if remove_english:
        text = remove_pure_english(text)

    if filter_english_nsfw or filter_hinglish_nsfw or filter_hindi_nsfw:
        text = filter_nsfw_words(
            text,
            remove_english_nsfw=filter_english_nsfw,
            remove_hinglish_nsfw=filter_hinglish_nsfw,
            remove_hindi_nsfw=filter_hindi_nsfw,
        )

    text = normalize_whitespace(text)

    # Drop if effectively empty (only punctuation/emojis left is fine,
    # but pure whitespace or single chars that lost meaning → None)
    if len(text.strip()) <= 1:
        return None

    return text


def clean_dataframe(df, **kwargs):
    """Apply clean_text to the 'text' column of a DataFrame."""
    import pandas as pd
    df = df.copy()
    df['text'] = df['text'].apply(lambda t: clean_text(t, **kwargs))
    before = len(df)
    df = df.dropna(subset=['text'])
    after = len(df)
    print(f"[INFO] Cleaning dropped {before - after:,} empty messages ({after:,} remaining)")
    return df.reset_index(drop=True)
