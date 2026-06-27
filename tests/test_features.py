import pandas as pd

from src.features.cleaner import clean_text
from src.features.context import build_context_windows
from src.features.engineer import tag_relationships


def test_tag_relationships_labels_your_replies_from_source_file():
    df = pd.DataFrame(
        [
            {"sender": "Rahul", "source_file": "Rahul Chat"},
            {"sender": "You", "source_file": "Rahul Chat"},
        ]
    )

    tagged = tag_relationships(df, "data/contacts.example.json", "You")

    assert tagged["relationship"].tolist() == ["best_friend", "best_friend"]
    assert tagged["recipient_phone"].tolist() == ["+919876543210", "+919876543210"]


def test_build_context_windows_respects_session_boundaries():
    df = pd.DataFrame(
        [
            {
                "conversation_id": "abc",
                "source_file": "Rahul Chat",
                "session_id": 1,
                "datetime": pd.Timestamp("2024-01-01 10:00:00"),
                "sender": "Rahul",
                "text": "hi",
            },
            {
                "conversation_id": "abc",
                "source_file": "Rahul Chat",
                "session_id": 1,
                "datetime": pd.Timestamp("2024-01-01 10:01:00"),
                "sender": "You",
                "text": "yo",
            },
            {
                "conversation_id": "abc",
                "source_file": "Rahul Chat",
                "session_id": 2,
                "datetime": pd.Timestamp("2024-01-03 10:00:00"),
                "sender": "You",
                "text": "new session",
            },
        ]
    )

    contexts = build_context_windows(df, "You", 5)

    assert contexts["reply"].tolist() == ["yo", "new session"]
    assert contexts["context"].tolist() == ["Rahul: hi", "(start of conversation)"]


def test_clean_text_filters_english_and_hinglish_nsfw_words():
    text = "this is fucking wild bc"

    cleaned = clean_text(
        text,
        remove_hindi=False,
        filter_english_nsfw=True,
        filter_hinglish_nsfw=True,
    )

    assert cleaned == "this is [NSFW] wild [NSFW]"


def test_clean_text_filters_hindi_nsfw_words_without_dropping_all_devanagari():
    text = "तू हरामी है"

    cleaned = clean_text(
        text,
        remove_hindi=False,
        filter_hindi_nsfw=True,
    )

    assert cleaned == "तू [NSFW] है"


def test_clean_text_drops_media_placeholders():
    cleaned = clean_text("<Media omitted>", remove_hindi=False)

    assert cleaned is None
