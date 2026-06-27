from pathlib import Path

from src.parser.whatsapp_parser import parse_all, parse_file


def test_parse_file_handles_multiline_system_messages_and_media(tmp_path: Path):
    chat_file = tmp_path / "Rahul Chat.txt"
    chat_file.write_text(
        "\n".join(
            [
                "15/03/2024, 22:14 - Messages and calls are end-to-end encrypted.",
                "15/03/2024, 22:15 - Rahul: first line",
                "still same message",
                "15/03/2024, 22:16 - You: <Media omitted>",
            ]
        ),
        encoding="utf-8",
    )

    df = parse_file(chat_file)

    assert list(df["sender"]) == ["Rahul", "You"]
    assert df.loc[0, "text"] == "first line still same message"
    assert bool(df.loc[1, "is_media"]) is True
    assert df.loc[0, "source_file"] == "Rahul Chat"


def test_parse_all_deduplicates_same_message_across_exports(tmp_path: Path):
    message = "15/03/2024, 22:15 - Rahul: same message"
    (tmp_path / "chat1.txt").write_text(message, encoding="utf-8")
    (tmp_path / "chat2.txt").write_text(message, encoding="utf-8")

    df = parse_all(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["text"] == "same message"
