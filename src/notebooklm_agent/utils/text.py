"""Text sanitization for display in various output formats.

Key problem solved: NotebookLM responses contain Markdown, <code> tags,
special characters that crash Telegram's parse_mode=MarkdownV2.
This module strips all that to plain text safely.
"""

import re


def sanitize_for_display(text: str) -> str:
    """Sanitize NotebookLM response for plain text display.

    Removes <code> tags but keeps their content.
    Strips HTML tags. Preserves Unicode and emojis.
    Normalizes whitespace but keeps paragraph breaks.
    """
    if not text:
        return ""

    # Remove <code> tags but keep content
    text = re.sub(r"</?code>", "", text)

    # Remove other HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse multiple spaces (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def sanitize_for_telegram(text: str) -> str:
    """Sanitize text for Telegram HTML parse_mode.

    Telegram supports a subset of HTML. This escapes plain text
    while preserving intentional HTML tags (b, i, code, pre, a).
    """
    if not text:
        return ""

    # Remove <code> tags but keep content
    text = re.sub(r"</?code>", "", text)

    # Escape HTML-special characters that aren't in allowed tags
    # Allowed tags: b, i, u, s, code, pre, a, tg-spoiler, blockquote
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Collapse excessive whitespace
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def truncate(text: str, max_len: int = 4096) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "\n\n... (truncated)"


def chunk_for_telegram(text: str, max_len: int = 4096) -> list[str]:
    """Split text into Telegram-message-sized chunks.

    Tries to split on paragraph boundaries (double newline).
    Falls back to sentence boundaries, then to hard cuts.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        # Try to split on paragraph boundary
        split_at = remaining.rfind("\n\n", 0, max_len)
        if split_at > max_len // 2:
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at + 2 :]
            continue

        # Try to split on sentence boundary
        split_at = remaining.rfind(". ", 0, max_len)
        if split_at > max_len // 2:
            chunks.append(remaining[: split_at + 1])
            remaining = remaining[split_at + 1 :].lstrip()
            continue

        # Hard cut
        chunks.append(remaining[:max_len])
        remaining = remaining[max_len:].lstrip()

    return chunks
