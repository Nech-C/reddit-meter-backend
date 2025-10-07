# app/ml/reprocessing.py
import re
import html
import unicodedata

from typing import List


def normalize_unicode(text: str) -> str:
    """Normalize unicode using NFKC form."""
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def strip_blockquotes(text: str) -> str:
    """Remove Reddit-style blockquote lines starting with >"""
    return re.sub(r"(?m)^>+\s*", "", text)


def remove_markdown_links(text: str) -> str:
    """Remove [anchor](url) markdown links, keep just anchor text."""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)


def drop_bare_urls(text: str) -> str:
    """Remove plain URLs (e.g. https://example.com)."""
    return re.sub(r"https?://\S+", "", text)


def decode_html_entities(text: str) -> str:
    """Turn HTML entities like &amp; or &quot; into actual characters."""
    return html.unescape(text)


def strip_nonbreaking_spaces(text: str) -> str:
    """Convert non-breaking spaces (U+00A0) to normal space."""
    return text.replace("\xa0", " ")


def remove_control_chars(text: str) -> str:
    """Remove emoji and control characters (non-ASCII)."""
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text.encode("ascii", "ignore").decode()


def clean_text(
    text: str,
    *,
    normalize=True,
    strip_nbsp=True,
    decode_entities=True,
    strip_quotes=True,
    remove_md_links=True,
    remove_urls=True,
    collapse_ws=True,
    strip_controls=False,
) -> str:
    """Apply a full suite of text cleaning steps."""
    if normalize:
        text = normalize_unicode(text)
    if strip_nbsp:
        text = strip_nonbreaking_spaces(text)
    if decode_entities:
        text = decode_html_entities(text)
    if strip_quotes:
        text = strip_blockquotes(text)
    if remove_md_links:
        text = remove_markdown_links(text)
    if remove_urls:
        text = drop_bare_urls(text)
    if strip_controls:
        text = remove_control_chars(text)
    if collapse_ws:
        text = collapse_whitespace(text)
    return text


def prepare_for_input(
    title: str,
    post_text: str,
    comments: List[str],
    post_limit: int = 1400,
    comment_limit: int = 300,
) -> str:
    if not post_limit:
        post_limit = len(post_text)
    if not comment_limit:
        comment_limit = len(comments)
    title = clean_text(title)
    post_text = clean_text(post_text[:post_limit])
    comments = [clean_text(c) for c in comments[:comment_limit]]
    text = f"""TITLE: {title}
                POST: {post_text}
                COMMENTS: {" | ".join(comments)}"""

    return text
