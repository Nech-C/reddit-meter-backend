import pytest

from app.ml.preprocessing import (
    normalize_unicode,
    collapse_whitespace,
    strip_blockquotes,
    remove_markdown_links,
    drop_bare_urls,
    decode_html_entities,
    strip_nonbreaking_spaces,
    remove_control_chars,
    clean_text,
    prepare_for_input,
)


@pytest.mark.parametrize(
    "text,expected", [("ﬃ", "ffi"), ("ℌ", "H"), ("①", "1"), ("a", "a"), ("", "")]
)
def test_normalize_unicode(text, expected):
    """Ensure Unicode normalization (NFKC) converts special forms correctly."""
    assert normalize_unicode(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [(" I can  do   this  ", "I can do this"), ("          ", ""), ("a\n\nb", "a b")],
)
def test_collapse_whitespace(text, expected):
    "Ensure consecutive whitespaces are collapsed into one and tailling whitespaces get removed."
    assert collapse_whitespace(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("This has no blockquotes!", "This has no blockquotes!"),
        ("> To be\n>>  or not\n>>>to be", "To be\nor not\nto be"),
        (">>>>>>>>>>>", ""),
        ("Don't remove>", "Don't remove>"),
    ],
)
def test_strip_blockquotes(text, expected):
    """Ensure blockquotes at the beginning of lines get removed."""
    assert strip_blockquotes(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is a [link](https://example.com)", "This is a link"),
        ("Multiple [links](a.com) here [too](b.org)", "Multiple links here too"),
        ("No links here", "No links here"),
        ("[Empty]() brackets", "[Empty]() brackets"),  # should remain unchanged
    ],
)
def test_remove_markdown_links(text, expected):
    """Ensure [anchor](url) patterns are replaced with anchor text only."""
    assert remove_markdown_links(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Visit https://example.com", "Visit "),
        ("http://a.com and https://b.org", " and "),
        ("no links", "no links"),
        ("end with https://a.com", "end with "),
    ],
)
def test_drop_bare_urls(text, expected):
    """Ensure bare URLs (http/https) are dropped."""
    assert drop_bare_urls(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("&lt;b&gt;bold&lt;/b&gt;", "<b>bold</b>"),
        ("no entities", "no entities"),
        ("Fish &gt; Chips &lt; Soup", "Fish > Chips < Soup"),
    ],
)
def test_decode_html_entities(text, expected):
    """Ensure HTML entities are decoded to their character equivalents."""
    assert decode_html_entities(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello\xa0World", "Hello World"),
        ("\xa0\xa0Leading spaces", "  Leading spaces"),
        ("NoNBSP", "NoNBSP"),
    ],
)
def test_strip_nonbreaking_spaces(text, expected):
    """Ensure non-breaking spaces (U+00A0) are replaced with normal spaces."""
    assert strip_nonbreaking_spaces(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello🙂", "Hello"),
        ("Café", "Caf"),
        ("Normal text", "Normal text"),
    ],
)
def test_remove_control_chars(text, expected):
    """Ensure non-ASCII and control characters are stripped."""
    assert remove_control_chars(text) == expected


@pytest.mark.parametrize(
    "text,kwargs,expected",
    [
        ("  A\u00a0B  ", {}, "A B"),  # nbsp, leading/trailing spaces
        # blockquotes
        ("> quoted line", {}, "quoted line"),
        # markdown link removal
        ("[link](https://a.com)", {}, "link"),
        # bare URL removal
        ("visit https://a.com today", {}, "visit today"),
        # HTML entities decode
        ("Tom &amp; Jerry", {}, "Tom & Jerry"),
        # control chars (only removed if strip_controls=True)
        ("bad\x01text", {}, "badtext"),
        ("bad\x01text", {"strip_controls": True}, "badtext"),
        # disabling flags
        ("[link](a.com)", {"remove_md_links": False}, "[link](a.com)"),
        ("https://abc.com", {"remove_urls": False}, "https://abc.com"),
        (" > keep quote", {"strip_quotes": False}, "> keep quote"),
    ],
)
def test_clean_text_combinations(text, kwargs, expected):
    """Verify clean_text chains steps correctly and honors flags."""
    result = clean_text(text, **kwargs)
    # collapse_whitespace always applied by default
    assert result == expected


def test_clean_text_pipeline_integration():
    """Test full pipeline on a combined messy input."""
    raw = "> [Hello](https://ex.com) &amp; bye  \xa0 https://foo.com🙂"
    cleaned = clean_text(raw, strip_controls=True)
    assert cleaned == "Hello & bye"


def test_prepare_for_input_basic():
    """Ensure prepare_for_input builds the expected formatted string."""
    title = "Hello"
    post = "This is [a link](https://a.com)"
    comments = ["Nice!", "Visit https://spam.com", "> quote"]

    text = prepare_for_input(title, post, comments, post_limit=100, comment_limit=20)

    # should remove markdown links and URLs
    assert "a link" in text
    assert "https://a.com" not in text
    assert "spam.com" not in text
    # blockquote marker removed
    assert "> quote" not in text
    assert "quote" in text
    # correct prefix sections
    assert text.strip().startswith("TITLE:")
    assert "POST:" in text
    assert "COMMENTS:" in text


def test_prepare_for_input_truncation():
    """Verify truncation limits and joining of comments."""
    title = "Short"
    post = "a" * 2000
    comments = [f"comment {i}" for i in range(10)]

    result = prepare_for_input(title, post, comments, post_limit=100, comment_limit=3)
    assert len(result) < len(post)  # truncated
    assert "comment 3" not in result  # only first 3 included
    assert "comment 2" in result
    assert result.count("comment") == 3


def test_prepare_for_input_empty_comments():
    """Empty comments list should not crash."""
    text = prepare_for_input("T", "Body", [])
    assert "COMMENTS:" in text
    assert "|" not in text  # no join artifacts
