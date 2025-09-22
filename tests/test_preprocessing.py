from app.ml import preprocessing as prep


def test_clean_text_full_pipeline():
    raw = "Hello &amp; world\n> quoted line\nvisit https://example.com [link](http://x.com)"
    cleaned = prep.clean_text(raw)
    assert cleaned == "Hello & world quoted line visit link"


def test_clean_text_with_options():
    text = "Value\u00a0with\tspaces https://example.com"
    cleaned = prep.clean_text(
        text,
        strip_nbsp=True,
        remove_urls=False,
        collapse_ws=False,
        strip_controls=True,
    )
    assert "\u00a0" not in cleaned
    assert "https://example.com" in cleaned
    assert "  " not in cleaned.replace("\t", "")


def test_prepare_for_input_limits_and_cleaning():
    out = prep.prepare_for_input(
        "  My Title  ",
        "Body text &amp; extras",
        ["First comment", "Second comment"],
        post_limit=9,
        comment_limit=1,
    )
    assert "TITLE: My Title" in out
    assert "POST: Body text" in out
    assert "COMMENTS: First comment" in out
    assert "Second comment" not in out
