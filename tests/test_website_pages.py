"""The pages: every value escaped, the viewport and stylesheet, the start page's presets, the
Viewer_Slot, and the static files' sizes."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from app import pages
from app.statuses import JobView
from tests.website_helpers import REPO_ROOT

HOSTILE = ['<script>alert(1)</script>', '"><img src=x onerror=alert(1)>', "' onmouseover='x",
           "</main>&amp;", "ünïcödé\u202e", "a\x00b"]


class Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, tuple[str, ...]]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, tuple(sorted(name for name, _ in attrs))))


def tags(html: str) -> list:
    parser = Tags()
    parser.feed(html)
    return parser.tags


def view(**changes) -> JobView:
    values = dict(job_id="20261201-120000-abcdef", preset="fast", started_at=None, status="done",
                  step=None, steps=5, readable_error=None, warnings=(), run_seconds=61.0,
                  step_times=(), result_file="clean.sog", has_cameras=True)
    values.update(changes)
    return JobView(**values)


@pytest.mark.parametrize("value", HOSTILE)
def test_values_are_escaped_everywhere(value: str) -> None:
    plain = "plain"
    built = [
        lambda v: pages.signin_page(v, v),
        lambda v: pages.showcase_page([("20261201-120000-abcdef", v, v)], csrf=v),
        lambda v: pages.result_page(view(readable_error=v, warnings=((v, v),), preset=v,
                                         status="failed"),
                                    csrf=v, poll_seconds=20, viewer=pages.Viewer(v, v), download=True,
                                    share=pages.ShareState(on=True, new_link=v), showcase_line=v,
                                    title=v),
        lambda v: pages.message_page(v, v, csrf=v),
        lambda v: pages.start_page([("fast", 30)], "fast", csrf=v, stop_line=None, part_retries=3),
    ]
    for page in built:
        hostile, safe = page(value), page(plain)
        assert tags(hostile) == tags(safe)
        assert "<script>alert" not in hostile and "<img" not in hostile


def test_every_page_has_the_viewport_and_the_stylesheet_and_no_inline_script() -> None:
    for html in (pages.signin_page("t"), pages.not_found_page(), pages.error_page(),
                 pages.job_list_page([], csrf="c"), pages.showcase_page([], csrf="c")):
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
        assert '<link rel="stylesheet" href="/static/style.css">' in html
        assert "<script>" not in html and "style=" not in html and " on" + "click" not in html


def test_the_start_page_offers_the_presets_with_the_default_checked() -> None:
    html = pages.start_page([("fast", 30), ("default", 60), ("quality", 120)], "fast", csrf="c",
                            stop_line=None, part_retries=3)
    assert 'accept=".mp4,.mov,video/mp4,video/quicktime"' in html
    assert 'value="fast" checked> fast, about 30 min' in html
    assert 'value="quality"> quality, about 120 min' in html
    assert 'data-part-retries="3"' in html and "<progress" in html


def test_a_job_without_a_splat_says_so_and_has_an_empty_viewer_slot() -> None:
    html = pages.result_page(view(result_file=None), csrf="c", poll_seconds=20, viewer=None,
                             download=True, share=None)
    assert "no splat to download" in html and "/download" not in html
    assert ('<section id="viewer-slot" aria-labelledby="viewer-label">'
            '<h2 id="viewer-label">3D viewer</h2></section>') in html


def test_the_static_files_are_small_and_the_favicon_is_an_icon() -> None:
    icon = (REPO_ROOT / "app" / "static" / "favicon.ico").read_bytes()
    assert icon[:4] == b"\x00\x00\x01\x00" and len(icon) < 1024
    script = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "X-CSRF-Token" in script and "eval(" not in script and "innerHTML" not in script
