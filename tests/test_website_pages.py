"""The pages: every value escaped, the viewport and stylesheet, the start page's presets, the
Viewer_Slot, the static files' sizes, and the redesign's layout: the split sign-in page, the header
and its phone menu, the cards, the result page's bar and folded details, and the shared page."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from app import pages
from app.statuses import JobView, StepTime
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
    assert ('value="fast" checked><span class="preset-name">fast</span>'
            '<span class="preset-time">about 30 min</span>') in html
    assert ('value="quality"><span class="preset-name">quality</span>'
            '<span class="preset-time">about 120 min</span>') in html
    assert html.count(" checked") == 1 and html.count('class="preset-card"') == 3
    assert 'data-part-retries="3"' in html and "<progress" in html


def test_a_job_without_a_splat_says_so_and_has_an_empty_viewer_slot() -> None:
    html = pages.result_page(view(result_file=None), csrf="c", poll_seconds=20, viewer=None,
                             download=True, share=None)
    assert "no splat to download" in html and "/download" not in html
    assert ('<section id="viewer-slot" aria-labelledby="viewer-label">'
            '<h2 id="viewer-label">3D viewer</h2></section>') in html


def test_the_copy_button_appears_only_with_a_new_share_link() -> None:
    new_link = "https://example.test/shared/new"
    for share in (None, pages.ShareState(on=False), pages.ShareState(on=True),
                  pages.ShareState(on=True, new_link=new_link)):
        html = pages.result_page(view(), csrf="c", poll_seconds=20, viewer=None, download=True,
                                 share=share)
        if share is None or share.new_link is None:
            assert 'id="copy-share-link"' not in html
        else:
            assert ('<label for="share-link">Room tour link</label><div class="share-link-row">'
                    f'<input id="share-link" readonly value="{new_link}">'
                    '<button id="copy-share-link" type="button">Copy link</button></div>') in html
            assert '<p id="share-copy-status" role="status"></p>' in html
            assert '<script src="/static/app.js" defer></script>' in html
            assert "<script>" not in html and "onclick=" not in html


def test_the_static_files_are_small_and_the_favicon_is_an_icon() -> None:
    icon = (REPO_ROOT / "app" / "static" / "favicon.ico").read_bytes()
    assert icon[:4] == b"\x00\x00\x01\x00" and len(icon) < 1024
    script = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "X-CSRF-Token" in script and "eval(" not in script and "innerHTML" not in script


RUNNING = dict(status="running", step=2, result_file=None, run_seconds=None)


def test_the_signin_page_is_split_into_the_story_and_the_card() -> None:
    html = pages.signin_page("tok", "Wrong name or password.")
    hero, card = html.index('class="auth-hero"'), html.index('class="auth-card"')
    assert hero < card
    assert "Film a room on your phone. Get a 3D tour anyone can walk through." in html
    assert [name for name in ("<strong>Film</strong>", "<strong>Upload</strong>",
                              "<strong>Share</strong>") if name in html] == [
        "<strong>Film</strong>", "<strong>Upload</strong>", "<strong>Share</strong>"]
    assert '<input type="hidden" name="csrf" value="tok">' in html
    assert 'action="/signin"' in html and "Wrong name or password." in html
    assert "site-nav" not in html and "/signout" not in html and "app.js" not in html


def test_the_header_has_the_nav_the_menu_button_and_marks_the_current_page() -> None:
    html = pages.job_list_page([], csrf="c")
    assert 'class="site-header"' in html
    assert ('<button type="button" class="menu-button" aria-expanded="false" '
            'aria-controls="site-nav">') in html
    assert '<nav id="site-nav" class="site-nav" aria-label="Main">' in html
    assert '<a href="/jobs" aria-current="page">Jobs</a>' in html
    assert '<a href="/start">Start</a>' in html and '<a href="/showcase">Showcase</a>' in html
    assert '<input type="hidden" name="csrf" value="c">' in html and 'action="/signout"' in html
    assert html.count('aria-current="page"') == 1
    assert '<a href="/showcase" aria-current="page">' in pages.showcase_page([], csrf="c")
    assert "menu-button" not in pages.not_found_page()      # signed out: the brand only


def test_the_job_list_is_cards_with_a_date_a_coloured_pill_and_a_button() -> None:
    html = pages.job_list_page([view(), view(job_id="20261202-080000-fedcba", **RUNNING)], csrf="c")
    assert html.count('<li class="card">') == 2
    assert '<span class="pill status-done">Done</span>' in html
    assert '<span class="pill status-running">Running (step 2 of 5)</span>' in html
    assert "1 Dec 2026, 12:00 UTC" in html and "2 Dec 2026, 08:00 UTC" in html
    assert 'href="/jobs/20261201-120000-abcdef">Open tour</a>' in html
    assert 'href="/jobs/20261202-080000-fedcba">View job</a>' in html
    assert "Your jobs" in html and 'href="/start"' in html


def test_the_showcase_is_cards_that_open_the_tour() -> None:
    html = pages.showcase_page([("20261201-120000-abcdef", "Old library", "Footage: CC BY, A. Person"),
                                ("20261203-093000-aaaaaa", "Kitchen", None)], csrf="c")
    assert html.count('<li class="card">') == 2 and 'class="cards showcase"' in html
    assert '<h2 class="card-title">Old library</h2>' in html
    assert '<p class="credit">Footage: CC BY, A. Person</p>' in html
    assert html.count('<span class="pill status-done">Ready</span>') == 2
    assert html.count(">Open tour</a>") == 2 and "3 Dec 2026, 09:30 UTC" in html


def test_a_card_date_comes_from_the_start_time_or_the_job_id() -> None:
    from datetime import datetime, timezone
    started = datetime(2026, 7, 4, 9, 5, tzinfo=timezone.utc)
    assert pages.job_date("20261201-120000-abcdef", started) == "4 Jul 2026, 09:05 UTC"
    assert pages.job_date("20261201-120000-abcdef") == "1 Dec 2026, 12:00 UTC"
    assert pages.job_date("not-a-job-id") == ""


def test_the_result_page_puts_the_bar_above_the_viewer_and_folds_the_details() -> None:
    html = pages.result_page(view(step_times=(StepTime("frames", "done", 12.0),)), csrf="c",
                             poll_seconds=20, viewer=pages.Viewer("https://s/x.sog", None),
                             download=True, share=pages.ShareState(on=False))
    bar, viewer = html.index('class="result-bar"'), html.index('id="viewer-slot"')
    details = html.index('<details class="details"><summary>Details</summary>')
    assert bar < html.index('<span id="status" class="pill status-done">Done</span>') < viewer
    assert bar < html.index('id="share"') < viewer
    assert bar < html.index('href="/jobs/20261201-120000-abcdef/download"') < viewer
    assert viewer < details < html.index('<ul class="steps">') and "1 min 1 s" in html[details:]
    assert html.count("<h1") == 1 and 'class="page-result"' in html
    assert '<details class="share-menu">' in html          # closed until there's a new link
    fresh = pages.result_page(view(), csrf="c", poll_seconds=20, viewer=None, download=True,
                              share=pages.ShareState(on=True, new_link="https://x/shared/t"))
    assert '<details class="share-menu" open>' in fresh and 'id="copy-share-link"' in fresh
    opened = fresh.index('<details class="share-menu" open>')
    assert opened < fresh.index('id="copy-share-link"') < fresh.index('id="share-copy-status"') \
        < fresh.index("</details>", opened)


def test_a_running_result_page_polls_and_its_pill_can_change() -> None:
    html = pages.result_page(view(**RUNNING), csrf="c", poll_seconds=20, viewer=None,
                             download=True, share=None)
    assert 'data-job-id="20261201-120000-abcdef" data-poll-seconds="20"' in html
    assert '<span id="status" class="pill status-running">Running (step 2 of 5)</span>' in html
    script = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "`pill status-${status.status}`" in script


def test_the_shared_page_is_only_the_viewer_and_a_small_label() -> None:
    html = pages.result_page(view(), csrf=None, poll_seconds=20,
                             viewer=pages.Viewer("https://s/x.sog", None, "/shared/t/viewer"),
                             download=False, share=None, shared=True)
    assert 'class="page-shared"' in html and 'data-splat-url="https://s/x.sog"' in html
    assert '<p class="made-with">' in html and "Made with Voxelis</p>" in html
    for absent in ("site-header", "result-bar", "Details", 'id="status"', "/download", "csrf",
                   "app.js", "data-job-id"):
        assert absent not in html, absent
    assert "<title>Room tour · Voxelis</title>" in html and html.count("<h1") == 1
    assert ('<div class="shared-label"><h1>Room tour</h1><p class="made-with">'
            in html[:html.index('id="viewer-slot"')])


def test_the_look_up_warning_sits_under_the_owners_bar_and_not_on_a_shared_page() -> None:
    warning = "The video never looked up. Film a second loop tilted up."
    owner = pages.result_page(view(capture_warning=warning), csrf="c", poll_seconds=20,
                              viewer=pages.Viewer("https://s/x.sog", None), download=True,
                              share=pages.ShareState(on=False))
    line = owner.index(f'<p class="warning notice">{warning}</p>')
    assert owner.index('class="result-bar"') < line < owner.index('id="viewer-slot"')
    shared = pages.result_page(view(capture_warning=warning), csrf=None, poll_seconds=20,
                               viewer=pages.Viewer("https://s/x.sog", None, "/shared/t/viewer"),
                               download=False, share=None, shared=True)
    assert warning not in shared and "warning" not in shared


def test_the_start_page_has_a_drop_zone_and_a_progress_bar_with_a_percentage() -> None:
    html = pages.start_page([("fast", 30)], "fast", csrf="c", stop_line=None, part_retries=3)
    zone = html.index('<div class="dropzone" id="dropzone">')
    assert zone < html.index('<input id="video" name="video" type="file"') < html.index("</div>", zone)
    assert '<label for="video" class="dropzone-title">' in html and 'id="video-name"' in html
    assert '<div class="upload-progress" id="upload-progress" hidden>' in html
    assert '<span id="progress-percent">0%</span>' in html
    script = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "progress-percent" in script and "upload-progress" in script and "dropzone" in script


def test_the_stylesheet_sets_the_look_and_the_breakpoints() -> None:
    css = (REPO_ROOT / "app" / "static" / "style.css").read_text(encoding="utf-8")
    for wanted in ("--bg: #0f0d14;", "--accent: #f58a1f;", "--radius: 12px;", "system-ui",
                   "position: sticky;", "@media (max-width: 599px)", "height: 72dvh;",
                   "height: 100dvh;", ".js .site-nav.open", "@media (prefers-reduced-motion"):
        assert wanted in css, wanted
    assert "@import" not in css and "url(" not in css      # nothing from outside 'self'
