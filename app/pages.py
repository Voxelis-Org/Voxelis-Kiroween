"""Every HTML page, as plain `str` templates. A value enters a page only through `esc`, in text and
in attributes alike, so titles, credits, file names, Account names and messages can't add markup.
No page has an inline script or style."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from app.statuses import DONE, FINAL, JobView, status_text

_SECONDS_PER_MINUTE = 60
_ID_TIME = "%Y%m%d-%H%M%S"           # a Job ID starts with its UTC start time
_NAV = (("/jobs", "Jobs"), ("/start", "Start"), ("/showcase", "Showcase"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


@dataclass(frozen=True)
class Viewer:
    """The Viewer_Slot's two links, and where its files are served from."""

    splat_url: str
    cameras_url: str | None
    base: str = "/viewer"


@dataclass(frozen=True)
class ShareState:
    """The owner's share controls on a Result_Page."""

    on: bool
    new_link: str | None = None


def layout(title: str, body: str, *, csrf: str | None, signed_in: bool,
           data: Mapping[str, object] | None = None, viewer_base: str | None = None,
           page_script: bool = True, page: str = "page", heading: bool = True,
           header: bool = True, current: str | None = None) -> str:
    """The page around `body`. `page` names the body's class (page-<page>); `heading` adds the
    title as the page's h1 (pages that place their own h1 pass False); `current` marks a nav link."""
    head = ['<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="color-scheme" content="dark">',
            '<meta name="theme-color" content="#0f0d14">',
            f"<title>{esc(title)} · Voxelis</title>",
            '<link rel="icon" href="/favicon.ico">',
            '<link rel="stylesheet" href="/static/style.css">']
    if viewer_base:
        head.append(f'<link rel="stylesheet" href="{esc(viewer_base)}/viewer.css">')
    if signed_in and csrf is not None:
        head.append(f'<meta name="csrf-token" content="{esc(csrf)}">')
    if signed_in and page_script:
        head.append('<script src="/static/app.js" defer></script>')
    if viewer_base:
        head.append(f'<script type="module" src="{esc(viewer_base)}/embed.js"></script>')
    top = _header(csrf if signed_in else None, current) if header else ""
    attributes = "".join(f' data-{esc(name)}="{esc(value)}"' for name, value in (data or {}).items())
    h1 = f'<h1 class="page-title">{esc(title)}</h1>\n' if heading else ""
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n" + "\n".join(head) + "\n</head>\n"
            f'<body class="page-{esc(page)}">\n{top}'
            f"<main{attributes}>\n{h1}{body}\n</main>\n</body>\n</html>\n")


def _header(csrf: str | None, current: str | None) -> str:
    brand = '<a class="brand" href="/"><span class="brand-mark" aria-hidden="true"></span>Voxelis</a>'
    menu = ""
    if csrf is not None:
        menu = ('<button type="button" class="menu-button" aria-expanded="false" '
                'aria-controls="site-nav"><span class="menu-icon" aria-hidden="true"></span>'
                '<span class="menu-label">Menu</span></button>' + _nav(csrf, current))
    return f'<header class="site-header"><div class="header-inner">{brand}{menu}</div></header>\n'


def _nav(csrf: str, current: str | None = None) -> str:
    mark = ' aria-current="page"'
    links = "".join(f'<a href="{href}"{mark if href == current else ""}>{label}</a>'
                    for href, label in _NAV)
    return (f'<nav id="site-nav" class="site-nav" aria-label="Main">{links}'
            '<form method="post" action="/signout" class="inline">'
            f'<input type="hidden" name="csrf" value="{esc(csrf)}">'
            '<button type="submit" class="link">Sign out</button></form></nav>')


def message(text: str | None, *, kind: str = "error") -> str:
    if not text:
        return '<p id="message" role="status"></p>'
    return f'<p id="message" role="status" class="{esc(kind)}">{esc(text)}</p>'


_STEPS = (("Film", "Walk slowly around the room with your phone, about a minute is plenty."),
          ("Upload", "Drop the video in. Voxelis builds the 3D tour in the cloud."),
          ("Share", "Send one link. Anyone can walk through it, no app and no sign-in."))


def signin_page(token: str, error: str | None = None) -> str:
    steps = "".join(f'<li><span class="step-number" aria-hidden="true">{n}</span>'
                    f'<span><strong>{name}</strong> {text}</span></li>'
                    for n, (name, text) in enumerate(_STEPS, start=1))
    hero = ('<section class="auth-hero" aria-label="About Voxelis">'
            '<p class="auth-brand"><span class="brand-mark" aria-hidden="true"></span>Voxelis</p>'
            '<p class="auth-tagline">Film a room on your phone. '
            'Get a 3D tour anyone can walk through.</p>'
            f'<ol class="auth-steps">{steps}</ol>'
            '<div class="auth-floor" aria-hidden="true"></div></section>')
    card = ('<section class="auth-card"><h1>Sign in</h1>'
            '<p class="muted">Use the account name and password you were given.</p>'
            + message(error) +
            '<form method="post" action="/signin" class="stack">'
            f'<input type="hidden" name="csrf" value="{esc(token)}">'
            '<label for="name">Account name</label>'
            '<input id="name" name="name" autocomplete="username" autocapitalize="none" required>'
            '<label for="password">Password</label>'
            '<input id="password" name="password" type="password" '
            'autocomplete="current-password" required>'
            '<button type="submit" class="button-wide">Sign in</button></form></section>')
    return layout("Sign in", f'<div class="auth">{hero}{card}</div>', csrf=None, signed_in=False,
                  page="signin", heading=False, header=False)


def when(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else "not yet"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "not yet"
    minutes, rest = divmod(int(round(seconds)), _SECONDS_PER_MINUTE)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def job_date(job_id: str, started_at: datetime | None = None) -> str:
    """A short date for a card: the start time, else the time a Job ID starts with, else ""."""
    value = started_at
    if value is None:
        try:
            value = datetime.strptime(job_id[:15], _ID_TIME).replace(tzinfo=timezone.utc)
        except ValueError:
            return ""
    return f"{value.day} {value:%b %Y, %H:%M} UTC"


def _pill(status: str, text: str, *, element_id: str | None = None) -> str:
    ident = f' id="{esc(element_id)}"' if element_id else ""
    return f'<span{ident} class="pill status-{esc(status)}">{esc(text)}</span>'


def _card(job_id: str, title: str, date: str, pill: str, extra: str, action: str,
          *, primary: bool = True) -> str:
    kind = "button card-action" if primary else "button secondary card-action"
    return ('<li class="card"><div class="card-thumb" aria-hidden="true">'
            '<span class="card-cube"></span></div><div class="card-body">'
            f'<div class="card-top">{pill}</div>'
            f'<h2 class="card-title">{esc(title)}</h2>'
            f'<p class="card-date">{esc(date) if date else "&nbsp;"}</p>{extra}'
            f'<a class="{kind}" href="/jobs/{esc(job_id)}">{action}</a></div></li>')


def job_list_page(views: Sequence[JobView], *, csrf: str) -> str:
    if not views:
        body = ('<div class="empty"><span class="empty-mark" aria-hidden="true"></span>'
                '<p>No jobs yet. Film a room and turn it into a tour.</p>'
                '<a class="button" href="/start">Start one</a></div>')
    else:
        cards = "".join(
            _card(v.job_id, v.job_id, job_date(v.job_id, v.started_at),
                  _pill(v.status, status_text(v)),
                  f'<p class="card-meta">{esc(v.preset)} preset</p>' if v.preset else "",
                  "Open tour" if v.status in DONE else "View job", primary=v.status in DONE)
            for v in views)
        body = f'<ul class="cards jobs">{cards}</ul>'
    intro = ('<div class="page-head-row"><p class="lede">Every tour you have started, newest first.'
             '</p><a class="button" href="/start">New tour</a></div>')
    return layout("Your jobs", intro + body, csrf=csrf, signed_in=True, current="/jobs")


def start_page(presets: Sequence[tuple[str, float]], default: str, *, csrf: str,
               stop_line: str | None, part_retries: int) -> str:
    if stop_line is not None:
        return layout("Start a job", f'<p class="stopped notice">{esc(stop_line)}</p>', csrf=csrf,
                      signed_in=True, current="/start")
    choices = "".join(
        f'<label class="preset-card"><input type="radio" name="preset" value="{esc(name)}"'
        f'{" checked" if name == default else ""}>'
        f'<span class="preset-name">{esc(name)}</span>'
        f'<span class="preset-time">about {minutes:g} min</span></label>'
        for name, minutes in presets)
    body = ('<p class="lede">Pick a video of a place, filmed slowly while you walk around it, then '
            'press Start. The video goes straight to storage, never through this page.</p>'
            '<form id="start-form" class="start-form">'
            '<div class="dropzone" id="dropzone">'
            '<input id="video" name="video" type="file" '
            'accept=".mp4,.mov,video/mp4,video/quicktime" required>'
            '<span class="dropzone-icon" aria-hidden="true"></span>'
            '<label for="video" class="dropzone-title">Drop your video here</label>'
            '<span class="dropzone-hint">or tap to choose a file. MP4 or MOV.</span>'
            '<span class="dropzone-file" id="video-name" aria-live="polite"></span></div>'
            '<fieldset class="presets"><legend>Quality</legend>'
            f'<div class="preset-grid">{choices}</div></fieldset>'
            '<div class="upload-progress" id="upload-progress" hidden>'
            '<div class="upload-progress-row"><span>Uploading</span>'
            '<span id="progress-percent">0%</span></div>'
            '<progress id="progress" max="1" value="0"></progress></div>'
            f'{message(None)}'
            '<button id="start" type="submit" class="button-wide">Start</button></form>')
    return layout("Start a job", body, csrf=csrf, signed_in=True, page="start",
                  data={"part-retries": part_retries}, current="/start")


def result_page(view: JobView, *, csrf: str | None, poll_seconds: int, viewer: Viewer | None,
                download: bool, share: ShareState | None, showcase_line: str | None = None,
                title: str | None = None, shared: bool = False) -> str:
    viewer_base = viewer.base if viewer else None
    if shared:
        body = ('<div class="shared-label"><h1>Room tour</h1>'
                '<p class="made-with"><span class="brand-mark" aria-hidden="true"></span>'
                f'Made with Voxelis</p></div>{_viewer_slot(viewer)}')
        return layout("Room tour", body, csrf=None, signed_in=False, viewer_base=viewer_base,
                      page="shared", heading=False, header=False)
    heading = title or f"Job {view.job_id}"
    actions = []
    if share is not None:
        actions.append(_share_controls(view.job_id, share, csrf or ""))
    if download and view.result_file:
        actions.append(f'<a class="button" href="/jobs/{esc(view.job_id)}/download" '
                       f'title="{esc(view.result_file)}">Download</a>')
    bar = ['<div class="result-bar"><div class="result-heading">'
           f'<h1>{esc(heading)}</h1>{_pill(view.status, status_text(view), element_id="status")}'
           '</div>']
    if actions:
        bar.append(f'<div class="result-actions">{"".join(actions)}</div>')
    bar.append("</div>")
    notes = []
    if showcase_line:
        notes.append(f'<p class="credit">{esc(showcase_line)}</p>')
    if view.readable_error:
        notes.append(f'<p class="error notice">{esc(view.readable_error)}</p>')
    for step, error in view.warnings:
        notes.append(f'<p class="warning notice">The {esc(step)} step failed: {esc(error)}</p>')
    if view.capture_warning:
        notes.append(f'<p class="warning notice">{esc(view.capture_warning)}</p>')
    if view.status in DONE and view.result_file is None:
        notes.append('<p class="notice">This job has no splat to download.</p>')
    facts = ('<dl class="facts">'
             f"<div><dt>Preset</dt><dd>{esc(view.preset or 'unknown')}</dd></div>"
             f"<div><dt>Started</dt><dd>{esc(when(view.started_at))}</dd></div>"
             f"<div><dt>Run time</dt><dd>{esc(duration(view.run_seconds))}</dd></div>"
             f"<div><dt>Job</dt><dd>{esc(view.job_id)}</dd></div></dl>")
    steps = ""
    if view.step_times:
        rows = "".join(f'<li><span class="step-name">{esc(t.step)}</span>'
                       f'{_pill(t.outcome, t.outcome)}'
                       f'<span class="step-time">{esc(duration(t.seconds))}</span></li>'
                       for t in view.step_times)
        steps = f'<h2>Steps</h2><ul class="steps">{rows}</ul>'
    details = f'<details class="details"><summary>Details</summary>{facts}{steps}</details>'
    parts = ["".join(bar), *notes, _viewer_slot(viewer), details]
    data: dict[str, object] = {}
    if view.status not in FINAL:
        data = {"job-id": view.job_id, "poll-seconds": poll_seconds}
    return layout(heading, "\n".join(parts), csrf=csrf, signed_in=True, data=data,
                  viewer_base=viewer_base, page="result", heading=False)


def _viewer_slot(viewer: Viewer | None) -> str:
    inner = ""
    if viewer is not None:
        cameras = (f' data-cameras-url="{esc(viewer.cameras_url)}"' if viewer.cameras_url else "")
        inner = f'<div class="splat-viewer" data-splat-url="{esc(viewer.splat_url)}"{cameras}></div>'
    return ('<section id="viewer-slot" aria-labelledby="viewer-label">'
            f'<h2 id="viewer-label">3D viewer</h2>{inner}</section>')


def _share_controls(job_id: str, share: ShareState, csrf: str) -> str:
    hidden = f'<input type="hidden" name="csrf" value="{esc(csrf)}">'
    opened = " open" if share.new_link else ""
    parts = [f'<details class="share-menu"{opened}><summary class="button secondary">Share</summary>'
             '<section id="share"><h2>Share</h2>']
    if share.new_link:
        parts.append('<p>Anyone with this link can open the room tour without signing in. '
                     'Copy it now: it\'s shown only once. Making a new link turns this one off.</p>'
                     '<label for="share-link">Room tour link</label><div class="share-link-row">'
                     f'<input id="share-link" readonly value="{esc(share.new_link)}">'
                     '<button id="copy-share-link" type="button">Copy link</button></div>'
                     '<p id="share-copy-status" role="status"></p>')
    elif share.on:
        parts.append("<p>A share link is on. Anyone with it sees this job's result and 3D viewer.</p>")
    else:
        parts.append("<p>Make a secret link that shows this job's result and 3D viewer to anyone "
                     "who has it, without signing in.</p>")
    label = "Make a new share link" if share.on else "Make a share link"
    parts.append(f'<form method="post" action="/jobs/{esc(job_id)}/share">{hidden}'
                 f'<button type="submit" class="button-wide">{label}</button></form>')
    if share.on:
        parts.append(f'<form method="post" action="/jobs/{esc(job_id)}/share/off">{hidden}'
                     '<button type="submit" class="secondary button-wide">Turn the link off</button>'
                     '</form>')
    parts.append("</section></details>")
    return "".join(parts)


def showcase_page(entries: Sequence[tuple[str, str, str | None]], *, csrf: str) -> str:
    intro = '<p class="lede">Tours picked to show what Voxelis can do. Open one and walk around.</p>'
    if not entries:
        return layout("Showcase", intro + '<div class="empty"><span class="empty-mark" '
                      'aria-hidden="true"></span><p>Nothing in the showcase yet.</p></div>',
                      csrf=csrf, signed_in=True, current="/showcase")
    cards = "".join(
        _card(job_id, title, job_date(job_id), _pill("done", "Ready"),
              f'<p class="credit">{esc(line)}</p>' if line else "", "Open tour")
        for job_id, title, line in entries)
    return layout("Showcase", f'{intro}<ul class="cards showcase">{cards}</ul>', csrf=csrf,
                  signed_in=True, current="/showcase")


def not_found_page() -> str:
    return layout("Not found", "<p>There's nothing here. Check the link.</p>", csrf=None,
                  signed_in=False, page="message")


def error_page() -> str:
    return layout("Something went wrong", "<p>Something went wrong on our side. Try again in a "
                  "minute.</p>", csrf=None, signed_in=False, page="message")


def message_page(title: str, text: str, *, csrf: str | None) -> str:
    return layout(title, message(text), csrf=csrf, signed_in=csrf is not None, page="message")
