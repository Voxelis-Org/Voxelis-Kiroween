"""Every HTML page, as plain `str` templates. A value enters a page only through `esc`, in text and
in attributes alike, so titles, credits, file names, Account names and messages can't add markup.
No page has an inline script or style."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.statuses import DONE, FINAL, JobView, status_text

_SECONDS_PER_MINUTE = 60


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
           page_script: bool = True) -> str:
    head = ['<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
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
    nav = _nav(csrf) if signed_in and csrf is not None else ""
    attributes = "".join(f' data-{esc(name)}="{esc(value)}"' for name, value in (data or {}).items())
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n" + "\n".join(head) + "\n</head>\n<body>\n"
            f"<header><a class=\"brand\" href=\"/\">Voxelis</a>{nav}</header>\n"
            f"<main{attributes}>\n<h1>{esc(title)}</h1>\n{body}\n</main>\n</body>\n</html>\n")


def _nav(csrf: str) -> str:
    return ('<nav><a href="/jobs">Jobs</a> <a href="/start">Start</a> '
            '<a href="/showcase">Showcase</a>'
            '<form method="post" action="/signout" class="inline">'
            f'<input type="hidden" name="csrf" value="{esc(csrf)}">'
            '<button type="submit" class="link">Sign out</button></form></nav>')


def message(text: str | None, *, kind: str = "error") -> str:
    if not text:
        return '<p id="message" role="status"></p>'
    return f'<p id="message" role="status" class="{esc(kind)}">{esc(text)}</p>'


def signin_page(token: str, error: str | None = None) -> str:
    body = (message(error) +
            '<form method="post" action="/signin">'
            f'<input type="hidden" name="csrf" value="{esc(token)}">'
            '<label for="name">Account name</label>'
            '<input id="name" name="name" autocomplete="username" autocapitalize="none" required>'
            '<label for="password">Password</label>'
            '<input id="password" name="password" type="password" '
            'autocomplete="current-password" required>'
            '<button type="submit">Sign in</button></form>')
    return layout("Sign in", body, csrf=None, signed_in=False)


def when(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else "not yet"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "not yet"
    minutes, rest = divmod(int(round(seconds)), _SECONDS_PER_MINUTE)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def job_list_page(views: Sequence[JobView], *, csrf: str) -> str:
    if not views:
        rows = '<p>No jobs yet. <a href="/start">Start one</a>.</p>'
    else:
        items = "".join(
            f'<li><a href="/jobs/{esc(v.job_id)}">{esc(v.job_id)}</a> '
            f'<span class="preset">{esc(v.preset)}</span> '
            f'<span class="status">{esc(status_text(v))}</span></li>' for v in views)
        rows = f'<ul class="jobs">{items}</ul>'
    return layout("Your jobs", rows, csrf=csrf, signed_in=True)


def start_page(presets: Sequence[tuple[str, float]], default: str, *, csrf: str,
               stop_line: str | None, part_retries: int) -> str:
    if stop_line is not None:
        return layout("Start a job", f'<p class="stopped">{esc(stop_line)}</p>', csrf=csrf,
                      signed_in=True)
    choices = "".join(
        f'<label class="preset"><input type="radio" name="preset" value="{esc(name)}"'
        f'{" checked" if name == default else ""}> {esc(name)}, about {minutes:g} min</label>'
        for name, minutes in presets)
    body = ('<p>Pick a video of a place, filmed slowly while you walk around it, then press Start. '
            'The video goes straight to storage, never through this page.</p>'
            '<form id="start-form">'
            '<label for="video">Video (MP4 or MOV)</label>'
            '<input id="video" name="video" type="file" '
            'accept=".mp4,.mov,video/mp4,video/quicktime" required>'
            f'<fieldset><legend>Preset</legend>{choices}</fieldset>'
            '<button id="start" type="submit">Start</button>'
            '<progress id="progress" max="1" value="0" hidden></progress>'
            f'{message(None)}</form>')
    return layout("Start a job", body, csrf=csrf, signed_in=True,
                  data={"part-retries": part_retries})


def result_page(view: JobView, *, csrf: str | None, poll_seconds: int, viewer: Viewer | None,
                download: bool, share: ShareState | None, showcase_line: str | None = None,
                title: str | None = None, shared: bool = False) -> str:
    parts = [f'<p>Status: <strong id="status">{esc(status_text(view))}</strong></p>',
             f"<p>Preset: {esc(view.preset or 'unknown')}. Started: {esc(when(view.started_at))}."
             f" Run time: {esc(duration(view.run_seconds))}.</p>"]
    if showcase_line:
        parts.append(f'<p class="credit">{esc(showcase_line)}</p>')
    if view.readable_error:
        parts.append(f'<p class="error">{esc(view.readable_error)}</p>')
    for step, error in view.warnings:
        parts.append(f'<p class="warning">The {esc(step)} step failed: {esc(error)}</p>')
    if view.step_times:
        rows = "".join(f"<li>{esc(t.step)}: {esc(t.outcome)}, {esc(duration(t.seconds))}</li>"
                       for t in view.step_times)
        parts.append(f'<h2>Steps</h2><ul class="steps">{rows}</ul>')
    if view.status in DONE and view.result_file is None:
        parts.append("<p>This job has no splat to download.</p>")
    if download and view.result_file:
        parts.append(f'<p><a class="button" href="/jobs/{esc(view.job_id)}/download">Download '
                     f"{esc(view.result_file)}</a></p>")
    if shared:
        parts.insert(0, _viewer_slot(viewer))
    else:
        parts.append(_viewer_slot(viewer))
    if share is not None:
        parts.append(_share_controls(view.job_id, share, csrf or ""))
    data: dict[str, object] = {}
    if view.status not in FINAL and not shared:
        data = {"job-id": view.job_id, "poll-seconds": poll_seconds}
    heading = "Room tour" if shared else title or f"Job {view.job_id}"
    return layout(heading, "\n".join(parts), csrf=csrf, signed_in=not shared, data=data,
                  viewer_base=viewer.base if viewer else None)


def _viewer_slot(viewer: Viewer | None) -> str:
    inner = ""
    if viewer is not None:
        cameras = (f' data-cameras-url="{esc(viewer.cameras_url)}"' if viewer.cameras_url else "")
        inner = f'<div class="splat-viewer" data-splat-url="{esc(viewer.splat_url)}"{cameras}></div>'
    return ('<section id="viewer-slot" aria-labelledby="viewer-label">'
            f'<h2 id="viewer-label">3D viewer</h2>{inner}</section>')


def _share_controls(job_id: str, share: ShareState, csrf: str) -> str:
    hidden = f'<input type="hidden" name="csrf" value="{esc(csrf)}">'
    parts = ['<section id="share"><h2>Share</h2>']
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
                 f'<button type="submit">{label}</button></form>')
    if share.on:
        parts.append(f'<form method="post" action="/jobs/{esc(job_id)}/share/off">{hidden}'
                     '<button type="submit" class="secondary">Turn the link off</button></form>')
    parts.append("</section>")
    return "".join(parts)


def showcase_page(entries: Sequence[tuple[str, str, str | None]], *, csrf: str) -> str:
    if not entries:
        return layout("Showcase", "<p>Nothing in the showcase yet.</p>", csrf=csrf, signed_in=True)
    items = "".join(
        f'<li><a href="/jobs/{esc(job_id)}">{esc(title)}</a>'
        + (f'<br><span class="credit">{esc(line)}</span>' if line else "") + "</li>"
        for job_id, title, line in entries)
    return layout("Showcase", f'<ul class="showcase">{items}</ul>', csrf=csrf, signed_in=True)


def not_found_page() -> str:
    return layout("Not found", "<p>There's nothing here. Check the link.</p>", csrf=None,
                  signed_in=False)


def error_page() -> str:
    return layout("Something went wrong", "<p>Something went wrong on our side. Try again in a "
                  "minute.</p>", csrf=None, signed_in=False)


def message_page(title: str, text: str, *, csrf: str | None) -> str:
    return layout(title, message(text), csrf=csrf, signed_in=csrf is not None)
