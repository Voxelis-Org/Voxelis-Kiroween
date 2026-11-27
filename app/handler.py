"""`handle(event, deps, now)`: the website's whole request handling, a pure function of a Lambda
payload-2.0 event, the `Deps` and the time.

In order: parse the event; route it; check the Session for every route but the open and shared
ones (a signed-out request gets a 303 to /signin, or a 401 under /api/, and nothing else); refuse a
body over `max_body_bytes`; check CSRF on every POST; run the route; turn a `PublicRefusal` into its
page or JSON and anything else into the 500 answer; and add the security headers to every answer.

A shared route runs with no Session: its share token is its key, and a wrong, malformed or turned-off
token gets the 404 page with no Stack lookup and no link.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import time
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from splatpipe.config import PRESETS
from splatpipe.errors import PipelineError
from splatpipe.jobs import check_job_id

from app import pages, sessions, shares, uploads, website_start
from app.errors import BAD_REQUEST, FORBIDDEN, NOT_FOUND, SERVER, TOO_LARGE, PublicRefusal, not_found
from app.limits import is_stopped, stop_line
from app.showcase import licence_line, list_entries, read_entry
from app.statuses import DONE, FINAL, JobView, job_view, list_jobs, may_see, out_prefix, status_text
from app.web import Deps, Request, Response
from cloud.aws import aws_setting, s3_origin
from cloud.redact import account_redaction
from cloud.stack import resolve_stack

HTML = "text/html; charset=utf-8"
JSON = "application/json"
SEE_OTHER = 303
OK = 200
UNAUTHORIZED = 401
_GZIP_FLOOR = 1024
_MS = 1000
_MINUTE = 60

# The files the website serves, by URL path, with their Content-Type.
STATIC_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".mjs": "text/javascript; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                ".ico": "image/x-icon"}
OPEN_STATIC = ("/static/style.css", "/favicon.ico")
VIEWER_FILES = ("viewer.css", "embed.js", "viewer.js", "params.js", "cameras.js", "controls.js",
                "tour.js", "haunt.js", "loading.js", "vendor/playcanvas.min.mjs",
                "vendor/PLAYCANVAS-LICENSE.txt")
_TEXT_PREFIXES = ("text/", "application/json")


@dataclass
class Ctx:
    request: Request
    deps: Deps
    now: datetime
    params: dict[str, str]
    session: sessions.Session | None = None
    viewer_page: bool = False


@dataclass(frozen=True)
class Route:
    method: str
    pattern: re.Pattern[str]
    kind: str                     # "open", "shared", "page" or "script"
    name: str
    run: Callable[[Ctx], Response]


def handle(event: Mapping[str, Any], deps: Deps, now: datetime) -> dict[str, Any]:
    started = time.perf_counter()
    request = parse_event(event)
    route, params = find_route(request.method, request.path)
    kind = route.kind if route else ("script" if request.path.startswith("/api/") else "page")
    name = route.name if route else "unknown"
    ctx = Ctx(request=request, deps=deps, now=now, params=params)
    response = _answer(ctx, route, kind, event)
    result = to_lambda(response, ctx, kind)
    deps.log.info("%s %s %d %dms", request.method, name, response.status,
                  int((time.perf_counter() - started) * _MS))
    return result


def _answer(ctx: Ctx, route: Route | None, kind: str, event: Mapping[str, Any]) -> Response:
    deps = ctx.deps
    try:
        if kind in ("page", "script"):
            ctx.session = sessions.current(ctx.request, deps, ctx.now)
            if ctx.session is None:
                return _signed_out(kind)
            _try_stack(deps)
        if kind != "shared" and body_size(event) > deps.website.max_body_bytes:
            raise PublicRefusal(TOO_LARGE, "too_large", "That request is too large.")
        if route is None:
            raise not_found()
        if ctx.request.method == "POST":
            check_csrf(ctx, kind, route)
        return route.run(ctx)
    except PublicRefusal as refusal:
        return refusal_response(ctx, kind, refusal, route)
    except Exception as error:   # anything unforeseen: the 500, with no detail
        deps.log.error("error: %s: %s", type(error).__name__, account_redaction(str(error)))
        if kind == "script":
            return json_response(SERVER, {"error": "server", "message": "Something went wrong on "
                                          "our side. Try again in a minute."})
        return Response(SERVER, HTML, pages.error_page().encode("utf-8"))


def _signed_out(kind: str) -> Response:
    if kind == "script":
        return json_response(UNAUTHORIZED, {"error": "signed_out"})
    return sessions.redirect("/signin")


# Parsing

def parse_event(event: Mapping[str, Any]) -> Request:
    context = event.get("requestContext") or {}
    http = context.get("http") or {}
    headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
    cookies: dict[str, str] = {}
    for raw in event.get("cookies") or []:
        cookie_name, sep, value = str(raw).partition("=")
        if sep and cookie_name.strip() not in cookies:
            cookies[cookie_name.strip()] = value.strip()
    body_text = event.get("body") or ""
    try:
        body = (base64.b64decode(body_text) if event.get("isBase64Encoded")
                else str(body_text).encode("utf-8"))
    except (ValueError, TypeError):
        body = b""
    return Request(method=str(http.get("method", "GET")).upper(),
                   path=str(event.get("rawPath") or http.get("path") or "/"),
                   query=dict(parse_qsl(str(event.get("rawQueryString") or ""))),
                   headers=headers, cookies=cookies, body=body,
                   source_ip=str(http.get("sourceIp", "")), host=str(context.get("domainName", "")))


def body_size(event: Mapping[str, Any]) -> int:
    """The body's size from the event's text, before any decoding."""
    text = str(event.get("body") or "")
    if event.get("isBase64Encoded"):
        return len(text) * 3 // 4
    return len(text.encode("utf-8", "replace"))


def find_route(method: str, path: str) -> tuple[Route | None, dict[str, str]]:
    for route in ROUTES:
        match = route.pattern.fullmatch(path)
        if match and route.method == method:
            return route, match.groupdict()
    return None, {}


# Checks

def check_csrf(ctx: Ctx, kind: str, route: Route) -> None:
    request = ctx.request
    origin = request.headers.get("origin")
    site = request.headers.get("sec-fetch-site")
    refusal = PublicRefusal(FORBIDDEN, "csrf", sessions.CSRF_MESSAGE)
    if origin is not None and origin != f"https://{request.host}":
        raise refusal
    if site is not None and site not in ("same-origin", "none"):
        raise refusal
    if kind == "open":
        return       # the sign-in form checks its own double-submit token
    assert ctx.session is not None
    given = (request.headers.get("x-csrf-token") if kind == "script"
             else sessions.form(request).get("csrf"))
    if not sessions.same_token(given, ctx.session.csrf):
        raise refusal


def stack_for(deps: Deps) -> Any:
    """The Stack's outputs, resolved once per container, with the bucket set on the client."""
    if deps.stack is None:
        deps.stack = resolve_stack(deps.aws, aws_setting(deps.config, "stack_name"))
    deps.aws.bucket = deps.stack.bucket
    return deps.stack


def _try_stack(deps: Deps) -> None:
    try:
        stack_for(deps)
    except Exception as error:   # the page still answers; the routes that need S3 raise again
        deps.log.warning("stack: %s", type(error).__name__)


# Answers

def json_response(status: int, data: Any) -> Response:
    return Response(status, JSON, json.dumps(data).encode("utf-8"))


def html_response(text: str, status: int = OK) -> Response:
    return Response(status, HTML, text.encode("utf-8"))


def refusal_response(ctx: Ctx, kind: str, refusal: PublicRefusal, route: Route | None) -> Response:
    text = account_redaction(refusal.message)
    if kind == "script":
        return json_response(refusal.status, {"error": refusal.kind, "message": text})
    if refusal.status == NOT_FOUND:
        return html_response(pages.not_found_page(), NOT_FOUND)
    if route is not None and route.name == "signin_post":
        return _signin_form(ctx, text, refusal.status)
    csrf = ctx.session.csrf if ctx.session else None
    return html_response(pages.message_page("That didn't work", text, csrf=csrf), refusal.status)


def csp(deps: Deps, *, viewer: bool, signed_in: bool) -> str:
    origin = ""
    if signed_in and deps.stack is not None:
        origin = " " + s3_origin(deps.stack.bucket, deps.aws.region)
    images = "'self' blob:" if viewer else "'self'"
    workers = "; worker-src blob:" if viewer else ""
    return (f"default-src 'none'; script-src 'self'; style-src 'self'; img-src {images}{workers}; "
            f"connect-src 'self'{origin}; form-action 'self'; frame-ancestors 'none'; "
            "base-uri 'none'")


def to_lambda(response: Response, ctx: Ctx, kind: str) -> dict[str, Any]:
    headers = {"Content-Type": response.content_type, **response.headers}
    signed_in = ctx.session is not None or (kind == "shared" and ctx.viewer_page)
    headers["Content-Security-Policy"] = csp(ctx.deps, viewer=ctx.viewer_page, signed_in=signed_in)
    headers["X-Content-Type-Options"] = "nosniff"
    headers["Referrer-Policy"] = "no-referrer"
    headers.setdefault("Cache-Control", "no-store")
    body = response.body
    encoded = not response.content_type.startswith(_TEXT_PREFIXES)
    if headers.get("Content-Encoding") == "gzip":
        encoded = True
    result: dict[str, Any] = {"statusCode": response.status, "headers": headers,
                              "isBase64Encoded": encoded,
                              "body": base64.b64encode(body).decode("ascii") if encoded
                              else body.decode("utf-8")}
    if response.cookies:
        result["cookies"] = list(response.cookies)
    return result


# Static files

def static_response(ctx: Ctx, key: str, *, cache: str) -> Response:
    body = ctx.deps.static.get(key)
    if body is None:
        raise not_found()
    suffix = key[key.rfind("."):]
    response = Response(OK, STATIC_TYPES.get(suffix, "application/octet-stream"), body,
                        headers={"Cache-Control": cache})
    if "gzip" in ctx.request.headers.get("accept-encoding", "") and len(body) > _GZIP_FLOOR:
        response.body = _gzipped(key, body)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
    return response


_GZIPPED: dict[tuple[str, int, int], bytes] = {}


def _gzipped(key: str, body: bytes) -> bytes:
    cache_key = (key, len(body), zlib.crc32(body))
    if cache_key not in _GZIPPED:
        _GZIPPED[cache_key] = gzip.compress(body, mtime=0)
    return _GZIPPED[cache_key]


def open_static(ctx: Ctx) -> Response:
    return static_response(ctx, ctx.request.path,
                           cache=f"public, max-age={ctx.deps.website.static_cache_seconds}")


def page_static(ctx: Ctx) -> Response:
    return static_response(ctx, ctx.request.path,
                           cache=f"private, max-age={ctx.deps.website.static_cache_seconds}")


def viewer_static(ctx: Ctx) -> Response:
    if ctx.params["file"] not in VIEWER_FILES:
        raise not_found()
    return static_response(ctx, f"/viewer/{ctx.params['file']}",
                           cache=f"private, max-age={ctx.deps.website.static_cache_seconds}")


# The sign-in routes

def _signin_form(ctx: Ctx, error: str | None, status: int = OK) -> Response:
    token = sessions.new_token(ctx.deps)
    response = html_response(pages.signin_page(token, error), status)
    response.cookies.append(sessions.set_cookie(sessions.SIGNIN_COOKIE, token,
                                                ctx.deps.website.signin_token_minutes * _MINUTE,
                                                "Strict"))
    return response


def signin_get(ctx: Ctx) -> Response:
    return _signin_form(ctx, None)


def signin_post(ctx: Ctx) -> Response:
    return sessions.sign_in(ctx.request, ctx.deps, ctx.now)


def signout(ctx: Ctx) -> Response:
    assert ctx.session is not None
    return sessions.sign_out(ctx.session, ctx.deps, ctx.now)


# The pages

def home(ctx: Ctx) -> Response:
    return sessions.redirect("/jobs")


def jobs_page(ctx: Ctx) -> Response:
    deps, session = ctx.deps, _session(ctx)
    stack_for(deps)
    views = list_jobs(deps.aws, session.account, deps.config, deps.website, ctx.now)
    return html_response(pages.job_list_page(views, csrf=session.csrf))


def start_get(ctx: Ctx) -> Response:
    deps, session = ctx.deps, _session(ctx)
    presets = [(name, deps.cloud.preset_plan(name).expected_minutes) for name in PRESETS]
    line = stop_line(deps.website) if is_stopped(ctx.now, deps.website) else None
    return html_response(pages.start_page(presets, deps.config.jobs().default_preset,
                                          csrf=session.csrf, stop_line=line,
                                          part_retries=deps.website.part_link_retries))


def _job_id(ctx: Ctx) -> str:
    try:
        return check_job_id(ctx.params["job_id"])
    except PipelineError:
        raise not_found() from None


def _visible(ctx: Ctx) -> tuple[str, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    job_id = _job_id(ctx)
    held, entry = may_see(ctx.deps.aws, _session(ctx).account, job_id)
    if held is None and entry is None:
        raise not_found()
    return job_id, held, entry


def _viewer(ctx: Ctx, view: JobView, base: str) -> pages.Viewer | None:
    if view.status not in DONE or view.result_file is None:
        return None
    deps = ctx.deps
    seconds = deps.website.download_link_seconds
    prefix = out_prefix(view.job_id)
    splat = deps.aws.s3_download_link(prefix + view.result_file, None, seconds)
    cameras = (deps.aws.s3_download_link(prefix + "cameras.json", None, seconds)
               if view.has_cameras else None)
    ctx.viewer_page = True
    return pages.Viewer(splat_url=splat, cameras_url=cameras, base=base)


def result(ctx: Ctx, share: pages.ShareState | None = None) -> Response:
    deps, session = ctx.deps, _session(ctx)
    job_id, held, entry = _visible(ctx)
    stack_for(deps)
    view = job_view(deps.aws, job_id, held or {}, deps.config, deps.website, ctx.now, timings=True)
    shown = read_entry(entry) if entry is not None else None
    finished = view.status in DONE and view.result_file is not None
    if held is not None and finished and share is None:
        share = pages.ShareState(on=isinstance(held.get("share_hash"), str))
    page = pages.result_page(view, csrf=session.csrf, poll_seconds=round(deps.cloud.poll_seconds),
                             viewer=_viewer(ctx, view, "/viewer"), download=True,
                             share=share if held is not None and finished else None,
                             showcase_line=licence_line(shown) if shown else None,
                             title=shown.title if shown else None)
    return html_response(page)


def download(ctx: Ctx) -> Response:
    deps = ctx.deps
    job_id, held, _ = _visible(ctx)
    stack_for(deps)
    view = job_view(deps.aws, job_id, held or {}, deps.config, deps.website, ctx.now, timings=False)
    if view.status not in DONE or view.result_file is None:
        raise not_found()
    link = deps.aws.s3_download_link(out_prefix(job_id) + view.result_file,
                                     f"{job_id}-{view.result_file}",
                                     deps.website.download_link_seconds)
    return sessions.redirect(link)


def _owned_finished(ctx: Ctx) -> tuple[str, Mapping[str, Any]]:
    deps = ctx.deps
    job_id, held, _ = _visible(ctx)
    if held is None:
        raise not_found()
    stack_for(deps)
    view = job_view(deps.aws, job_id, held, deps.config, deps.website, ctx.now, timings=False)
    if view.status not in DONE or view.result_file is None:
        raise PublicRefusal(BAD_REQUEST, "bad_request", shares.NOT_FINISHED)
    return job_id, held


def share_make(ctx: Ctx) -> Response:
    job_id, held = _owned_finished(ctx)
    token = shares.make_link(ctx.deps, _session(ctx), job_id, held, ctx.now)
    link = f"https://{ctx.request.host}/shared/{token}"
    return result(ctx, pages.ShareState(on=True, new_link=link))


def share_off(ctx: Ctx) -> Response:
    job_id, held = _owned_finished(ctx)
    shares.turn_off(ctx.deps, _session(ctx), job_id, held)
    return sessions.redirect(f"/jobs/{job_id}")


def showcase(ctx: Ctx) -> Response:
    session = _session(ctx)
    entries = [(e.job_id, e.title, licence_line(e)) for e in list_entries(ctx.deps.aws)]
    return html_response(pages.showcase_page(entries, csrf=session.csrf))


# The shared routes

def shared_page(ctx: Ctx) -> Response:
    deps = ctx.deps
    shared = shares.shared_job(deps, ctx.params["token"])
    stack_for(deps)
    view = job_view(deps.aws, shared.job_id, shared.job_item, deps.config, deps.website, ctx.now,
                    timings=False)
    if view.status not in DONE or view.result_file is None:
        raise not_found()
    viewer = _viewer(ctx, view, f"/shared/{ctx.params['token']}/viewer")
    page = pages.result_page(view, csrf=None, poll_seconds=round(deps.cloud.poll_seconds), viewer=viewer,
                             download=False, share=None, shared=True)
    response = html_response(page)
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


def shared_static(ctx: Ctx) -> Response:
    shares.shared_job(ctx.deps, ctx.params["token"])
    return viewer_static(ctx)


# The Script_Requests

def _json(ctx: Ctx) -> dict[str, Any]:
    try:
        data = json.loads(ctx.request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, ValueError):
        data = None
    if not isinstance(data, dict):
        raise PublicRefusal(BAD_REQUEST, "bad_request", "That request wasn't understood.")
    return data


def api_precheck(ctx: Ctx) -> Response:
    website_start.precheck(ctx.deps, _session(ctx), _json(ctx).get("preset"), ctx.now)
    return json_response(OK, {"ok": True})


def api_begin(ctx: Ctx) -> Response:
    body = _json(ctx)
    stack_for(ctx.deps)
    upload = uploads.begin(ctx.deps, _session(ctx), body.get("name"), body.get("size"), ctx.now)
    return json_response(OK, uploads.describe(upload, ctx.deps.website.upload_part_bytes))


def api_part(ctx: Ctx) -> Response:
    _json(ctx)
    stack_for(ctx.deps)
    url = uploads.part_link(ctx.deps, _session(ctx), ctx.params["upload_id"], int(ctx.params["n"]))
    return json_response(OK, {"url": url})


def api_complete(ctx: Ctx) -> Response:
    body = _json(ctx)
    stack_for(ctx.deps)
    uploads.complete(ctx.deps, _session(ctx), ctx.params["upload_id"], body.get("parts"))
    return json_response(OK, {"ok": True})


def api_abort(ctx: Ctx) -> Response:
    _json(ctx)
    stack_for(ctx.deps)
    uploads.abort(ctx.deps, _session(ctx), ctx.params["upload_id"])
    return json_response(OK, {"ok": True})


def api_start(ctx: Ctx) -> Response:
    body = _json(ctx)
    started = website_start.start(ctx.deps, _session(ctx), body.get("upload"), body.get("preset"),
                                  ctx.now)
    data: dict[str, Any] = {"job": started.job_id, "outcome": started.outcome}
    if started.outcome == "unknown":
        data["message"] = website_start.UNKNOWN
    return json_response(OK, data)


def api_status(ctx: Ctx) -> Response:
    deps = ctx.deps
    job_id, held, _ = _visible(ctx)
    stack_for(deps)
    view = job_view(deps.aws, job_id, held or {}, deps.config, deps.website, ctx.now, timings=False)
    return json_response(OK, {"status": view.status, "text": status_text(view),
                              "final": view.status in FINAL})


def _session(ctx: Ctx) -> sessions.Session:
    assert ctx.session is not None
    return ctx.session


def _route(method: str, pattern: str, kind: str, run: Callable[[Ctx], Response]) -> Route:
    return Route(method, re.compile(pattern), kind, run.__name__, run)


_JOB = r"(?P<job_id>[^/]+)"
_UPLOAD = r"(?P<upload_id>[0-9a-f]{32})"
_TOKEN = r"(?P<token>[^/]+)"

ROUTES: tuple[Route, ...] = (
    _route("GET", r"/signin", "open", signin_get),
    _route("POST", r"/signin", "open", signin_post),
    _route("GET", r"/static/style\.css", "open", open_static),
    _route("GET", r"/favicon\.ico", "open", open_static),
    _route("GET", rf"/shared/{_TOKEN}", "shared", shared_page),
    _route("GET", rf"/shared/{_TOKEN}/viewer/(?P<file>[A-Za-z0-9_./-]+)", "shared", shared_static),
    _route("GET", r"/", "page", home),
    _route("GET", r"/jobs", "page", jobs_page),
    _route("GET", r"/start", "page", start_get),
    _route("GET", rf"/jobs/{_JOB}", "page", result),
    _route("GET", rf"/jobs/{_JOB}/download", "page", download),
    _route("POST", rf"/jobs/{_JOB}/share", "page", share_make),
    _route("POST", rf"/jobs/{_JOB}/share/off", "page", share_off),
    _route("GET", r"/showcase", "page", showcase),
    _route("GET", r"/static/app\.js", "page", page_static),
    _route("GET", r"/viewer/(?P<file>[A-Za-z0-9_./-]+)", "page", viewer_static),
    _route("POST", r"/signout", "page", signout),
    _route("POST", r"/api/precheck", "script", api_precheck),
    _route("POST", r"/api/uploads", "script", api_begin),
    _route("POST", rf"/api/uploads/{_UPLOAD}/parts/(?P<n>[1-9][0-9]{{0,4}})", "script", api_part),
    _route("POST", rf"/api/uploads/{_UPLOAD}/complete", "script", api_complete),
    _route("POST", rf"/api/uploads/{_UPLOAD}/abort", "script", api_abort),
    _route("POST", r"/api/jobs", "script", api_start),
    _route("GET", rf"/api/jobs/{_JOB}/status", "script", api_status),
)


def load_static(root: Path) -> dict[str, bytes]:
    """The website's static files from a bundle or repo root, by URL path. A missing file is left
    out, so its route answers 404."""
    base = Path(root) / "app"
    files = {"/static/style.css": base / "static" / "style.css",
             "/favicon.ico": base / "static" / "favicon.ico",
             "/static/app.js": base / "static" / "app.js"}
    files.update({f"/viewer/{name}": base / "viewer" / name for name in VIEWER_FILES})
    return {url: path.read_bytes() for url, path in files.items() if path.is_file()}

