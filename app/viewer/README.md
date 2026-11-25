# The 3D viewer

A web viewer for one finished job's splat. It opens `out/clean.sog`, or `out/train.ply` when the job has no clean copy, and starts at the job's first camera in `out/cameras.json`, with up pointing up.

- **Laptop:** drag to turn, scroll (or pinch on a trackpad) to zoom, and WASD or the arrow keys to walk.
- **Phone:** drag with one finger to turn, and pinch to zoom.

It's plain HTML, CSS and JavaScript with no build step and no iframe. It runs on the PlayCanvas engine 2.24.3 (MIT), drawn with WebGL 2. Engine 2.11.0 was the first to read the SOG version 2 that splat-transform 3.8.0 writes, 2.11.1 to 2.11.7 fixed it, and 2.24.0 added reading each file's anti-aliasing flag. The engine also reads SOG version 3, but not the old version 3 layout of splat-transform 3.11.0 and 3.11.1, so if splat-transform ever moves past 3.8.0, skip those two.

## Files

| File | What it is |
|---|---|
| `viewer.js` | `mountViewer(container, sources)`: the engine, the canvas, loading and the controls |
| `params.js` | Which files to load, from a URL parameter or from data attributes |
| `cameras.js` | Reads `out/cameras.json` and works out the starting camera |
| `controls.js` | How a drag, a pinch, the wheel and the keys move the camera |
| `embed.js` | Starts a viewer in every element with a `data-splat-url`, for the website |
| `page.js`, `index.html` | The standalone page, for testing on a laptop |
| `viewer.css` | Styles only `.splat-viewer`, what's inside it and the standalone page |
| `package.json` | Only tells Node that these `.js` files are ES modules, for the tests. No dependencies |
| `vendor/playcanvas.min.mjs`, `vendor/PLAYCANVAS-LICENSE.txt` | The engine and its licence. Git ignores `vendor/`: `scripts/fetch_viewer_engine.py` downloads them |

The tests are in `tests/viewer/*.test.mjs`, which `tests/test_viewer.py` runs with Node's built-in test runner. They need no npm package.

## On a laptop

```
uv run python scripts/fetch_viewer_engine.py
uv run python -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/app/viewer/?url=/data/jobs/<job id>/out/` from the repo root. A URL ending in `/` is a job's `out/` folder: the viewer tries `clean.sog`, then `train.ply`, and reads `cameras.json` beside them. `?url=` can also name one `.sog` or `.ply` file, and `&cameras=` a cameras file. A signed link signs only its own file, so with a signed splat link and no `&cameras=`, the viewer frames the splat from outside without the cameras. The standalone page has the same Content-Security-Policy as below in a `<meta>` tag, so a laptop test also tests the policy.

## What it needs from the website

This fits the website design on `feat/website` (`.kiro/specs/website/design.md`), so the website's spec takes these changes.

### Its files

The website serves these from the bundle, read into `Deps.static` like `app.js`:

| Route | File | Content-Type |
|---|---|---|
| `GET /viewer/viewer.css` | `app/viewer/viewer.css` | `text/css; charset=utf-8` |
| `GET /viewer/embed.js` | `app/viewer/embed.js` | `text/javascript; charset=utf-8` |
| `GET /viewer/viewer.js`, `/viewer/params.js`, `/viewer/cameras.js`, `/viewer/controls.js` | the same names in `app/viewer/` | `text/javascript; charset=utf-8` |
| `GET /viewer/vendor/playcanvas.min.mjs` | the engine, 2,895,086 bytes | `text/javascript; charset=utf-8` |
| `GET /viewer/vendor/PLAYCANVAS-LICENSE.txt` | the engine's MIT licence, which its terms ask us to keep with it | `text/plain; charset=utf-8` |

- The scripts are ES modules, and browsers refuse a module with any other Content-Type under `nosniff`. They import each other by relative path, so they must sit side by side under `/viewer/`, with the engine under `/viewer/vendor/`.
- `index.html` and `page.js` are only for the laptop. The website doesn't need to serve them.
- **The engine isn't in git**, so the Deploy_Command's `git archive` doesn't have it. After extracting `app/` into the build folder, the deploy runs `python3.11 scripts/fetch_viewer_engine.py --into <build>/app/viewer/vendor` from the same commit. It uses only the standard library, and it checks both files against their pinned sha256, so a commit still always builds the same bundle bytes.
- A Lambda function URL answers at most 6 MB. The engine is 2.9 MB, about 3.9 MB once base64-encoded, so it fits. Sending it gzipped, 0.75 MB, is faster on a phone when the request's `Accept-Encoding` allows it.
- `Cache-Control: no-store` would send the 2.9 MB engine again with every Result_Page. The viewer's files hold no Account data, so something like `Cache-Control: private, max-age=3600` suits them.

### The Content-Security-Policy

The website's policy plus two sources:

```text
default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; worker-src blob:; connect-src 'self' <s3_origin>; form-action 'self'; frame-ancestors 'none'; base-uri 'none'
```

- `worker-src blob:`: the engine sorts the splats in a Web Worker that it starts from a `blob:` URL. Without `worker-src`, browsers fall back to `script-src 'self'`, which blocks it. Tested in Chrome: the worker is refused and nothing is drawn.
- `img-src blob:`: Safari on WebGL decodes the `.sog`'s WebP images through a `blob:` URL in an `<img>`. Chrome and Firefox use `createImageBitmap` and don't need it. This comes from the engine's source and hasn't been tested in Safari yet.
- `connect-src 'self' <s3_origin>` is already the website's. The viewer fetches the splat and `cameras.json` from the job bucket with `fetch()`, and nothing else.
- There's no inline script or style, no `eval` and no WebAssembly, so nothing else is needed. The engine sets the canvas's size through `element.style`, which the policy doesn't block.

### The job bucket's CORS

The viewer reads the bucket from the website's origin, so the bucket's CORS needs a second rule beside `website-part-uploads`:

```yaml
- Id: website-viewer-reads
  AllowedMethods: [GET, HEAD]
  AllowedOrigins:
    - !Join ["", ["https://", !Select [2, !Split ["/", !GetAtt WebsiteFunctionUrl.FunctionUrl]]]]
```

It needs no `AllowedHeaders`, because the viewer's requests set no header of their own, and no `ExposedHeaders`, because `Content-Length`, which the loading percentage reads, is always exposed. The viewer sends no cookies to the bucket.

### How the Result_Page starts it

For a Done or Done with warnings Job with a Result_File, the Viewer_Slot gets the viewer's element after its heading, and the page loads the stylesheet and `embed.js`:

```html
<link rel="stylesheet" href="/viewer/viewer.css">
...
<section id="viewer-slot" aria-labelledby="viewer-label">
  <h2 id="viewer-label">3D viewer</h2>
  <div class="splat-viewer" data-splat-url="{esc(splat link)}" data-cameras-url="{esc(cameras link)}"></div>
</section>
<script type="module" src="/viewer/embed.js"></script>
```

- `data-splat-url` is a Presigned_Link that signs `get_object` for the Result_File: `out/clean.sog`, or `out/train.ply` when there's no clean copy, which is what `result_file` already picks. The viewer reads the file type from the link's path, so the link must end in `.sog` or `.ply` before its query string. A link to `/jobs/<id>/download` won't do, because its path has no extension.
- `data-cameras-url` is a second Presigned_Link, for `out/cameras.json`. Leave the attribute out when the Job has no `out/cameras.json`, or when `out/reports/cameras.json` doesn't say done. The viewer then frames the splat from outside.
- `esc()` both values, as every value on a page is: a signed link holds `&`.
- The links only need to last until the page has started the downloads, because S3 checks a link's expiry when a request starts. The Download_Links' expiry is enough.
- The website design makes links only for the Result_File. These two GET links widen that rule, and its property test, to the Result_File and `out/cameras.json` of the same Jobs: the Account's held job or a Showcase_Entry.
- The viewer draws only while it's on screen. It keeps the keyboard to itself only while it has focus, which it takes on the first tap or click.
- A Job with no Result_File gets no viewer element, and so no viewer.

## Phones

- It draws with WebGL 2, with no anti-aliasing, at no more than 1.5 device pixels per CSS pixel on a touch screen (2 with a mouse), and only while it's on screen.
- The `.sog` is the phone copy. A `train.ply` of a `quality` job can hold 3,000,000 splats, about 700 MB, too much for a phone, so the website should show the PLY only when there's no clean copy, as `result_file` does.
- The canvas has `touch-action: none`, so a drag or a pinch moves the camera, not the page.
