---
name: playcanvas-splats
description: How the PlayCanvas engine and @playcanvas/supersplat-viewer 1.43.3 show Gaussian splats in a browser. Which files they load (the .sog that splat-transform 3.8.0 writes, a 3DGS .ply), the first engine release that reads that .sog, putting the viewer in a web page with no build step and giving it a file's URL, its camera controls on a laptop and on a phone, what it needs from a page's Content-Security-Policy, and its licences. Use when building or reviewing the web viewer page, picking viewer or engine versions, writing the page's CSP or the file host's CORS rules, or explaining the viewer's controls.
metadata:
  sources: the v1.43.3 tag of supersplat-viewer (README, docs/api.md, src/, LICENSE, THIRD_PARTY_NOTICES.md), its files on jsDelivr, the v2.24.1 tag of the PlayCanvas engine and its release notes back to v2.11.0, the v3.8.0 tag of splat-transform, and the PlayCanvas developer docs, read November 2026. Nothing was installed or run.
---

# PlayCanvas splats in the browser

Notes on two PlayCanvas projects: the engine's Gaussian splat loading, and the SuperSplat
Viewer (`@playcanvas/supersplat-viewer`), the open-source page that superspl.at uses to show a
splat. They come from reading the source on GitHub and the PlayCanvas docs, in our own words.
Links are at the bottom.

The CSP section is worked out from the source, not from a test in a browser. Check it in
Chrome and in Safari before relying on it.

## Versions at a glance

| Thing | Version read | Note |
| --- | --- | --- |
| supersplat-viewer | 1.43.3 (latest tag, November 2026) | Its `public/index.js` bundles the engine it was built with |
| Engine bundled in viewer 1.43.3 | 2.24.1 | `createViewer` from `/viewer` needs `playcanvas` ^2.24.1 as a peer |
| Engine that first reads a 3.8.0 `.sog` | 2.11.0 | 2.11.7 is the first without the known bugs, see below |
| splat-transform | 3.8.0 | Writes SOG version 2. Its own `.html` output embeds viewer 1.36.2 |

The viewer changes quickly: seven releases came after 1.40.0, up to 1.43.3. Pin an exact
version in every URL. The docs also warn that SOG version 3 files from splat-transform 3.11.0
or 3.11.1 use an old layout the viewer no longer reads. That doesn't touch 3.8.0's version 2.

## What each loads

The viewer picks the format from the file name's extension. It takes the name from the last
segment of the URL's path, so a query string (an S3 presigned URL, say) doesn't matter.

| Format | File name | Engine parser |
| --- | --- | --- |
| Bundled SOG | `*.sog` | `SogBundleParser` |
| Unbundled SOG | `meta.json`, with its WebP files beside it | `SogParser` |
| Streamed SOG | `lod-meta.json` | `GSplatOctreeParser` |
| PLY | `*.ply`, `*.compressed.ply` | `PlyParser` |
| SPZ | `*.spz` (versions 3 and 4) | SPZ parser from the engine's scripts. Version 4 needs a wasm decoder |
| XGRIDS LCC2 | `*.lcc2` | `GSplatLcc2Parser` |
| Animated sequence | `*.gsaf` | The viewer's own wasm decoder |

An unknown extension fails with "No parser found". There is no fallback to PLY in 2.24.

### The .sog that splat-transform 3.8.0 writes

- It is a plain zip. 3.8.0 stores every entry uncompressed (method 0). It sets the
  "sizes follow the data" flag, so the local headers say 0 bytes and the real sizes are in
  the central directory. No Zip64, so a bundle stops at 4 GiB.
- The engine reads the zip from its central directory, so the zero local sizes don't matter.
  It inflates entries that use deflate (method 8) with the browser's `DecompressionStream`,
  and ignores entries that `meta.json` doesn't name.
- Inside are `meta.json` and the WebP images it lists. 3.8.0 writes `"version": 2`, `count`,
  `asset.generator`, and a `model` key when the scene isn't plain 3DGS (`"antialiased"` or
  `"2dgs"`).
- Engine 2.24 accepts `version` 2, 3 or missing, and refuses anything else. Older engines
  don't check the version.
- Each WebP becomes a texture. The engine decodes it with `createImageBitmap` from the bytes
  in memory. Where that isn't used, it hands the bytes to an `<img>` through a `blob:` URL.
  That happens in Safari on WebGL, because the engine turns ImageBitmap off on Safari's WebGL.
- Engine 2.24 turns on the anti-aliasing correction when `model` is `"antialiased"`.
- For the unbundled form, name the file `meta.json`. The viewer reads it as JSON itself, and
  the engine fetches the images relative to it.

### A 3DGS .ply

- Only `binary_little_endian`. ASCII and big-endian files fail with "Unsupported ply format".
- A file with one `vertex` element and only `float` properties takes the fast path. Other
  property types go through a slower path. SuperSplat's `.compressed.ply` (`chunk`,
  `vertex` and optional `sh` elements) has its own path.
- It reads `x y z`, `scale_0..2` (log), `rot_0..3` (w first), `f_dc_0..2`, and `opacity`
  (logit). If `opacity_linear` is there, it wins over `opacity`. The number of `f_rest_*`
  columns sets the SH bands: 9, 24 or 45 for 1-3 bands, and 72 for band 4 in 2.24. Other
  columns such as `nx ny nz` are loaded and ignored.
- The engine reorders the splats in memory at load time, for faster drawing.
- Header comments set the anti-aliasing mode (2.24 and later). `SplatRenderMode: mip` or
  `postshot.anti_aliasing=1` turns it on. The last such comment wins, and case doesn't
  matter.
- Progress comes from `Content-Length`. Without that header the loading bar can't show a
  percentage.
- A PLY with 3 SH bands is 248 bytes per splat, so 3 million splats are about 700 MB. That's
  why the web copy is the SOG.

### Orientation

The viewer turns every format except `.spz` and `.lcc2` by 180° about z, which treats the file
as y-down. That matches the 3DGS convention and our levelled frame (`docs/pipeline.md` puts
world up at (0, −1, 0)), so our PLY and SOG should show upright without changes.

## The first engine release that reads that .sog

**2.11.0, released 3 September 2025.** PR #7929 added the bundled `.sog` parser and SOG
version 2. Before it, in 2.10.x, the gsplat handler knew only `.json` (SOGS version 1) and
sent every other extension to the PLY parser. A `.sog` would fail there.

Fixes after 2.11.0 that a 3.8.0 file can hit:

| Release | PR | Fix | When it matters to us |
| --- | --- | --- | --- |
| 2.11.1 | #7947 | Colour precision on mobile GPUs | Every SOG viewed on a phone |
| 2.11.6 | #7972 | SH read wrongly for SOGs with 1 or 2 bands | When we write `-H 1` or `-H 2` |
| 2.11.7 | #7982 | Two bundles' images collided in the texture cache | More than one `.sog` in one page |
| 2.24.0 | #9921 | Anti-aliasing mode read from each file | Our splats trained with anti-aliasing |
| 2.24.0 | #9873 | Reads SOG version 3 | Only after we move past splat-transform 3.8.0 |

So 2.11.7 is the practical minimum, and 2.24.0 or later is needed for the `model` tag to
count. Viewer 1.43.3 ships 2.24.1, so it has all of these.

## Putting the viewer in a page with no build step

### What the package ships

The npm package's `public/` folder is the whole viewer as static files:

| File | Size | What it is |
| --- | --- | --- |
| `index.html` | 7 KB | The standalone page. Reads URL parameters |
| `index.js` | 4.1 MB | The viewer and the engine in one ES module, which exports `createViewer` |
| `index.js.map` | 8.3 MB | Source map |
| `index.css` | 35 KB | Styles. They only style the viewer's own `.sse-viewer` subtree |
| `zstd-decoder.wasm`, `gsaf-decoder.wasm` | 54 KB, 171 KB | Fetched only for `.spz` version 4 and `.gsaf` |

`index.js` and its map are over our 1 MB git limit, so they never go in the repo. Either load
them from a CDN at a pinned version, or download them when we deploy. Node is only in our
stack for splat-transform, so don't `npm install` the viewer. jsDelivr serves the package's
files at `https://cdn.jsdelivr.net/npm/@playcanvas/supersplat-viewer@1.43.3/public/...`.

### Option A: our own page, viewer module from the CDN

This works with three small static files and no bundler, because `public/index.js` already
includes the engine. Keeping our script in its own file means the CSP needs no
`'unsafe-inline'`.

`viewer.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
  <title>Splat</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@playcanvas/supersplat-viewer@1.43.3/public/index.css">
  <link rel="stylesheet" href="./viewer-page.css">
  <script type="module" src="./viewer-page.js"></script>
</head>
<body>
  <div id="viewer"></div>
</body>
</html>
```

`viewer-page.css`. The container must have a size, because an element with no height shows
nothing:

```css
html, body { height: 100%; margin: 0; overflow: hidden; }
#viewer { position: fixed; inset: 0; }
```

`viewer-page.js`:

```js
import { createViewer } from 'https://cdn.jsdelivr.net/npm/@playcanvas/supersplat-viewer@1.43.3/public/index.js';

// A full version 2 settings object: the values defaultSettings() gives, but with no cameras,
// so the viewer frames the splat itself.
const settings = {
  version: 2,
  tonemapping: 'linear',
  highPrecisionRendering: false,
  background: { color: [0, 0, 0] },
  postEffectSettings: {
    sharpness: { enabled: false, amount: 0 },
    bloom: { enabled: false, intensity: 0.1, blurLevel: 2 },
    grading: { enabled: false, brightness: 1, contrast: 1, saturation: 1, tint: [1, 1, 1] },
    vignette: { enabled: false, intensity: 0.5, inner: 0.3, outer: 0.75, curvature: 1 },
    fringing: { enabled: false, intensity: 0.5 }
  },
  animTracks: [],
  cameras: [],
  annotations: [],
  startMode: 'default'
};

const contentUrl = new URL(location.href).searchParams.get('content') ?? './scene.sog';

const viewer = await createViewer({
  container: document.getElementById('viewer'),
  settings,      // an object, a promise of one, or a URL to a settings.json
  contentUrl     // .sog, .ply or meta.json; the extension picks the parser
});

viewer.events.on('progress:changed', (percent) => { /* a loading bar of our own */ });
viewer.events.on('loaded:changed', () => { /* first frame is on screen */ });
```

Things to know about `createViewer`:

- `settings` is required and has to be a complete version 2 object. Something like `{}`
  crashes: with no `version` it's read as version 1 and the code reaches for
  `background.color`. A settings URL that returns 404 fails too.
- With `cameras: []` the viewer places the camera outside the splat's bounds, along
  (2, 1, 2) from the centre. So the splat counts as an object: the start loop circles it,
  and the first input leaves the visitor in orbit mode. For a filmed room that means looking
  at the room from outside its walls.
- A camera in `settings.cameras[0].initial` (`position`, `target`, `fov` in degrees) that
  sits inside the bounds makes the splat a scene. The loop is then a figure-eight, and the
  visitor lands in fly mode, or walk mode when there's collision data. The values are in
  the viewer's world frame, which is the file's frame turned 180° about z, so a point
  (x, y, z) from our cameras becomes (−x, −y, z).
- It resolves once the graphics device exists and loading has started, before the splat is
  on screen. Wait for `loaded:changed` before calling `frameScene()`, `resetCamera()` and
  the other camera methods. They throw before that.
- For a `data:` or `blob:` URL, also pass `contentFilename: 'scene.sog'`, because those URLs
  have no file name to take the extension from.
- Call `viewer.destroy()` when the viewer goes away. Browsers allow about 16 live WebGL
  contexts per page.
- `renderer: 'webgl'` forces WebGL 2. The default asks for WebGPU and falls back to WebGL 2.

### Option B: host the prebuilt viewer and link to it

Copy `public/` (downloaded at deploy, not committed) to a folder on our host and open
`index.html?content=<url>&settings=<url>`. With no parameters it loads `./settings.json` and
`./scene.compressed.ply` from beside it. Other parameters: `poster`, `skybox`, `noui`,
`noanim`, `webgl`, `aa`, `budget`, `lang` and a few more listed in the README. To put it inside
another page, use an `<iframe>`. Give it `allow="fullscreen; xr-spatial-tracking"`, or the
fullscreen and VR buttons can't work. That comes from the browser rules for iframes, not
from the viewer's docs.

The shipped `index.html` contains an inline `<style>` and two inline module scripts, so a
strict CSP needs their hashes or `'unsafe-inline'`. That's one reason to prefer option A.

### Serving the files

- When the splat lives on another origin (S3 or CloudFront), that host must send
  `Access-Control-Allow-Origin` for our page's origin. The viewer fetches the file with
  `fetch()`, and the engine asks for images with `crossOrigin = 'anonymous'`. Credentials go
  only to the same origin.
- Send `Content-Length`, so the loading bar has a percentage.
- The `?content=` parameter lets anyone point our page at any URL. The CSP's `connect-src`
  limits which hosts can actually be fetched, so keep that list to our own file host.

## Camera controls

On load, the camera plays a slow 20-second loop: an orbit around an object, or a figure-eight
inside a scene. The first input stops it and gives the camera to the visitor. `noanim` (or
`noanim: true`) loads it paused instead.

There are four modes: orbit (key 1), fly (key 2), walk (key 3) and animation. Walk needs
collision data and a scene at least 5 m across, and our pipeline writes no collision data,
so for us it's orbit and fly. Fly and walk each have two schemes. By default you click or tap
to move and drag to look. **Gaming Controls** captures the mouse on a laptop (toggle with
G, release with Esc) and shows a joystick on a phone (turned on in the settings panel). The
viewer keeps that choice and the Quality setting in `localStorage`, shared by every viewer on
our origin.

### Laptop: mouse and keyboard

| Action | Orbit | Fly |
| --- | --- | --- |
| Orbit or look around | Left-drag | Left-drag |
| Pan | Right-drag | Right-drag |
| Zoom or move forward and back | Wheel | Wheel, or W and S |
| Set focus or fly to the spot | Click | Click |
| Fly to a point | Double-click | Double-click focuses there |
| Strafe | | A and D, plus Q and E for up and down with Gaming Controls |
| Faster or slower | | Shift or Ctrl |

Keys that work in any mode: F frames the scene (not in walk mode), R resets the camera, Space
plays or pauses the animation, H shows or hides the controls panel, Shift+H shows the info
panel (versions, renderer, GPU), Esc releases the mouse or leaves walk mode. Pressing W, A, S
or D in orbit mode switches to walk mode, or to fly when walking isn't possible, and turns on
Gaming Controls. Letter shortcuts follow the keyboard layout's characters, while WASD always
uses the key positions.

### Laptop: trackpad

The docs don't cover the trackpad. From `src/input/devices/trackpad.ts`:

| Gesture | Effect |
| --- | --- |
| Two-finger scroll | Same as the mouse wheel: zoom in orbit, forward and back in fly |
| Pinch | Zoom, tuned for trackpads. macOS sends a pinch as a wheel event with Ctrl set |
| Shift + two-finger scroll | Pan in orbit, strafe in fly |
| Held Ctrl + two-finger scroll | Orbit in orbit mode, look around in fly |
| One-finger click-drag | Orbit or look, like a left-drag |

The viewer tells a pinch apart from a real Ctrl press by watching the Ctrl key itself.

### Phone: touch

| Action | Orbit | Fly |
| --- | --- | --- |
| Orbit or look around | One-finger drag | Touch and drag |
| Pan | Two-finger drag | |
| Zoom or move | Pinch | Pinch or two-finger drag |
| Set focus or fly to the spot | Tap | Tap |
| Fly to a point | Double-tap | Double-tap focuses there |

The animation has no touch controls: any touch ends it. With Gaming Controls on, a joystick
moves and dragging looks around. On phones, Quality starts at **Performance**, which draws at
half resolution on the standard renderer. Turning the phone to landscape asks for fullscreen,
and turning it back leaves fullscreen. The shipped page's viewport tag
(`maximum-scale=1, user-scalable=no`) keeps a pinch from zooming the page instead of the
splat, so copy it.

### Keyboard focus in a page of our own

The viewer listens for keys on `window`, so its shortcuts fire even while our own text box has
focus. Set `viewer.state.inputEnabled = false` while our own controls have focus, and back to
`true` afterwards.

## What it needs from a Content-Security-Policy

| Directive | Needs | Why |
| --- | --- | --- |
| `script-src` | `'self'`, plus `https://cdn.jsdelivr.net` for option A | Our module and the viewer's module. The viewer and the engine use no `eval` or `new Function`. The shipped `index.html` needs hashes or `'unsafe-inline'` for its two inline scripts |
| `worker-src` | `blob:` | The engine's splat-sorting worker is started from a `blob:` URL. WebGL always sorts there, and WebGPU falls back to it when the GPU sort hits a limit. Without `worker-src`, browsers fall back to `child-src`, then `script-src` |
| `connect-src` | Our file host, and `'self'` for anything on our origin | `fetch()` of the `.sog`, `.ply`, `meta.json`, `settings.json` and skybox, and the engine's requests for unbundled SOG images. Add `data:` if `contentUrl` is a `data:` URI. In VR or AR, the engine's controller script loads hand and controller models from `https://cdn.jsdelivr.net` |
| `img-src` | `blob:`, our file host, the poster's host | Safari on WebGL decodes the bundled SOG's WebP images through `blob:` URLs in `<img>`. It also loads unbundled SOG images and the skybox through `<img>`. The poster is an `<img>` shown as a CSS background |
| `style-src` | `'self'`, plus `https://cdn.jsdelivr.net` for option A | `index.css`. The viewer sets styles from script, which CSP doesn't block, and its markup has no `style` attributes. The shipped `index.html` has one inline `<style>` |
| `script-src` `'wasm-unsafe-eval'` | Not needed for `.sog` or `.ply` | Only the `.spz` version 4 and `.gsaf` decoders use WebAssembly. The `.spz` decoder's glue is also loaded from a `blob:` URL |
| `media-src` | Only if settings have a `soundUrl` | That plays through an `<audio>` element |
| `font-src` | Nothing | The UI uses Arial from the system |
| Trusted Types | Don't require them | The viewer builds its UI with `innerHTML`, which `require-trusted-types-for 'script'` blocks |

WebGPU, WebGL, `createImageBitmap` and `DecompressionStream` need nothing from the CSP.

A starting policy for option A, with `https://files.example` standing in for our file host:

```text
default-src 'none';
script-src 'self' https://cdn.jsdelivr.net;
style-src 'self' https://cdn.jsdelivr.net;
worker-src blob:;
connect-src 'self' https://files.example;
img-src 'self' blob: https://files.example;
base-uri 'none';
form-action 'none';
frame-ancestors 'none';
```

Add `https://cdn.jsdelivr.net` to `connect-src` if we offer VR or AR. If we self-host the
viewer files instead of using the CDN, drop the CDN from `script-src` and `style-src`. If our
page is itself put in an iframe, replace `frame-ancestors 'none'` with the sites allowed to
embed it.

## Licences

| Part | Licence | Where it is |
| --- | --- | --- |
| SuperSplat Viewer 1.43.3 | MIT, © PlayCanvas Ltd | All of `public/` and `dist/` |
| PlayCanvas engine 2.24.1 | MIT, © PlayCanvas Ltd | Bundled inside `public/index.js` |
| gsaf decoder (Snap Inc.) | MIT | `gsaf-decoder.wasm`, and its glue inside `index.js` |
| GLM 0.9.9.8, in the gsaf decoder | Happy Bunny or MIT, so we take MIT | `gsaf-decoder.wasm` |
| xsimd 13.0.0, in the gsaf decoder | BSD-3-Clause | `gsaf-decoder.wasm` |
| Emscripten 6.0.1 runtime | MIT or University of Illinois/NCSA, so we take MIT | Both wasm decoders |
| musl libc, through Emscripten | MIT. A few files are BSD, or public domain with a permissive fallback | Both wasm decoders |
| LLVM runtimes (libc++, libc++abi, compiler-rt) | Apache-2.0 with the LLVM exception | Both wasm decoders |
| zstddec 0.3.4 | MIT | `zstd-decoder.wasm` and its glue |
| Zstandard | BSD-3-Clause, as the notices ship it | `zstd-decoder.wasm` |

Everything is on our permitted list. The only odd case is musl's public-domain files with a
fallback licence. Ask Samarth if we want to be strict about it, though the files we load for
`.sog` and `.ply` never run the decoders. MIT asks us to keep the copyright notice, so
serve the package's `LICENSE` and `THIRD_PARTY_NOTICES.md` beside the files when we self-host
them.

## Our rules that touch this

- The viewer and the engine each get a row in `docs/third-party-licences.md`, in the commit
  that adds the viewer page. Note the decoders bundled in `index.js` on the viewer's row.
- `index.js` (4.1 MB) and `index.js.map` (8.3 MB) are over 1 MB. Never commit them. Use the
  pinned CDN URL, or download them at deploy time.
- Node is in our stack for splat-transform alone. The no-build page needs no Node or npm.
- Splats and SOGs stay in `data/` and on the file host, never in git.
- A splat made from CC BY or CC BY-SA footage shows the author's credit next to the viewer.
  A CC BY-SA one is also shared under CC BY-SA 4.0.

## Links

Viewer, at the version read:
- Repo at v1.43.3: <https://github.com/playcanvas/supersplat-viewer/tree/v1.43.3>
- README, URL parameters: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/README.md>
- API reference (`createViewer`, `renderViewerHtml`, settings): <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/docs/api.md>
- The standalone page: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/src/index.html>
- `createViewer` and format loading: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/src/index.ts>
- Controls per mode and input: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/src/ui/controls-hint.ts>
- Keyboard shortcuts: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/src/input/app/mode-shortcuts.ts>
- Trackpad: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/src/input/devices/trackpad.ts>
- Starting camera and mode: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/src/camera-manager.ts>
- Licence and notices: <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/LICENSE>, <https://github.com/playcanvas/supersplat-viewer/blob/v1.43.3/THIRD_PARTY_NOTICES.md>
- npm package: <https://www.npmjs.com/package/@playcanvas/supersplat-viewer>
- Published files and sizes: <https://data.jsdelivr.com/v1/packages/npm/@playcanvas/supersplat-viewer@1.43.3?structure=flat>

PlayCanvas docs:
- Viewer overview: <https://developer.playcanvas.com/user-manual/supersplat/developers/viewer/>
- Embedding, URL parameters, supported content: <https://developer.playcanvas.com/user-manual/supersplat/developers/embedding/>
- Viewer API: <https://developer.playcanvas.com/user-manual/supersplat/developers/viewer-api/>
- Self-hosting: <https://developer.playcanvas.com/user-manual/supersplat/developers/self-hosting/>
- Experience settings: <https://developer.playcanvas.com/user-manual/supersplat/developers/experience-settings/>
- Viewer controls: <https://developer.playcanvas.com/user-manual/supersplat/viewing/controls/>
- PLY and SOG version 2 formats: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/>, <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog-v2/>

Engine, at the version the viewer bundles:
- gsplat handler and its parsers: <https://github.com/playcanvas/engine/blob/v2.24.1/src/framework/handlers/gsplat.js>
- Bundled SOG parser: <https://github.com/playcanvas/engine/blob/v2.24.1/src/framework/parsers/sog-bundle.js>
- PLY parser: <https://github.com/playcanvas/engine/blob/v2.24.1/src/framework/parsers/ply.js>
- Image decoding (ImageBitmap or `blob:` `<img>`): <https://github.com/playcanvas/engine/blob/v2.24.1/src/framework/parsers/texture/img.js>
- Sort worker from a `blob:` URL: <https://github.com/playcanvas/engine/blob/v2.24.1/src/scene/gsplat-unified/gsplat-unified-sorter.js>
- Release 2.11.0 and PR #7929: <https://github.com/playcanvas/engine/releases/tag/v2.11.0>, <https://github.com/playcanvas/engine/pull/7929>
- Later fixes: <https://github.com/playcanvas/engine/pull/7947>, <https://github.com/playcanvas/engine/pull/7972>, <https://github.com/playcanvas/engine/pull/7982>
- Release 2.24.0 (anti-aliasing per file, SOG version 3): <https://github.com/playcanvas/engine/releases/tag/v2.24.0>

splat-transform 3.8.0:
- Zip writer (stored entries): <https://github.com/playcanvas/splat-transform/blob/v3.8.0/src/lib/io/write/zip-file-system.ts>
- SOG writer and `meta.json`: <https://github.com/playcanvas/splat-transform/blob/v3.8.0/src/lib/writers/write-sog.ts>
