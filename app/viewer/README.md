# The 3D viewer

A web viewer for one finished job's splat. It opens `out/clean.sog`, or `out/train.ply` when the job has no clean copy, and starts at the job's first camera in `out/cameras.json`, with up pointing up.

- **Laptop:** drag to turn, scroll (or pinch on a trackpad) to zoom, WASD or the arrow keys to walk, and Q and E to go down and up. Space or the **Tour** button plays and pauses the tour.
- **Phone:** drag with one finger to turn, and pinch to zoom. The **Tour** button plays and pauses the tour.

Until the splat is on screen, a loading screen shows how much of the file has arrived, as a bar, a percentage and megabytes. With no `Content-Length` from the host, it shows only the megabytes and a sliding bar. Once the download is complete, it says the splat is unpacking while the engine decodes it.

## The tour

When the job has a usable `out/cameras.json` with at least two cameras, the **Tour** button walks the filmed path: the camera centres in pick order, looking where the phone looked, with no roll.

- **Smooth:** the centres and view directions are smoothed over their neighbouring picks, which takes out the hand's shake. The path is then smoothed again over half a second of the tour's clock, so speed and turning change gently. The tour speeds up over its first 0.8 s and slows into its end.
- **Walking speed:** COLMAP's scale is arbitrary, so the viewer has no metres. The tour covers about one scene length (the diagonal of the camera centres' box) every 8 s, which is about 0.6 m/s for a room 5 m across. The whole walk takes between 20 s and 90 s, and it never turns faster than 30° a second, so a pan on the spot still takes time. These numbers are at the top of `tour.js`.
- **Looking around:** a one-finger or mouse drag while the tour plays turns the view without stopping the walk. 1.5 s after the last drag, the view eases back to the path's heading.
- **Taking over:** walking (WASD or the arrows), the wheel, a trackpad scroll or a pinch stops the tour and leaves the camera where it is.
- **Picking up again:** play starts from the path's point nearest the camera and glides onto the path from wherever the camera is, over 0.6 s to 3 s depending on the distance. Within the last 0.5 s of the path, play starts again from the beginning.

## Haunted mode

A Halloween easter egg, off by default and never saved. Press H while the viewer has focus, or tap the Voxelis logo in the top left three times within a second, to turn it on or off. It dims the room, with a cold vignette at the edges, and sends a faint ghost along the tour's path at the tour's pace, looping. The ghost starts at the point of the path nearest the middle of the view that's at least 0.3 scene lengths in front of the camera. It's an SVG over the canvas, placed each frame where its point shows on screen, so the splat never hides it. With `prefers-reduced-motion`, it glides without bobbing. A job with no tour has no ghost, but the room still dims.

It's plain HTML, CSS and JavaScript with no build step and no iframe. It runs on the PlayCanvas engine 2.24.3 (MIT), drawn with WebGL 2. Engine 2.11.0 was the first to read the SOG version 2 that splat-transform 3.8.0 writes, 2.11.1 to 2.11.7 fixed it, and 2.24.0 added reading each file's anti-aliasing flag. The engine also reads SOG version 3, but not the old version 3 layout of splat-transform 3.11.0 and 3.11.1, so if splat-transform ever moves past 3.8.0, skip those two.

## Pumpkin smash

A Halloween game for any filmed place, played like a PC game on a laptop. Glowing pumpkins float in the room, and the ones that see you come after you. Smash them with candy corn, and don't hit the lanterns. It needs the tour's path, so a job with no tour has no game.

- **Starting and ending:** the **Pumpkin smash** button above the Tour button, or P. It stops the tour, turns haunted mode off and starts a 60-second round. The score, a bar for how fired up you are and the time left are at the top right, and the controls show along the bottom for the first few seconds. When the time is up, a panel shows the score and **Play again**. The button (now **End smash**), P or Esc ends the game and puts the room back as it was, haunted mode included. H and the Tour button do nothing while a round runs.
- **Laptop controls:** the mouse is locked to the view, so moving it looks around, and a click throws at the aim dot in the middle. WASD walk, Space jumps, G switches between walking and flying (Q and E go down and up), V switches between your own eyes and a camera over your shoulder, and M mutes the sounds. Esc frees the mouse and pauses the clock: a click carries on, and Esc again or End smash ends the game. Where the browser won't lock the mouse, a click that doesn't drag throws where it lands.
- **Phone controls:** drag to look, tap to throw where you tap, and pinch to move.
- **Scoring:** a smashed pumpkin scores 1, a broken lantern costs 2 and a pumpkin that catches you costs 1, so the score can go below 0. The end panel lists all three.
- **The chase:** a pumpkin starts chasing once it's on screen with nothing in the way, or once you're within 0.25 scene lengths. It glows hotter, waits 0.8 s and comes after you around the walls, slow at first and quicker as the round goes on, from 0.06 to 0.14 scene lengths a second against your walk of 0.4. One that reaches you bursts, and the edges of the view glow orange.
- **The lanterns:** 3 at a time, iron lanterns with glowing amber glass, made in code. One hit shatters it into shards, and a new one appears somewhere else 2.5 s later.
- **Walls and physics:** the splat's centres are counted into a grid of cubes, and a cube with enough splats in it is solid. You bump into walls and things in the room, and the pumpkins go around them. Floaters and fog don't count: a cube needs at least 3 splats and 10% of the mean count of the cubes with any, a clump of fewer than 12 solid cubes is dropped, and everything within 0.05 scene lengths of where the phone went is open. Walking, gravity holds you to the floor, you step up onto anything lower than 30% of your height, and the phone's height above the floor under the filmed path sets your eye height. With no floor found, you fly. Splats only show surfaces, so the inside of something filmed all round counts as open. A candy corn stops at the first wall, so a pumpkin behind one can't be hit.
- **The light:** the splat itself is lit through the engine's `gsplatModifyVS` shader chunk on the gsplat template material. Calm, the room is a cool dim blue, keeping about 45% of its colour. The more you throw and smash, the brighter and warmer it gets, up to about 70%, and it calms down over about 5 s. A lantern or being caught knocks it back. A warm light follows you and grows with it, each pumpkin throws flickering orange light on the walls and the lanterns throw a steady amber one. Haunted mode's canvas filter isn't used, because it would dim the pumpkins too.
- **You:** a person made in code, in an orange hoodie, jeans and white trainers, built from rounded parts joined at the shoulders, elbows, wrists, hips, knees and ankles. They walk, jump, fly and throw overarm. You see them over your shoulder with V. The camera stays level with your head, so it doesn't rise out of a room whose ceiling wasn't filmed, and it stops short of any splats behind you.
- **The pumpkins:** 6 at a time, made in code with no model or image files: 8 ribbed wedges, a stem and a glowing carved face. They bob and sway, and they're real meshes in the scene, so a wall in front of one hides it. Each one appears near the filmed path, short of the first wall and clear of everything solid, spread out from the others, at least 0.35 scene lengths from you and far enough from the camera to be at most a quarter of the view's height. One that isn't chasing is never drawn filling the view. Sizes and distances are shares of the scene length the tour uses: a pumpkin's radius is 0.035 of it.
- **A smash:** the wedges and the stem fly apart with spin and gravity, wrapped sweets and sparkles spray out, the pumpkin's light flashes and goes out, and everything is gone within 1.5 s. A new pumpkin appears somewhere else a second later.
- **Sound:** a whoosh for a throw, a splat for a smash, a tinkle of glass for a lantern and a bonk when you're caught, all made with Web Audio in code, and a mute button at the top right.
- **Phones:** each shape is one instanced draw. The pieces, sweets, sparkles and shards come from pools made once, with at most 300 sweets, sparkles and shards at a time. With `prefers-reduced-motion`, the pumpkins don't bob or sway and their light doesn't flicker.
## Files

| File | What it is |
|---|---|
| `viewer.js` | `mountViewer(container, sources)`: the engine, the canvas, loading and the controls |
| `params.js` | Which files to load, from a URL parameter or from data attributes |
| `cameras.js` | Reads `out/cameras.json` and works out the starting camera and every camera in the viewer's frame |
| `controls.js` | How a drag, a pinch, the wheel and the keys move the camera |
| `tour.js` | The tour's path from the cameras, and the player that walks it |
| `haunt.js` | Haunted mode: the dimming, the ghost and the logo's three taps |
| `smash.js` | Pumpkin smash: its button, the light, the meshes, the mouse lock, input, and the round on screen |
| `smash-play.js` | Pumpkin smash's rules and motion: the round's clock, score and penalties, the throw and what it smashes, the pieces' gravity, spin and fading, the pools, and the light that follows how fired up you are |
| `smash-place.js` | The grid of open and solid space from the splat's centres, without floaters and fog, and where pumpkins and lanterns go |
| `smash-physics.js` | Your body bumping into walls, standing, stepping, jumping and flying, the pumpkins' chase, line of sight, mouse look and the camera over your shoulder |
| `smash-human.js` | The person you play, made in code, and their walk, jump, flight and throw |
| `smash-meshes.js` | The pumpkin's wedges, stem and face, the candy corn, the sweet, the sparkle, the lantern and its shards, made in code |
| `smash-sound.js` | The throw, smash, lantern and caught sounds, made with Web Audio |
| `smash.css` | Pumpkin smash's score, fury bar, clock, mute button, aim dot, pause and end panels. `smash.js` links it from beside itself |
| `loading.js` | The loading screen's percentage and megabytes |
| `embed.js` | Starts a viewer in every element with a `data-splat-url`, for the website |
| `page.js`, `index.html` | The standalone page, for testing on a laptop |
| `viewer.css` | Styles only `.splat-viewer`, what's inside it and the standalone page, and pumpkin smash's button |
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

The website does all of this: its handler serves the files under `/viewer/`, and under `/shared/<token>/viewer/` for a share link, and its result page fills the Viewer_Slot.

### Its files

The website serves these from the bundle, read into `Deps.static` like `app.js`:

| Route | File | Content-Type |
|---|---|---|
| `GET /viewer/viewer.css`, `/viewer/smash.css` | the same names in `app/viewer/` | `text/css; charset=utf-8` |
| `GET /viewer/embed.js` | `app/viewer/embed.js` | `text/javascript; charset=utf-8` |
| `GET /viewer/viewer.js`, `/viewer/params.js`, `/viewer/cameras.js`, `/viewer/controls.js`, `/viewer/tour.js`, `/viewer/haunt.js`, `/viewer/loading.js`, `/viewer/smash.js`, `/viewer/smash-play.js`, `/viewer/smash-place.js`, `/viewer/smash-meshes.js`, `/viewer/smash-sound.js`, `/viewer/smash-physics.js`, `/viewer/smash-human.js` | the same names in `app/viewer/` | `text/javascript; charset=utf-8` |
| `GET /viewer/vendor/playcanvas.min.mjs` | the engine, 2,895,086 bytes | `text/javascript; charset=utf-8` |
| `GET /viewer/vendor/PLAYCANVAS-LICENSE.txt` | the engine's MIT licence, which its terms ask us to keep with it | `text/plain; charset=utf-8` |

- The scripts are ES modules, and browsers refuse a module with any other Content-Type under `nosniff`. They import each other by relative path, so they must sit side by side under `/viewer/`, with the engine under `/viewer/vendor/`.
- `index.html` and `page.js` are only for the laptop. The website doesn't need to serve them.
- Every other file here is in `VIEWER_FILES` in `app/handler.py`, which serves them under `/viewer/` and `/shared/<token>/viewer/`. `tests/test_viewer.py` checks that every file a viewer module imports or links is in it.
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
- There's no inline script or style, no `eval` and no WebAssembly, so nothing else is needed. The engine sets the canvas's size through `element.style`, and the viewer moves the loading bar and the ghost the same way, which the policy doesn't block. The ghost's SVG is built with `createElementNS`, with no `style` attributes. Pumpkin smash adds a `<link>` to `smash.css` on our own origin, which `style-src 'self'` allows, and Web Audio needs nothing from the policy.

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
