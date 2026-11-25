// The 3D viewer: shows one job's splat with the PlayCanvas engine, starting at the job's first
// camera. mountViewer(container, sources) fills an element with the canvas and its controls.
//
// The engine is the pinned build that scripts/fetch_viewer_engine.py downloads into vendor/,
// which git ignores. Everything here is plain JavaScript with no build step.

import * as pc from './vendor/playcanvas.min.mjs';
import { CamerasError, fallbackView, parseCameras, startView } from './cameras.js';
import { WALK_KEYS, dolly, pinchLengths, turn, walk, wheelLengths } from './controls.js';
import { JOB_SPLATS, ViewerError, splatKind } from './params.js';

// Drawing at the screen's full pixel ratio costs a phone a lot for little gain on a splat, so a
// touch screen draws at most 1.5 pixels per CSS pixel, and a mouse screen at most 2.
const MAX_PIXEL_RATIO_TOUCH = 1.5;
const MAX_PIXEL_RATIO_MOUSE = 2;
// The clip planes, in scene lengths (the spread of the camera centres).
const NEAR_CLIP = 0.002;
const FAR_CLIP = 1000;
const BACKGROUND = new pc.Color(0.06, 0.06, 0.07);

const HINT_TOUCH = 'Drag to turn, pinch to zoom.';
const HINT_MOUSE = 'Drag to turn, scroll to zoom, WASD or the arrow keys to walk.';

function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text) {
        node.textContent = text;
    }
    return node;
}

// The first of the candidate splat URLs that answers. A file URL is a single candidate and is
// used as it is; a job folder gives out/clean.sog then out/train.ply, checked with HEAD.
async function pickSplat(urls) {
    if (urls.length === 1) {
        return urls[0];
    }
    for (const url of urls) {
        try {
            const answer = await fetch(url, { method: 'HEAD', credentials: 'same-origin' });
            if (answer.ok) {
                return url;
            }
        } catch {
            // Try the next one.
        }
    }
    throw new ViewerError(`This job has neither out/${JOB_SPLATS[0]} nor out/${JOB_SPLATS[1]}.`);
}

// The job's cameras, or null when there's no file or it can't be used. Then the viewer frames
// the splat on its own, so a missing cameras file never stops the splat from showing.
async function loadCameras(url) {
    if (url === null) {
        return null;
    }
    try {
        const answer = await fetch(url, { credentials: 'same-origin' });
        if (!answer.ok) {
            console.warn(`viewer: no cameras file at ${new URL(url).pathname} (${answer.status})`);
            return null;
        }
        return parseCameras(await answer.json());
    } catch (error) {
        console.warn(`viewer: the cameras file can't be used: ${error instanceof CamerasError ? error.message : error}`);
        return null;
    }
}

function loadSplat(app, url, onProgress) {
    return new Promise((resolve, reject) => {
        const asset = new pc.Asset('splat', 'gsplat', { url });
        asset.on('progress', onProgress);
        asset.once('load', () => resolve(asset));
        asset.once('error', (error) => reject(new ViewerError(`The splat didn't load: ${error}`)));
        app.assets.add(asset);
        app.assets.load(asset);
    });
}

// Fills container with the viewer. sources is { splatUrls, camerasUrl }, as params.js's
// viewerSources gives it. Returns { destroy } at once; loading goes on in the background and
// shows its progress and any error in the container.
export function mountViewer(container, sources) {
    const touch = window.matchMedia('(pointer: coarse)').matches;
    const canvas = element('canvas', 'splat-viewer-canvas');
    const status = element('p', 'splat-viewer-status', 'Loading the 3D viewer…');
    const hint = element('p', 'splat-viewer-hint', touch ? HINT_TOUCH : HINT_MOUSE);
    status.setAttribute('role', 'status');
    hint.hidden = true;
    container.classList.add('splat-viewer');
    container.tabIndex = 0;
    container.setAttribute('aria-label', `3D view. ${touch ? HINT_TOUCH : HINT_MOUSE}`);
    container.replaceChildren(canvas, status, hint);

    const cleanups = [];
    const listen = (target, type, handler, options) => {
        target.addEventListener(type, handler, options);
        cleanups.push(() => target.removeEventListener(type, handler, options));
    };
    let app = null;
    let destroyed = false;

    const show = (text) => {
        status.textContent = text;
        status.hidden = text === '';
    };

    async function start() {
        const camerasPromise = loadCameras(sources.camerasUrl);
        let device;
        try {
            device = await pc.createGraphicsDevice(canvas, {
                deviceTypes: ['webgl2'],
                antialias: false,
                alpha: false,
                powerPreference: 'high-performance',
            });
        } catch {
            throw new ViewerError("This browser can't show the 3D viewer: it needs WebGL 2.");
        }
        if (destroyed) {
            device.destroy();
            return;
        }
        device.maxPixelRatio = Math.min(window.devicePixelRatio || 1, touch ? MAX_PIXEL_RATIO_TOUCH : MAX_PIXEL_RATIO_MOUSE);

        const options = new pc.AppOptions();
        options.graphicsDevice = device;
        options.componentSystems = [pc.CameraComponentSystem, pc.GSplatComponentSystem];
        options.resourceHandlers = [pc.TextureHandler, pc.GSplatHandler];
        app = new pc.AppBase(canvas);
        app.init(options);
        app.setCanvasFillMode(pc.FILLMODE_NONE);
        app.setCanvasResolution(pc.RESOLUTION_AUTO);
        const resize = () => app.resizeCanvas(container.clientWidth, container.clientHeight);
        const sizer = new ResizeObserver(resize);
        sizer.observe(container);
        cleanups.push(() => sizer.disconnect());
        resize();

        // Draw only while the viewer is on screen, so a phone doesn't spend its battery on a
        // viewer the visitor has scrolled past.
        const onScreen = new IntersectionObserver((entries) => {
            app.autoRender = entries[entries.length - 1].isIntersecting;
            app.renderNextFrame = true;
        });
        onScreen.observe(container);
        cleanups.push(() => onScreen.disconnect());

        const camera = new pc.Entity('camera');
        camera.addComponent('camera', { clearColor: BACKGROUND });
        app.root.addChild(camera);
        app.start();

        const splatUrl = await pickSplat(sources.splatUrls);
        const kind = splatKind(splatUrl) === 'ply' ? 'trained splat (out/train.ply)' : 'splat';
        show(`Loading the ${kind}…`);
        const [cameras, asset] = await Promise.all([
            camerasPromise,
            loadSplat(app, splatUrl, (received, total) => {
                if (total > 0) {
                    show(`Loading the ${kind}… ${Math.min(100, Math.floor((received / total) * 100))}%`);
                }
            }),
        ]);
        if (destroyed) {
            return;
        }

        let view;
        if (cameras !== null) {
            view = startView(cameras);
        } else {
            const box = asset.resource?.aabb;
            const centre = box ? [box.center.x, box.center.y, box.center.z] : [0, 0, 0];
            const half = box ? [box.halfExtents.x, box.halfExtents.y, box.halfExtents.z] : [1, 1, 1];
            view = fallbackView(centre, half);
        }
        const splat = new pc.Entity('splat');
        splat.addComponent('gsplat', { asset });
        splat.setLocalRotation(new pc.Quat(...view.sceneRotation));
        app.root.addChild(splat);
        camera.camera.fov = view.fov;
        camera.camera.nearClip = Math.max(view.sceneLength * NEAR_CLIP, 1e-4);
        camera.camera.farClip = view.sceneLength * FAR_CLIP;

        const yawQ = new pc.Quat();
        const pitchQ = new pc.Quat();
        const place = () => {
            camera.setPosition(view.position[0], view.position[1], view.position[2]);
            yawQ.setFromAxisAngle(pc.Vec3.UP, view.yaw);
            pitchQ.setFromAxisAngle(pc.Vec3.RIGHT, view.pitch);
            camera.setRotation(yawQ.mul(pitchQ));
            app.renderNextFrame = true;
        };
        place();
        bindInput(view, place);
        show('');
        hint.hidden = false;
    }

    function bindInput(view, place) {
        const pointers = new Map();
        let spread = 0;
        const spreadOf = () => {
            const [a, b] = [...pointers.values()];
            return Math.hypot(a.x - b.x, a.y - b.y);
        };
        const firstInput = () => {
            hint.hidden = true;
        };
        listen(canvas, 'pointerdown', (event) => {
            if (event.pointerType === 'mouse' && event.button !== 0) {
                return;
            }
            canvas.setPointerCapture(event.pointerId);
            pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
            if (pointers.size === 2) {
                spread = spreadOf();
            }
            container.focus({ preventScroll: true });
            firstInput();
        });
        listen(canvas, 'pointermove', (event) => {
            const last = pointers.get(event.pointerId);
            if (!last) {
                return;
            }
            const dx = event.clientX - last.x;
            const dy = event.clientY - last.y;
            last.x = event.clientX;
            last.y = event.clientY;
            if (pointers.size === 1) {
                turn(view, dx, dy, canvas.clientHeight);
            } else if (pointers.size === 2) {
                const now = spreadOf();
                dolly(view, pinchLengths(now - spread, canvas.clientHeight));
                spread = now;
            }
            place();
        });
        const lift = (event) => {
            pointers.delete(event.pointerId);
            if (pointers.size === 2) {
                spread = spreadOf();
            }
        };
        listen(canvas, 'pointerup', lift);
        listen(canvas, 'pointercancel', lift);
        listen(canvas, 'wheel', (event) => {
            event.preventDefault();
            dolly(view, wheelLengths(event.deltaY, event.deltaMode, canvas.clientHeight));
            place();
            firstInput();
        }, { passive: false });

        const held = new Set();
        listen(container, 'keydown', (event) => {
            const direction = WALK_KEYS[event.code];
            if (direction && !event.altKey && !event.ctrlKey && !event.metaKey) {
                event.preventDefault();
                held.add(direction);
                firstInput();
            }
        });
        listen(container, 'keyup', (event) => {
            const direction = WALK_KEYS[event.code];
            if (direction) {
                held.delete(direction);
            }
        });
        listen(container, 'blur', () => held.clear());
        app.on('update', (dt) => {
            if (walk(view, held, Math.min(dt, 0.1))) {
                place();
            }
        });
    }

    start().catch((error) => {
        console.error(error);
        show(error instanceof ViewerError ? error.message : "The 3D viewer stopped because of an error. Try reloading the page.");
    });

    return {
        destroy() {
            destroyed = true;
            cleanups.forEach((cleanup) => cleanup());
            app?.destroy();
            app = null;
        },
    };
}
