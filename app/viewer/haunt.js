// Haunted mode, the Halloween easter egg. Off by default: H, or three taps on the logo, turns it
// on and off. It dims the room and sends a faint ghost down the tour's path at walking speed.
//
// The ghost is an SVG over the canvas, placed each frame where its point on the path shows on
// screen, so it needs no extra engine parts and the page's CSP stays as it is. The splat doesn't
// hide it, which suits a ghost.

import { forwardOf } from './cameras.js';
import { sampleTour, smoothstep } from './tour.js';

// The ghost's height, in scene lengths, and the nearest it starts to the camera, in scene lengths.
const GHOST_SIZE = 0.08;
const GHOST_MIN_START = 0.3;
const GHOST_OPACITY = 0.45;
// It bobs up and down by this much, in scene lengths, this many times a second.
const BOB_HEIGHT = 0.008;
const BOB_RATE = 0.5;
// It fades in and out over this many seconds at each end of the path, and fades away when it's
// closer than this many scene lengths.
const END_FADE = 1;
const NEAR_FADE = 0.15;
const TAP_COUNT = 3;
const TAP_WINDOW_MS = 1000;
const SVG = 'http://www.w3.org/2000/svg';

// Where a viewer-frame point shows on a width by height screen for a view { position, yaw, pitch,
// fov } with a vertical field of view, as { x, y, depth } in pixels from the top left and scene
// units in front of the camera, or null when it's behind the camera.
export function projectPoint(view, point, width, height) {
    const y = (view.yaw * Math.PI) / 180;
    const f = forwardOf(view.yaw, view.pitch);
    const r = [Math.cos(y), 0, -Math.sin(y)];
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    const d = [point[0] - view.position[0], point[1] - view.position[1], point[2] - view.position[2]];
    const depth = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
    if (depth <= 1e-9) {
        return null;
    }
    const half = Math.tan((view.fov * Math.PI) / 360);
    const aspect = width / Math.max(height, 1);
    const sx = (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / (depth * half * aspect);
    const sy = (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) / (depth * half);
    return { x: ((sx + 1) / 2) * width, y: ((1 - sy) / 2) * height, depth };
}

// Where on the path the ghost starts, as the tour's clock time: the point nearest the middle of
// the view that's at least GHOST_MIN_START scene lengths in front of the camera, the earliest on
// a tie. With no such point, the point furthest in front.
export function ghostStart(path, view) {
    const f = forwardOf(view.yaw, view.pitch);
    const nearest = GHOST_MIN_START * view.sceneLength;
    let best = 0;
    let bestScore = -Infinity;
    let furthest = 0;
    let furthestDepth = -Infinity;
    path.positions.forEach((p, k) => {
        const d = [p[0] - view.position[0], p[1] - view.position[1], p[2] - view.position[2]];
        const depth = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
        if (depth > furthestDepth) {
            furthestDepth = depth;
            furthest = k;
        }
        const score = depth / Math.max(Math.hypot(d[0], d[1], d[2]), 1e-12);
        if (depth >= nearest && score > bestScore) {
            bestScore = score;
            best = k;
        }
    });
    const k = bestScore === -Infinity ? furthest : best;
    return Math.min(k / path.rate, path.duration);
}

// Counts taps: returns a function of the time in milliseconds that's true on the third tap within
// a second, then starts counting again.
export function tapCounter(count = TAP_COUNT, windowMs = TAP_WINDOW_MS) {
    let taps = [];
    return (now) => {
        taps = taps.filter((time) => now - time < windowMs);
        taps.push(now);
        if (taps.length >= count) {
            taps = [];
            return true;
        }
        return false;
    };
}

function ghostSvg() {
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '-10 0 120 120');
    svg.setAttribute('aria-hidden', 'true');
    const body = document.createElementNS(SVG, 'path');
    body.setAttribute('d', 'M10 110 V50 A40 40 0 0 1 90 50 V110 L77 100 L64 110 L50 100 L36 110 L23 100 Z');
    body.setAttribute('class', 'splat-viewer-ghost-body');
    svg.append(body);
    for (const cx of [37, 63]) {
        const eye = document.createElementNS(SVG, 'ellipse');
        eye.setAttribute('cx', String(cx));
        eye.setAttribute('cy', '52');
        eye.setAttribute('rx', '6');
        eye.setAttribute('ry', '9');
        eye.setAttribute('class', 'splat-viewer-ghost-eye');
        svg.append(eye);
    }
    return svg;
}

// Adds haunted mode's layers to the viewer's container. Returns { on, toggle(view), update(dt,
// view, width, height), setPath(path, view) }. toggle() and update() take the camera's view.
export function createHaunt(container, announce) {
    const veil = document.createElement('div');
    veil.className = 'splat-viewer-veil';
    const ghost = document.createElement('div');
    ghost.className = 'splat-viewer-ghost';
    ghost.append(ghostSvg());
    ghost.hidden = true;
    container.append(veil, ghost);

    // With reduced motion, the ghost glides without bobbing.
    const still = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    let on = false;
    let path = null;
    let clock = 0;
    let age = 0;

    const start = (view) => {
        clock = path === null ? 0 : ghostStart(path, view);
        age = 0;
    };

    return {
        get on() {
            return on;
        },
        // The tour's path, once the splat has loaded. A ghost already on starts from the view.
        setPath(value, view) {
            path = value;
            if (on && view) {
                start(view);
            }
        },
        toggle(view) {
            on = !on;
            container.classList.toggle('is-haunted', on);
            if (on && view) {
                start(view);
            }
            if (!on) {
                ghost.hidden = true;
            }
            announce(on ? 'Haunted mode on. Press H to turn it off.' : 'Haunted mode off.');
        },
        update(dt, view, width, height) {
            if (!on || path === null || view === null) {
                return;
            }
            clock += dt;
            age += dt;
            if (clock > path.duration) {
                clock = 0;
            }
            const pose = sampleTour(path, clock);
            const bob = still ? 0 : Math.sin(age * BOB_RATE * 2 * Math.PI) * BOB_HEIGHT * view.sceneLength;
            const point = [pose.position[0], pose.position[1] + bob, pose.position[2]];
            const spot = projectPoint(view, point, width, height);
            if (spot === null) {
                ghost.hidden = true;
                return;
            }
            const fade = smoothstep(clock / END_FADE) * smoothstep((path.duration - clock) / END_FADE)
                * smoothstep(spot.depth / (NEAR_FADE * view.sceneLength)) * smoothstep(age / END_FADE);
            const half = Math.tan((view.fov * Math.PI) / 360);
            const size = Math.min(height * 0.8, (GHOST_SIZE * view.sceneLength * height) / (2 * spot.depth * half));
            ghost.hidden = fade <= 0.01;
            ghost.style.opacity = String(GHOST_OPACITY * fade);
            ghost.style.transform = `translate(${spot.x - size / 2}px, ${spot.y - size / 2}px) scale(${size / 100})`;
        },
    };
}
