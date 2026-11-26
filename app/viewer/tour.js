// The tour: a walk along the filmed path, the camera centres of out/cameras.json in pick order,
// looking where the phone looked. Pure maths with no engine and no DOM, so the tests run it in
// Node. Everything is in the viewer's frame (+y up), as cameras.js's viewerPoses gives it.
//
// The path:
// 1. The camera centres and view directions are smoothed over their neighbours in pick order,
//    which takes out the hand's shake.
// 2. Each step from one pick to the next gets a time: its length at walking speed, or its turn at
//    TURN_SPEED, whichever is longer, so a pan on the spot still takes time.
// 3. The path is sampled SAMPLE_RATE times a second on that clock and smoothed again over
//    TIME_SIGMA seconds, so the speed and the turning change smoothly too.
//
// Walking speed: COLMAP's scale is arbitrary, so the viewer has no metres. The tour walks at
// about a scene length (the spread of the camera centres) every PACE_SECONDS, which is about
// 0.6 m/s for a room 5 m across, a slow filming walk. The whole walk is kept between
// MIN_SECONDS and MAX_SECONDS, so a tabletop orbit isn't rushed and a long street isn't endless.

import { yawPitchOf } from './cameras.js';
import { turn } from './controls.js';

export const PACE_SECONDS = 8;
export const MIN_SECONDS = 20;
export const MAX_SECONDS = 90;
// The fastest the tour turns, in degrees a second.
export const TURN_SPEED = 30;
export const SAMPLE_RATE = 30;
// The smoothing over neighbouring picks, in picks, and over the tour's clock, in seconds.
const FRAME_SIGMA = 1.5;
const TIME_SIGMA = 0.5;

// The player's timings, in seconds.
const RAMP_IN = 0.8;
const RAMP_OUT = 1.2;
const END_MIN_RATE = 0.1;
const MIN_GLIDE = 0.6;
const MAX_GLIDE = 3;
// Pressing play this close to the end starts again from the beginning.
const RESTART_WITHIN = 0.5;
// After a drag, the look stays where it was put for LOOK_HOLD, then eases back to the path's
// heading with this time constant.
const LOOK_HOLD = 1.5;
const LOOK_RETURN = 0.8;
const MAX_PITCH = 89;

// ---- small helpers ----

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const mix = (a, b, s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function unit(v, fallback) {
    const n = Math.hypot(v[0], v[1], v[2]);
    return n > 1e-9 ? [v[0] / n, v[1] / n, v[2] / n] : fallback;
}

// The angle between two unit vectors, in degrees.
function angleBetween(a, b) {
    const c = clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1, 1);
    return (Math.acos(c) * 180) / Math.PI;
}

export function smoothstep(x) {
    const s = clamp(x, 0, 1);
    return s * s * (3 - 2 * s);
}

// The angle from a to b by the shorter way round, in degrees, from -180 to 180.
export function angleDelta(a, b) {
    return ((((b - a) % 360) + 540) % 360) - 180;
}

export function mixAngle(a, b, s) {
    return a + angleDelta(a, b) * s;
}

// A Gaussian blur of a list of 3-vectors over their index, sigma in places. Past each end the
// list is mirrored through its end point (p[-k] = 2 p[0] - p[k]), so the end points stay exactly
// where they are and a steady walk or turn keeps its full length up to the ends.
export function blur(points, sigma) {
    if (sigma <= 0 || points.length < 3) {
        return points.map((p) => [...p]);
    }
    const last = points.length - 1;
    const radius = Math.min(Math.ceil(3 * sigma), last);
    const weights = [];
    for (let k = -radius; k <= radius; k++) {
        weights.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
    }
    const total = weights.reduce((a, b) => a + b, 0);
    const at = (j) => {
        if (j < 0) {
            return points[0].map((value, c) => 2 * value - points[-j][c]);
        }
        if (j > last) {
            return points[last].map((value, c) => 2 * value - points[2 * last - j][c]);
        }
        return points[j];
    };
    return points.map((_, i) => {
        const out = [0, 0, 0];
        for (let k = -radius; k <= radius; k++) {
            const p = at(i + k);
            const w = weights[k + radius] / total;
            out[0] += p[0] * w;
            out[1] += p[1] * w;
            out[2] += p[2] * w;
        }
        return out;
    });
}

// ---- the path ----

// The tour's path from the viewer-frame poses [{ centre, forward }] and the scene length, or null
// when there's no walk to take: fewer than two picks, or every pick at the same place looking
// the same way.
//
// Returns { duration, rate, positions, forwards, speed, length }: the walk's seconds, the samples
// a second, a sample's position and unit view direction for every 1 / rate seconds from 0 (the
// last one at duration), the walking speed in scene units a second, and the path's length.
export function buildTourPath(poses, sceneLength) {
    if (!Array.isArray(poses) || poses.length < 2) {
        return null;
    }
    const centres = blur(poses.map((pose) => pose.centre), FRAME_SIGMA);
    const forwards = blur(poses.map((pose) => pose.forward), FRAME_SIGMA).map((f, i) => unit(f, poses[i].forward));

    const steps = [];
    let length = 0;
    for (let i = 1; i < centres.length; i++) {
        const step = { length: dist(centres[i - 1], centres[i]), turn: angleBetween(forwards[i - 1], forwards[i]) };
        length += step.length;
        steps.push(step);
    }
    const fit = sceneLength > 0 ? sceneLength : 1;
    const seconds = clamp((length / fit) * PACE_SECONDS, MIN_SECONDS, MAX_SECONDS);
    const speed = length > 1e-9 ? length / seconds : fit / PACE_SECONDS;

    // The clock time at each pick.
    const times = [0];
    for (const step of steps) {
        times.push(times[times.length - 1] + Math.max(step.length / speed, step.turn / TURN_SPEED));
    }
    const duration = times[times.length - 1];
    if (!(duration > 1e-3)) {
        return null;
    }

    const count = Math.ceil(duration * SAMPLE_RATE) + 1;
    let positions = [];
    let samples = [];
    let seg = 0;
    for (let k = 0; k < count; k++) {
        const t = Math.min(k / SAMPLE_RATE, duration);
        while (seg < times.length - 2 && times[seg + 1] < t) {
            seg++;
        }
        const span = times[seg + 1] - times[seg];
        const s = span > 0 ? clamp((t - times[seg]) / span, 0, 1) : 1;
        positions.push(mix(centres[seg], centres[seg + 1], s));
        samples.push(mix(forwards[seg], forwards[seg + 1], s));
    }
    positions = blur(positions, TIME_SIGMA * SAMPLE_RATE);
    samples = blur(samples, TIME_SIGMA * SAMPLE_RATE);
    let previous = forwards[0];
    const directions = samples.map((f) => {
        previous = unit(f, previous);
        return previous;
    });
    return { duration, rate: SAMPLE_RATE, positions, forwards: directions, speed, length };
}

// The path's pose at clock time t, clamped to the walk: { position, forward, yaw, pitch }.
export function sampleTour(path, t) {
    const last = path.positions.length - 1;
    const x = clamp(t, 0, path.duration) * path.rate;
    const i = Math.min(Math.floor(x), last);
    const j = Math.min(i + 1, last);
    // The last sample sits at duration, which may be less than a whole sample after the one before.
    const spanEnd = j === last ? path.duration * path.rate : j;
    const s = spanEnd > i ? clamp((x - i) / (spanEnd - i), 0, 1) : 0;
    const position = mix(path.positions[i], path.positions[j], s);
    const forward = unit(mix(path.forwards[i], path.forwards[j], s), path.forwards[i]);
    return { position, forward, ...yawPitchOf(forward) };
}

// The clock time of the path's point nearest to a position, the earliest on a tie.
export function nearestTime(path, position) {
    let best = 0;
    let bestDistance = Infinity;
    path.positions.forEach((p, k) => {
        const d = sub(p, position);
        const squared = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (squared < bestDistance) {
            bestDistance = squared;
            best = k;
        }
    });
    return Math.min(best / path.rate, path.duration);
}

// ---- the player ----

// Plays a path on a view object { position, yaw, pitch, fov } (controls.js's view). The viewer
// calls step(dt, view) every frame while playing. While it plays, look() turns the view away from
// the path's heading; anything else the visitor does should call pause(). play() picks up from
// the path's point nearest the view, gliding there from wherever the view is.
export function createTour(path) {
    let playing = false;
    let t = 0;
    let elapsed = 0;
    let glide = { from: null, seconds: 0 };
    const look = { yaw: 0, pitch: 0, idle: Infinity };

    return {
        path,
        get playing() {
            return playing;
        },
        get time() {
            return t;
        },
        play(view) {
            t = nearestTime(path, view.position);
            if (t >= path.duration - RESTART_WITHIN) {
                t = 0;
            }
            const target = sampleTour(path, t).position;
            glide = {
                from: { position: [...view.position], yaw: view.yaw, pitch: view.pitch },
                seconds: clamp(dist(view.position, target) / (2 * path.speed), MIN_GLIDE, MAX_GLIDE),
            };
            look.yaw = 0;
            look.pitch = 0;
            look.idle = Infinity;
            elapsed = 0;
            playing = true;
        },
        pause() {
            playing = false;
        },
        // A drag while playing, in pixels, the same way controls.js's turn() reads one.
        look(dx, dy, screenHeight, fov) {
            const offset = { yaw: look.yaw, pitch: look.pitch, fov };
            turn(offset, dx, dy, screenHeight);
            look.yaw = offset.yaw;
            look.pitch = offset.pitch;
            look.idle = 0;
        },
        // Moves the tour on by dt seconds and writes its pose into view. Returns true when the
        // walk reached its end and stopped.
        step(dt, view) {
            if (!playing) {
                return false;
            }
            elapsed += dt;
            const rate = smoothstep(elapsed / RAMP_IN) * Math.max(END_MIN_RATE, smoothstep((path.duration - t) / RAMP_OUT));
            t = Math.min(path.duration, t + dt * rate);
            look.idle += dt;
            if (look.idle > LOOK_HOLD) {
                const keep = Math.exp(-dt / LOOK_RETURN);
                look.yaw *= keep;
                look.pitch *= keep;
            }
            const pose = sampleTour(path, t);
            const yaw = pose.yaw + look.yaw;
            const pitch = clamp(pose.pitch + look.pitch, -MAX_PITCH, MAX_PITCH);
            const g = glide.from === null ? 1 : smoothstep(elapsed / glide.seconds);
            if (g >= 1) {
                glide.from = null;
            }
            const from = glide.from ?? { position: pose.position, yaw, pitch };
            view.position = mix(from.position, pose.position, g);
            view.yaw = mixAngle(from.yaw, yaw, g);
            view.pitch = from.pitch + (pitch - from.pitch) * g;
            if (t >= path.duration) {
                playing = false;
                return true;
            }
            return false;
        },
    };
}
