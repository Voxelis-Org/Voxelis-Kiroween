// Pumpkin smash's rules and motion: the round's clock and score, where a throw goes and which
// pumpkin it smashes, the candy corn's flight, and the gravity, spin and fading of the pieces,
// sweets and sparkles. Pure maths with no engine and no DOM, so the tests run it in Node.
// Everything is in the viewer's frame (+y up). Sizes and speeds are in pumpkin radii, and the
// pumpkin's radius is a share of the scene length, so they all scale with the scene.
//
// The flying bits are written fresh into objects the pool already holds, never made new each
// frame, so a phone's garbage collector has nothing to do while a round runs.

import { forwardOf } from './cameras.js';

export const ROUND_SECONDS = 60;
export const PUMPKIN_COUNT = 6;
// A smashed pumpkin's place is filled this long after it bursts, in seconds.
export const RESPAWN_SECONDS = 1;
// The pumpkin's radius, in scene lengths.
export const PUMPKIN_RADIUS = 0.035;

// The throw: it starts this far in front of the camera, in scene lengths, low in the view, and
// reaches where it was aimed in THROW_SECONDS. A throw that passes no pumpkin flies MISS_DISTANCE
// scene lengths and fades.
export const THROW_SECONDS = 0.2;
const THROW_START = 0.06;
const THROW_DROP = 0.7;
export const MISS_DISTANCE = 0.8;
// The candy corn's length, in pumpkin radii, and how high its flight arcs, as a share of its length.
export const CANDY_SIZE = 0.55;
const THROW_ARC = 0.04;

// The burst, in pumpkin radii and seconds. Everything is gone PIECE_SECONDS after the smash.
export const PIECE_SECONDS = 1.5;
export const MAX_PARTICLES = 300;
export const SWEETS_PER_SMASH = 18;
export const SPARKLES_PER_SMASH = 36;
// Lighter than real gravity, so the pieces stay in view while they fade.
export const GRAVITY = 16;
export const SWEET_SIZE = 0.24;
export const SPARKLE_SIZE = 0.24;
const PIECE_FADE = 0.5;
const SWEET_FADE = 0.4;

export const SWEET_COLOURS = [
    [0.95, 0.2, 0.26], [0.62, 0.32, 0.92], [0.3, 0.85, 0.38],
    [1.0, 0.84, 0.2], [1.0, 0.45, 0.76], [0.32, 0.6, 1.0],
];
const SPARKLE_COLOURS = [[1.0, 0.86, 0.42], [1.0, 0.97, 0.9], [1.0, 0.62, 0.22]];

// ---- small vector and quaternion helpers; quaternions are [x, y, z, w] ----

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export function unit(v, fallback = [0, 1, 0]) {
    const n = Math.hypot(v[0], v[1], v[2]);
    return n > 1e-12 ? [v[0] / n, v[1] / n, v[2] / n] : [...fallback];
}

// The Hamilton product a b: turning by b, then by a.
export function quatMul(a, b, out = [0, 0, 0, 1]) {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    out[0] = aw * bx + ax * bw + ay * bz - az * by;
    out[1] = aw * by - ax * bz + ay * bw + az * bx;
    out[2] = aw * bz + ax * by - ay * bx + az * bw;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
}

// A turn by angle radians about a unit axis.
export function quatAxis(axis, angle, out = [0, 0, 0, 1]) {
    const s = Math.sin(angle / 2);
    out[0] = axis[0] * s;
    out[1] = axis[1] * s;
    out[2] = axis[2] * s;
    out[3] = Math.cos(angle / 2);
    return out;
}

// v turned by the unit quaternion q, the same sum as the shader's.
export function turnVector(q, v, out = [0, 0, 0]) {
    const [x, y, z, w] = q;
    const ix = y * v[2] - z * v[1] + w * v[0];
    const iy = z * v[0] - x * v[2] + w * v[1];
    const iz = x * v[1] - y * v[0] + w * v[2];
    out[0] = v[0] + 2 * (y * iz - z * iy);
    out[1] = v[1] + 2 * (z * ix - x * iz);
    out[2] = v[2] + 2 * (x * iy - y * ix);
    return out;
}

export function smoothstep(x) {
    const s = clamp(x, 0, 1);
    return s * s * (3 - 2 * s);
}

// ---- the round ----

// A round's clock and score. state is 'idle' before the first start and after end(), 'playing'
// while the clock runs, and 'over' once it has run out, until start() or end().
export function createRound(seconds = ROUND_SECONDS) {
    let state = 'idle';
    let timeLeft = seconds;
    let score = 0;
    return {
        get state() {
            return state;
        },
        get timeLeft() {
            return timeLeft;
        },
        get score() {
            return score;
        },
        start() {
            state = 'playing';
            timeLeft = seconds;
            score = 0;
        },
        // Moves the clock on by dt seconds. Returns true on the step where the time runs out.
        step(dt) {
            if (state !== 'playing') {
                return false;
            }
            timeLeft = Math.max(0, timeLeft - Math.max(0, dt));
            if (timeLeft === 0) {
                state = 'over';
                return true;
            }
            return false;
        },
        // A smashed pumpkin. Counts only while the clock runs; returns whether it counted.
        smash() {
            if (state !== 'playing') {
                return false;
            }
            score += 1;
            return true;
        },
        end() {
            state = 'idle';
        },
    };
}

// The time left as the clock shows it, m:ss, rounded up so it shows 1:00 at the start and 0:00
// only once the time is up.
export function clockText(seconds) {
    const whole = Math.max(0, Math.ceil(seconds - 1e-9));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

// ---- aiming and the throw ----

// The camera's right and up for a view { yaw, pitch } in degrees, the same as haunt.js's
// projectPoint uses, so a point and its pixel go both ways.
function cameraAxes(view) {
    const y = (view.yaw * Math.PI) / 180;
    const forward = forwardOf(view.yaw, view.pitch);
    const right = [Math.cos(y), 0, -Math.sin(y)];
    return { forward, right, up: cross(right, forward) };
}

// The ray from the camera through pixel (x, y) of a width by height view, from the top left:
// { origin, direction }, with a unit direction.
export function screenRay(view, x, y, width, height) {
    const { forward, right, up } = cameraAxes(view);
    const half = Math.tan((view.fov * Math.PI) / 360);
    const aspect = width / Math.max(height, 1);
    const sx = ((2 * x) / Math.max(width, 1) - 1) * half * aspect;
    const sy = (1 - (2 * y) / Math.max(height, 1)) * half;
    const direction = unit([
        forward[0] + right[0] * sx + up[0] * sy,
        forward[1] + right[1] * sx + up[1] * sy,
        forward[2] + right[2] * sx + up[2] * sy,
    ]);
    return { origin: [...view.position], direction };
}

// The pumpkin a throw smashes: the nearest one whose middle the ray passes within its radius of,
// as { index, distance }, with the distance along the ray to where the ray enters it. targets is
// [{ centre, radius }], with null for an empty place. Pumpkins behind the start, or further than
// maxDistance, don't count. null when the throw passes every pumpkin.
export function throwHit(origin, direction, targets, maxDistance = Infinity) {
    let best = null;
    targets.forEach((target, index) => {
        if (!target) {
            return;
        }
        const d = [target.centre[0] - origin[0], target.centre[1] - origin[1], target.centre[2] - origin[2]];
        const along = dot(d, direction);
        const missSquared = dot(d, d) - along * along;
        const radiusSquared = target.radius * target.radius;
        if (along < 0 || missSquared > radiusSquared) {
            return;
        }
        const distance = Math.max(0, along - Math.sqrt(radiusSquared - missSquared));
        if (distance <= maxDistance && (best === null || distance < best.distance)) {
            best = { index, distance };
        }
    });
    return best;
}

// Where the candy corn starts: just below the middle of the view, a little in front of the camera.
export function throwStart(view) {
    const { forward, up } = cameraAxes(view);
    const ahead = THROW_START * view.sceneLength;
    const drop = ahead * Math.tan((view.fov * Math.PI) / 360) * THROW_DROP;
    return [0, 1, 2].map((c) => view.position[c] + forward[c] * ahead - up[c] * drop);
}

// A throw from start to target, arcing a little upward, over seconds.
export function createThrow(start, target, seconds = THROW_SECONDS) {
    const length = Math.hypot(target[0] - start[0], target[1] - start[1], target[2] - start[2]);
    return { start: [...start], target: [...target], seconds, arc: length * THROW_ARC };
}

// The candy corn at t seconds into its throw: { position, direction, done }, direction being the
// way it's moving, which its tip points along.
export function throwAt(flight, t, out = { position: [0, 0, 0], direction: [0, 0, 1], done: false }) {
    const s = clamp(t / flight.seconds, 0, 1);
    const lift = 4 * flight.arc * s * (1 - s);
    const climb = (4 * flight.arc * (1 - 2 * s)) / flight.seconds;
    for (let c = 0; c < 3; c++) {
        out.position[c] = flight.start[c] + (flight.target[c] - flight.start[c]) * s;
        out.direction[c] = (flight.target[c] - flight.start[c]) / flight.seconds;
    }
    out.position[1] += lift;
    out.direction[1] += climb;
    const n = Math.hypot(...out.direction) || 1;
    for (let c = 0; c < 3; c++) {
        out.direction[c] /= n;
    }
    out.done = t >= flight.seconds;
    return out;
}

// The turn that points +z along a unit direction, with a roll of angle radians about it.
export function pointAlong(direction, roll, out = [0, 0, 0, 1]) {
    const z = [0, 0, 1];
    const c = clamp(dot(z, direction), -1, 1);
    const axis = c < -0.999999 ? [0, 1, 0] : unit(cross(z, direction), [0, 1, 0]);
    const aim = quatAxis(axis, Math.acos(c));
    return quatMul(aim, quatAxis(z, roll), out);
}

// ---- the pool ----

// capacity objects made once by make(), handed out by spawn() and given back by retire(). When
// every one is out, spawn() takes back the one handed out longest ago, so there are never more
// than capacity at a time.
export function createPool(capacity, make) {
    const items = Array.from({ length: capacity }, () => ({ ...make(), alive: false, born: 0 }));
    return {
        items,
        capacity,
        spawn(now) {
            let pick = null;
            for (const item of items) {
                if (!item.alive) {
                    pick = item;
                    break;
                }
                if (pick === null || item.born < pick.born) {
                    pick = item;
                }
            }
            pick.alive = true;
            pick.born = now;
            return pick;
        },
        retire(item) {
            item.alive = false;
        },
        count() {
            return items.reduce((n, item) => n + (item.alive ? 1 : 0), 0);
        },
        clear() {
            items.forEach((item) => {
                item.alive = false;
            });
        },
    };
}

// What every flying bit holds: a piece of a pumpkin, a sweet or a sparkle.
export function blankBit() {
    return {
        kind: '',
        position: [0, 0, 0],
        velocity: [0, 0, 0],
        turn: [0, 0, 0, 1],
        spinAxis: [0, 1, 0],
        spinRate: 0,
        gravity: 0,
        size: 1,
        life: 1,
        fade: 1,
        twinkle: 0,
        tint: [1, 1, 1],
    };
}

// ---- the pieces' motion ----

// How much of a bit is left at age seconds: 1 until fade seconds before the end of its life, then
// smoothly down to 0 at the end, and 0 after.
export function fadeOf(age, life, fade) {
    if (age >= life) {
        return 0;
    }
    return smoothstep((life - age) / Math.max(fade, 1e-6));
}

// Where a bit is at age seconds after it was launched, written into out { position, turn, size }.
// The motion has a closed form, so it's exact at any frame rate:
//   position = start + velocity age - (gravity age^2 / 2) up
//   turn     = (a turn of spinRate age about spinAxis) start turn
//   size     = size x fadeOf(age), twinkling for a sparkle
export function bitAt(bit, age, out) {
    for (let c = 0; c < 3; c++) {
        out.position[c] = bit.position[c] + bit.velocity[c] * age;
    }
    out.position[1] -= 0.5 * bit.gravity * age * age;
    quatMul(quatAxis(bit.spinAxis, bit.spinRate * age, out.spin), bit.turn, out.turn);
    const twinkle = bit.twinkle > 0 ? 0.65 + 0.35 * Math.cos(bit.twinkle * age) : 1;
    out.size = bit.size * fadeOf(age, bit.life, bit.fade) * twinkle;
    return out;
}

export function blankPose() {
    return { position: [0, 0, 0], turn: [0, 0, 0, 1], spin: [0, 0, 0, 1], size: 0 };
}

// A part of a whole pumpkin { centre, turn, radius }, written into out { position, turn, size }:
// part is one of smash-meshes.js's pumpkinParts().
export function partPose(pumpkin, part, out) {
    turnVector(pumpkin.turn, part.offset, out.position);
    for (let c = 0; c < 3; c++) {
        out.position[c] = pumpkin.centre[c] + out.position[c] * pumpkin.radius;
    }
    quatMul(pumpkin.turn, part.turn, out.turn);
    out.size = pumpkin.radius;
    return out;
}

const between = (random, lo, hi) => lo + (hi - lo) * random();

function copy(target, values) {
    for (let i = 0; i < target.length; i++) {
        target[i] = values[i];
    }
}

function randomDirection(random) {
    const y = between(random, -1, 1);
    const a = between(random, 0, 2 * Math.PI);
    const r = Math.sqrt(1 - y * y);
    return [r * Math.cos(a), y, r * Math.sin(a)];
}

// Sends a smashed pumpkin { centre, turn, radius }'s parts flying from where they sat: each wedge
// outward and up, tumbling over its outer edge, and the stem up. spawn(now) gives a bit to fill.
export function launchParts(pumpkin, parts, random, now, spawn) {
    const r = pumpkin.radius;
    const pose = blankPose();
    for (const part of parts) {
        const bit = spawn(now);
        partPose(pumpkin, part, pose);
        bit.kind = part.kind;
        copy(bit.position, pose.position);
        copy(bit.turn, pose.turn);
        const outward = part.kind === 'stem'
            ? unit([between(random, -1, 1), 0, between(random, -1, 1)], [1, 0, 0])
            : unit(turnVector(pumpkin.turn, [part.offset[0], 0, part.offset[2]]), [1, 0, 0]);
        const out = part.kind === 'stem' ? between(random, 0.5, 1.5) : between(random, 3, 5.5);
        const upward = part.kind === 'stem' ? between(random, 6, 8) : between(random, 2.5, 5.5);
        copy(bit.velocity, [outward[0] * out * r, upward * r, outward[2] * out * r]);
        copy(bit.spinAxis, unit([
            -outward[2] + between(random, -0.3, 0.3),
            between(random, -0.3, 0.3),
            outward[0] + between(random, -0.3, 0.3),
        ], [1, 0, 0]));
        bit.spinRate = between(random, 3, 9) * (random() < 0.5 ? -1 : 1);
        bit.gravity = GRAVITY * r;
        bit.size = r;
        bit.life = PIECE_SECONDS;
        bit.fade = PIECE_FADE;
        bit.twinkle = 0;
        copy(bit.tint, [1, 1, 1]);
    }
}

// Sprays wrapped sweets and sparkles out of a smashed pumpkin's middle. spawn(now) gives a bit to
// fill; the sweets fly up and fall, and the sparkles fly faster, barely fall and go out sooner.
export function launchSweets(centre, radius, random, now, spawn, sweets = SWEETS_PER_SMASH, sparkles = SPARKLES_PER_SMASH) {
    for (let n = 0; n < sweets + sparkles; n++) {
        const sweet = n < sweets;
        const bit = spawn(now);
        const d = randomDirection(random);
        if (sweet) {
            d[1] = Math.abs(d[1]) * 0.8 + 0.2;
        }
        const dir = unit(d);
        const speed = sweet ? between(random, 4, 9) : between(random, 6, 13);
        bit.kind = sweet ? 'sweet' : 'sparkle';
        for (let c = 0; c < 3; c++) {
            bit.position[c] = centre[c] + dir[c] * radius * 0.35;
            bit.velocity[c] = dir[c] * speed * radius;
        }
        copy(bit.turn, [0, 0, 0, 1]);
        copy(bit.spinAxis, randomDirection(random));
        bit.spinRate = between(random, 5, 14);
        bit.gravity = GRAVITY * radius * (sweet ? 1 : 0.25);
        bit.size = radius * (sweet ? SWEET_SIZE : SPARKLE_SIZE) * between(random, 0.8, 1.2);
        bit.life = sweet ? between(random, 1.1, PIECE_SECONDS) : between(random, 0.5, 1.0);
        bit.fade = sweet ? SWEET_FADE : bit.life;
        bit.twinkle = sweet ? 0 : between(random, 18, 30);
        const palette = sweet ? SWEET_COLOURS : SPARKLE_COLOURS;
        copy(bit.tint, palette[Math.floor(random() * palette.length) % palette.length]);
    }
}

// ---- the light ----

// A candle's flicker for a pumpkin's light at time t seconds, with the pumpkin's own phase, from
// about 0.75 to 1.05: three slow waves on top of each other, so it never repeats in a way you'd see.
export function flicker(t, phase) {
    const tau = 2 * Math.PI;
    return 0.9
        + 0.07 * Math.sin(tau * (1.3 * t + phase))
        + 0.05 * Math.sin(tau * (3.7 * t + 2.1 * phase))
        + 0.03 * Math.sin(tau * (8.9 * t + 4.3 * phase));
}

// A smashed pumpkin's light flashes to FLASH and goes out over FLASH_SECONDS.
export const FLASH = 3;
export const FLASH_SECONDS = 0.6;
export function flashOf(age) {
    if (age < 0 || age >= FLASH_SECONDS) {
        return 0;
    }
    const left = 1 - age / FLASH_SECONDS;
    return FLASH * left * left;
}
