// Pumpkin smash: which pumpkin a throw smashes, where pumpkins go, how the pieces, sweets and
// sparkles move and fade, and the round's clock and score. Run by tests/test_viewer.py with Node's
// own test runner, so no npm package is needed.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { forwardOf } from '../../app/viewer/cameras.js';
import { projectPoint } from '../../app/viewer/haunt.js';
import {
    CLEARANCE, MAX_AHEAD, MAX_SCREEN_SHARE, MIN_SPACING, SCREEN_SHARE_LIMIT, boxAround, buildOpenGrid,
    isOpen, minViewDistance, openDepth, placePumpkin, screenFit, solidAt,
} from '../../app/viewer/smash-place.js';
import { WEDGES, pumpkinParts, shapes, skinPoint } from '../../app/viewer/smash-meshes.js';
import {
    FLASH_SECONDS, GRAVITY, MAX_PARTICLES, PIECE_SECONDS, PUMPKIN_COUNT, PUMPKIN_RADIUS, ROUND_SECONDS,
    SPARKLES_PER_SMASH, SWEETS_PER_SMASH, THROW_SECONDS, bitAt, blankBit, blankPose, clockText,
    createPool, createRound, createThrow, fadeOf, flashOf, flicker, launchParts, launchSweets,
    partPose, pointAlong, quatAxis, screenRay, throwAt, throwHit, throwStart, turnVector,
} from '../../app/viewer/smash-play.js';

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const close = (a, b, tolerance = 1e-9) => dist(a, b) <= tolerance;

// A small seeded random source, so every run is the same.
function seeded(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

// ---- the hit test ----

test('a throw smashes the nearest pumpkin its ray passes within the radius of', () => {
    const origin = [0, 0, 0];
    const ahead = [0, 0, -1];
    const pumpkins = [
        { centre: [0.05, 0, -5], radius: 0.3 },
        { centre: [0, 0, -2], radius: 0.3 },
        null,
        { centre: [0, 0, 3], radius: 0.3 },
    ];
    const hit = throwHit(origin, ahead, pumpkins);
    assert.equal(hit.index, 1);
    // It enters the pumpkin a radius before the middle.
    assert.ok(Math.abs(hit.distance - 1.7) < 1e-12);
    // One behind the camera never counts, even straight behind.
    assert.equal(throwHit(origin, ahead, [null, null, null, pumpkins[3]]), null);
    assert.equal(throwHit(origin, [0, 0, 1], pumpkins).index, 3);
    // Past maxDistance it doesn't count either.
    assert.equal(throwHit(origin, ahead, pumpkins, 1.5), null);
});

test('a ray just inside the radius smashes and one just outside misses', () => {
    const pumpkin = [{ centre: [0, 0, -4], radius: 0.25 }];
    const passing = (offset) => throwHit([offset, 0, 0], [0, 0, -1], pumpkin);
    assert.notEqual(passing(0.2499), null);
    assert.equal(passing(0.2501), null);
    // A slanted ray, measured by its closest approach to the middle.
    const slant = [0.06, 0, -1].map((value, _, all) => value / Math.hypot(...all));
    const miss = 4 * Math.sin(Math.atan(0.06));
    assert.equal(throwHit([0, 0, 0], slant, [{ centre: [0, 0, -4], radius: miss * 0.99 }]), null);
    assert.notEqual(throwHit([0, 0, 0], slant, [{ centre: [0, 0, -4], radius: miss * 1.01 }]), null);
});

test('the ray through a pixel goes back to the same pixel, and a tap on a pumpkin smashes it', () => {
    const view = { position: [1, 2, 3], yaw: 37, pitch: -12, fov: 50, sceneLength: 4 };
    const [width, height] = [390, 844];
    const middle = screenRay(view, width / 2, height / 2, width, height);
    assert.ok(close(middle.direction, forwardOf(view.yaw, view.pitch), 1e-12));
    for (const [x, y] of [[0, 0], [width, 0], [37, 700], [width, height], [200, 100]]) {
        const ray = screenRay(view, x, y, width, height);
        const point = ray.origin.map((value, c) => value + ray.direction[c] * 3);
        const spot = projectPoint(view, point, width, height);
        assert.ok(Math.abs(spot.x - x) < 1e-6 && Math.abs(spot.y - y) < 1e-6, `pixel ${x}, ${y}`);
    }
    const pumpkin = { centre: [0, 1.5, -1], radius: 0.2 };
    const seen = projectPoint(view, pumpkin.centre, width, height);
    const tap = screenRay(view, seen.x, seen.y, width, height);
    assert.equal(throwHit(tap.origin, tap.direction, [pumpkin]).index, 0);
});

test('the candy corn flies from just below the view to where it was aimed in 0.2 s', () => {
    const view = { position: [0, 0, 0], yaw: 0, pitch: 0, fov: 60, sceneLength: 5 };
    const start = throwStart(view);
    const spot = projectPoint(view, start, 400, 300);
    // In front of the camera, low in the middle of the picture.
    assert.ok(spot.depth > 0 && Math.abs(spot.x - 200) < 1e-6 && spot.y > 150 && spot.y < 300);
    const target = [0.5, 0.2, -3];
    const flight = createThrow(start, target);
    assert.equal(THROW_SECONDS, 0.2);
    const out = { position: [0, 0, 0], direction: [0, 0, 1], done: false };
    assert.ok(close(throwAt(flight, 0, out).position, start, 1e-12) && !out.done);
    const middle = throwAt(flight, 0.1, out);
    assert.ok(Math.abs(Math.hypot(...middle.direction) - 1) < 1e-12);
    assert.ok(close(throwAt(flight, THROW_SECONDS, out).position, target, 1e-12) && out.done);
    // Its tip points the way it flies.
    const turn = pointAlong(out.direction, 1.3);
    assert.ok(close(turnVector(turn, [0, 0, 1]), out.direction, 1e-9));
});

// ---- placement ----

// A room 10 by 3 by 10 with a pillar in it, as dense splat centres on every surface, and a
// filmed walk across its middle looking at the far wall and the pillar.
function room() {
    const points = [];
    const step = 0.05;
    const face = (fixed, axis, value, ranges) => {
        const [a, b] = [0, 1, 2].filter((c) => c !== axis);
        for (let u = ranges[a][0]; u <= ranges[a][1] + 1e-9; u += step) {
            for (let v = ranges[b][0]; v <= ranges[b][1] + 1e-9; v += step) {
                const p = [0, 0, 0];
                p[axis] = value;
                p[a] = u;
                p[b] = v;
                points.push(p);
            }
        }
    };
    const walls = [[-5, 5], [-1.5, 1.5], [-5, 5]];
    for (const axis of [0, 1, 2]) {
        for (const side of [0, 1]) {
            face(null, axis, walls[axis][side], walls);
        }
    }
    // A pillar half a unit across, 1.5 in front of the walk's middle.
    const pillar = [[-0.25, 0.25], [-1.5, 1.5], [-1.75, -1.25]];
    for (const axis of [0, 2]) {
        for (const side of [0, 1]) {
            face(null, axis, pillar[axis][side], pillar);
        }
    }
    const positions = [];
    const forwards = [];
    for (let k = 0; k <= 180; k++) {
        const x = -3 + (6 * k) / 180;
        positions.push([x, 0, 0]);
        forwards.push(forwardOf(k < 90 ? 0 : 180, 0));
    }
    const path = { positions, forwards, rate: 30, duration: 6 };
    return { points, path, pillar };
}

const identity = { rows: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], offset: [0, 0, 0] };

function grid(points, path, length) {
    const flat = new Float32Array(points.flat());
    return buildOpenGrid(flat, identity, boxAround(path.positions, 0.75 * length));
}

function nearestSurface(points, p) {
    let best = Infinity;
    for (const q of points) {
        best = Math.min(best, dist(p, q));
    }
    return best;
}

test('the grid counts walls as solid and the air as open', () => {
    const { points, path } = room();
    const g = grid(points, path, 6);
    assert.equal(solidAt(g, [0, 0, -5]), true);
    assert.equal(solidAt(g, [0.25, 0, -1.5]), true);
    assert.equal(solidAt(g, [2, 0, 2]), false);
    assert.equal(isOpen(g, [2, 0, 2], 0.3), true);
    assert.equal(isOpen(g, [0, 0, -1.0], 0.4), false);
    // Looking from the walk's middle at the pillar, the first solid cube is at the pillar.
    const depth = openDepth(g, [0, 0, 0], [0, 0, -1], 4);
    assert.ok(depth > 1.15 && depth < 1.35, `depth ${depth}`);
    // Stray floaters in the air don't make it solid.
    const floaters = [...points, [2, 0.3, 2], [2.01, 0.31, 2.01]];
    assert.equal(solidAt(grid(floaters, path, 6), [2, 0.3, 2]), false);
});

test('property: pumpkins go near the path, spread out, out of every wall and never filling the view', () => {
    const { points, path } = room();
    const length = 6;
    const g = grid(points, path, length);
    const radius = PUMPKIN_RADIUS * length;
    for (let seed = 1; seed <= 12; seed++) {
        const random = seeded(seed);
        const view = { position: [...path.positions[Math.floor(random() * 181)]], fov: 30 + random() * 60, sceneLength: length };
        const placed = [];
        for (let n = 0; n < PUMPKIN_COUNT; n++) {
            const p = placePumpkin({ path, grid: g, view, others: placed, radius, random });
            assert.notEqual(p, null, `seed ${seed}, pumpkin ${n}`);
            placed.push(p);
        }
        for (const p of placed) {
            const where = `seed ${seed}, at ${p.map((v) => v.toFixed(2))}`;
            // Out of every wall and the pillar, with room to spare.
            assert.ok(nearestSurface(points, p) >= radius * 1.2, where);
            // At most MAX_SCREEN_SHARE of the view's height across.
            const share = (2 * radius) / (2 * dist(p, view.position) * Math.tan((view.fov * Math.PI) / 360));
            assert.ok(share <= MAX_SCREEN_SHARE + 1e-9, `${where}: ${share}`);
            // Near the filmed path.
            const fromPath = Math.min(...path.positions.map((q) => dist(p, q)));
            assert.ok(fromPath <= MAX_AHEAD * length + 1e-9, where);
            for (const q of placed) {
                assert.ok(p === q || dist(p, q) >= MIN_SPACING * length, `${where}: too close to another`);
            }
        }
    }
});

test('with no splat centres, pumpkins still go near the path and the rules still hold', () => {
    const { path } = room();
    const view = { position: [0, 0, 0], fov: 60, sceneLength: 6 };
    const radius = PUMPKIN_RADIUS * 6;
    const random = seeded(5);
    const placed = [];
    for (let n = 0; n < PUMPKIN_COUNT; n++) {
        placed.push(placePumpkin({ path, grid: null, view, others: placed, radius, random }));
    }
    for (const p of placed) {
        assert.notEqual(p, null);
        assert.ok(dist(p, view.position) >= minViewDistance(radius, view.fov) - 1e-9);
        assert.ok(Math.min(...path.positions.map((q) => dist(p, q))) <= 0.5 * MAX_AHEAD * 6 + 1e-9);
    }
});

test('a pumpkin with nowhere open to go is not placed', () => {
    const { path } = room();
    // Solid everywhere: one splat in every cube.
    const box = boxAround(path.positions, 4.5);
    const fill = [];
    for (let x = box.lo[0]; x < box.hi[0]; x += 0.2) {
        for (let y = box.lo[1]; y < box.hi[1]; y += 0.2) {
            for (let z = box.lo[2]; z < box.hi[2]; z += 0.2) {
                fill.push(x, y, z, x + 0.01, y, z, x, y + 0.01, z);
            }
        }
    }
    // Cubes bigger than the fill's spacing, so every cube holds splats.
    const full = buildOpenGrid(new Float32Array(fill), identity, box, { cells: 48 });
    const view = { position: [0, 0, 0], fov: 60, sceneLength: 6 };
    assert.equal(placePumpkin({ path, grid: full, view, radius: 0.2, random: seeded(2) }), null);
});

test('a pumpkin is shrunk only when the camera comes close enough for it to fill the view', () => {
    assert.equal(screenFit(10, 0.2, 60), 1);
    const near = 0.1;
    const fit = screenFit(near, 0.2, 60);
    assert.ok(fit < 1);
    const share = (2 * 0.2 * fit) / (2 * near * Math.tan(Math.PI / 6));
    assert.ok(Math.abs(share - SCREEN_SHARE_LIMIT) < 1e-12);
    assert.ok(CLEARANCE > 1);
});

// ---- the pieces' motion ----

const pumpkin = { centre: [1, 2, 3], turn: quatAxis([0, 1, 0], 0.7), radius: 0.25 };

test('the 8 wedges and the stem start where they sat in the pumpkin, then fly apart and fall', () => {
    const parts = pumpkinParts();
    assert.equal(parts.filter((part) => part.kind === 'wedge').length, WEDGES);
    assert.equal(parts.filter((part) => part.kind === 'stem').length, 1);
    const pool = createPool(9, blankBit);
    launchParts(pumpkin, parts, seeded(3), 10, (now) => pool.spawn(now));
    const whole = blankPose();
    const flying = blankPose();
    pool.items.forEach((bit, i) => {
        partPose(pumpkin, parts[i], whole);
        bitAt(bit, 0, flying);
        assert.ok(close(flying.position, whole.position, 1e-12), `part ${i}`);
        assert.ok(close(flying.turn.slice(0, 3), whole.turn.slice(0, 3), 1e-12) && Math.abs(flying.turn[3] - whole.turn[3]) < 1e-12);
        assert.equal(flying.size, pumpkin.radius);
        // Later it's further out, and in the end it's falling.
        const later = bitAt(bit, 1, blankPose());
        const outward = (p) => Math.hypot(p[0] - pumpkin.centre[0], p[2] - pumpkin.centre[2]);
        assert.ok(outward(later.position) >= outward(whole.position) - 1e-12, `part ${i} moves out`);
        const v = (t) => bit.velocity[1] - bit.gravity * t;
        assert.ok(v(PIECE_SECONDS) < 0, `part ${i} falls`);
    });
});

test('a piece moves by gravity, spins at its rate and fades out by 1.5 s', () => {
    const bit = { ...blankBit(), position: [0, 1, 0], velocity: [0.5, 2, -1], spinAxis: [0, 0, 1], spinRate: 3, gravity: 4, size: 0.2, life: 1.5, fade: 0.5 };
    const out = blankPose();
    for (const t of [0, 0.3, 0.9, 1.2]) {
        bitAt(bit, t, out);
        assert.ok(close(out.position, [0.5 * t, 1 + 2 * t - 2 * t * t, -t], 1e-12), `t ${t}`);
        assert.ok(Math.abs(Math.hypot(...out.turn) - 1) < 1e-12);
        // About its axis by spinRate t.
        assert.ok(Math.abs(2 * Math.atan2(out.turn[2], out.turn[3]) - 3 * t) < 1e-9);
    }
    // Steady by gravity alone: a frame-by-frame step lands where the closed form says.
    let p = [...bit.position];
    let v = [...bit.velocity];
    const dt = 1 / 60;
    for (let k = 0; k < 60; k++) {
        p = p.map((value, c) => value + v[c] * dt - (c === 1 ? 0.5 * bit.gravity * dt * dt : 0));
        v[1] -= bit.gravity * dt;
    }
    assert.ok(close(p, bitAt(bit, 1, out).position, 1e-9));
    assert.equal(fadeOf(0, 1.5, 0.5), 1);
    assert.equal(fadeOf(0.99, 1.5, 0.5), 1);
    assert.ok(fadeOf(1.25, 1.5, 0.5) > 0 && fadeOf(1.25, 1.5, 0.5) < 1);
    assert.equal(fadeOf(1.5, 1.5, 0.5), 0);
    assert.equal(bitAt(bit, 1.5, out).size, 0);
});

test('every piece, sweet and sparkle of a burst is gone within 1.5 s', () => {
    const pool = createPool(MAX_PARTICLES, blankBit);
    const random = seeded(8);
    launchParts(pumpkin, pumpkinParts(), random, 0, (now) => pool.spawn(now));
    launchSweets(pumpkin.centre, pumpkin.radius, random, 0, (now) => pool.spawn(now));
    const alive = pool.items.filter((bit) => bit.alive);
    assert.equal(alive.length, 9 + SWEETS_PER_SMASH + SPARKLES_PER_SMASH);
    assert.equal(alive.filter((bit) => bit.kind === 'sweet').length, SWEETS_PER_SMASH);
    assert.equal(alive.filter((bit) => bit.kind === 'sparkle').length, SPARKLES_PER_SMASH);
    const out = blankPose();
    for (const bit of alive) {
        assert.ok(bit.life <= PIECE_SECONDS, bit.kind);
        assert.ok(bitAt(bit, 0, out).size > 0, bit.kind);
        assert.equal(bitAt(bit, PIECE_SECONDS, out).size, 0, bit.kind);
        // The sweets start going up; everything falls with some of the game's gravity.
        if (bit.kind === 'sweet') {
            assert.ok(bit.velocity[1] > 0);
        }
        assert.ok(bit.gravity > 0 && bit.gravity <= GRAVITY * pumpkin.radius);
    }
});

test('the pool reuses its objects and never holds more than its capacity', () => {
    const pool = createPool(MAX_PARTICLES, blankBit);
    const objects = new Set(pool.items);
    const random = seeded(4);
    for (let burst = 0; burst < 20; burst++) {
        launchSweets([0, 0, 0], 1, random, burst * 0.1, (now) => pool.spawn(now));
        assert.ok(pool.count() <= MAX_PARTICLES);
    }
    assert.equal(pool.count(), MAX_PARTICLES);
    assert.ok(pool.items.every((item) => objects.has(item)));
    // When it's full, the next one is the oldest.
    const oldest = pool.items.reduce((a, b) => (b.born < a.born ? b : a));
    assert.equal(pool.spawn(99), oldest);
    // A retired one is handed out before any live one.
    const small = createPool(3, blankBit);
    const [a, b] = [small.spawn(1), small.spawn(2)];
    small.retire(a);
    assert.equal(small.spawn(3), a);
    assert.notEqual(small.spawn(4), b);
});

test('the candle flickers gently, and a smashed pumpkin flashes and goes out', () => {
    for (let t = 0; t < 20; t += 0.013) {
        const f = flicker(t, 0.37);
        assert.ok(f >= 0.74 && f <= 1.06, `t ${t}: ${f}`);
    }
    assert.ok(flicker(0.5, 0.1) !== flicker(0.5, 0.6));
    assert.ok(flashOf(0) > flicker(0, 0) * 2);
    assert.ok(flashOf(FLASH_SECONDS / 2) < flashOf(0));
    assert.equal(flashOf(FLASH_SECONDS), 0);
    assert.equal(flashOf(-0.1), 0);
});

// ---- the round ----

test('a round runs 60 seconds, counts smashes only while it runs, and starts again from 0', () => {
    const round = createRound();
    assert.equal(round.state, 'idle');
    assert.equal(round.smash(), false);
    round.start();
    assert.equal(ROUND_SECONDS, 60);
    assert.equal(round.timeLeft, 60);
    assert.equal(clockText(round.timeLeft), '1:00');
    assert.equal(round.smash(), true);
    round.smash();
    assert.equal(round.score, 2);
    assert.equal(round.step(30), false);
    assert.equal(round.timeLeft, 30);
    assert.equal(clockText(29.2), '0:30');
    assert.equal(round.step(29.99), false);
    assert.equal(clockText(round.timeLeft), '0:01');
    // The step where the time runs out says so, once.
    assert.equal(round.step(0.5), true);
    assert.equal(round.state, 'over');
    assert.equal(round.timeLeft, 0);
    assert.equal(clockText(round.timeLeft), '0:00');
    assert.equal(round.step(1), false);
    assert.equal(round.smash(), false);
    assert.equal(round.score, 2);
    round.start();
    assert.equal(round.state, 'playing');
    assert.equal(round.score, 0);
    assert.equal(round.timeLeft, 60);
    round.end();
    assert.equal(round.state, 'idle');
    assert.equal(round.step(5), false);
});

test('property: at any frame rate the round ends after 60 seconds, on exactly one step', () => {
    const random = seeded(17);
    for (let run = 0; run < 30; run++) {
        const round = createRound();
        round.start();
        let elapsed = 0;
        let ends = 0;
        while (elapsed < 70) {
            const dt = 0.004 + random() * 0.1;
            if (round.step(dt)) {
                ends += 1;
                assert.ok(elapsed < 60 + 1e-9 && elapsed + dt >= 60 - 1e-9, `run ${run}`);
            }
            elapsed += dt;
        }
        assert.equal(ends, 1);
    }
});

// ---- the shapes ----

test('the shapes are whole: every index names a vertex, and the face sits outside the skin', () => {
    const all = shapes();
    for (const [name, shape] of Object.entries(all)) {
        const vertices = shape.positions.length / 3;
        assert.equal(shape.normals.length, shape.positions.length, name);
        assert.equal(shape.colors.length, vertices * 4, name);
        assert.equal(shape.indices.length % 3, 0, name);
        assert.ok(shape.indices.every((i) => i < vertices), name);
        assert.ok(vertices < 65536, name);
    }
    // Each face vertex is further out than the skin in the same direction.
    const face = all.face.positions;
    for (let i = 0; i < face.length; i += 3) {
        const p = [face[i], face[i + 1], face[i + 2]];
        const phi = Math.atan2(p[0], p[2]);
        const across = Math.hypot(p[0], p[2]);
        // The skin's point at the same angle about +y and the same height, found by the polar angle.
        let lo = 0;
        let hi = Math.PI;
        for (let k = 0; k < 60; k++) {
            const mid = (lo + hi) / 2;
            if (skinPoint(mid, phi)[1] > p[1]) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const skin = skinPoint(lo, phi);
        assert.ok(across > Math.hypot(skin[0], skin[2]), `face vertex ${i / 3}`);
    }
    // The 8 wedges go round the pumpkin, an eighth of a turn apart.
    const offsets = pumpkinParts().filter((part) => part.kind === 'wedge').map((part) => part.offset);
    for (let k = 0; k < WEDGES; k++) {
        const a = Math.atan2(offsets[k][0], offsets[k][2]);
        const expected = (2 * Math.PI * k) / WEDGES;
        assert.ok(Math.abs(Math.atan2(Math.sin(a - expected), Math.cos(a - expected))) < 1e-9, `wedge ${k}`);
    }
});
