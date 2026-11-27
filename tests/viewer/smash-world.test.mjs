// Pumpkin smash's world: which cubes are walls (and that floaters, fog and the phone's own path
// aren't), the player's body bumping into walls, standing, stepping, jumping and flying, the
// pumpkins chasing around a pillar, the camera behind the player, mouse look, the penalties, the
// light that follows how fired up you are, and the human model. Run by tests/test_viewer.py with
// Node's own test runner, so no npm package is needed.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { forwardOf } from '../../app/viewer/cameras.js';
import { WALK_KEYS, walk } from '../../app/viewer/controls.js';
import {
    FOREARM, HUMAN_PARTS, SHIN, THIGH, THROW_POSE, UPPER_ARM, humanPose, humanShapes, rightHand,
} from '../../app/viewer/smash-human.js';
import {
    EYE_SHARE, STEP_SHARE, bodyFits, bodyFor, canSee, chaseStep, createBody, eyeOf, floorUnder,
    followCamera, mouseLook, rescueBody, stepBody,
} from '../../app/viewer/smash-physics.js';
import { MIN_CLUMP, buildOpenGrid, isOpen, openDepth, solidAt } from '../../app/viewer/smash-place.js';
import {
    CAUGHT_PENALTY, LANTERN_PENALTY, MAX_PARTICLES, PIECE_SECONDS, bitAt, blankBit, blankPose,
    chaseSpeed, createPool, createRound, furyAfter, furyKick, launchShards, lightingFor, resultLines,
    throwHit, turnVector,
} from '../../app/viewer/smash-play.js';

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const identity = { rows: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], offset: [0, 0, 0] };
const CELLS = 48;

function seeded(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

// Dense splat centres on every face of a box from lo to hi.
function boxSurface(points, lo, hi, step = 0.05) {
    for (let axis = 0; axis < 3; axis++) {
        const [a, b] = [0, 1, 2].filter((c) => c !== axis);
        for (const value of [lo[axis], hi[axis]]) {
            for (let u = lo[a]; u <= hi[a] + 1e-9; u += step) {
                for (let v = lo[b]; v <= hi[b] + 1e-9; v += step) {
                    const p = [0, 0, 0];
                    p[axis] = value;
                    p[a] = u;
                    p[b] = v;
                    points.push(p);
                }
            }
        }
    }
    return points;
}

// A room 6 by 3 by 6 with its floor at y = 0, a pillar in the middle, a low step (0.3 high) and a
// tall crate (1.0 high), filmed along x at eye height 1.5. The scene length is the walk's 3.
function room(extra = []) {
    const points = boxSurface([], [-3, 0, -3], [3, 3, 3]);
    boxSurface(points, [-0.3, 0, -0.3], [0.3, 3, 0.3]);
    boxSurface(points, [0.8, 0, 1.2], [1.6, 0.3, 2.0]);
    boxSurface(points, [-2.2, 0, 1.2], [-1.4, 1.0, 2.0]);
    points.push(...extra);
    const positions = [];
    const forwards = [];
    for (let k = 0; k <= 90; k++) {
        positions.push([-1.5 + (3 * k) / 90, 1.5, -1.5]);
        forwards.push(forwardOf(0, 0));
    }
    const path = { positions, forwards, rate: 30, duration: 3 };
    const options = { cells: CELLS, keepOpen: positions, openRadius: 0.15 };
    const grid = buildOpenGrid(new Float32Array(points.flat()), identity, { lo: [-4, -1, -4], hi: [4, 4, 4] }, options);
    return { grid, path, length: 3 };
}

const world = room();

// ---- the walls ----

test('walls, the floor and things in the room are solid, and the air is open', () => {
    const { grid } = world;
    assert.equal(solidAt(grid, [3, 1.5, 0]), true);
    assert.equal(solidAt(grid, [0, 0, 2.5]), true);
    assert.equal(solidAt(grid, [0.3, 1.5, 0]), true);
    assert.equal(solidAt(grid, [1.5, 1.5, 0]), false);
    assert.equal(isOpen(grid, [-1.5, 1.5, 0], 0.4), true);
});

test('fog, a floater and anything on the phone path are left open', () => {
    const random = seeded(3);
    const fog = [];
    for (let n = 0; n < 300; n++) {
        fog.push([1.2 + random(), 1.8 + random(), -2 + random()]);
    }
    // A dense little floater in mid-air, smaller than MIN_CLUMP cubes.
    const floater = [];
    for (let n = 0; n < 80; n++) {
        floater.push([-1 + random() * 0.08, 2.2 + random() * 0.08, 1 + random() * 0.08]);
    }
    // A dense blob right where the phone was.
    const onPath = [];
    for (let n = 0; n < 200; n++) {
        onPath.push([0.6 + random() * 0.1, 1.5 + random() * 0.1, -1.5 + random() * 0.1]);
    }
    const { grid } = room([...fog, ...floater, ...onPath]);
    assert.ok(MIN_CLUMP > 8);
    for (let n = 0; n < 50; n++) {
        assert.equal(solidAt(grid, [1.2 + random(), 1.8 + random(), -2 + random()]), false, 'fog');
    }
    assert.equal(solidAt(grid, [-0.96, 2.24, 1.04]), false, 'floater');
    assert.equal(solidAt(grid, [0.65, 1.55, -1.45]), false, 'the phone path');
    // The walls are still there.
    assert.equal(solidAt(grid, [-3, 1, 0]), true);
});

// ---- the body ----

test('the eye height comes from how far the floor is under the filmed path', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    assert.ok(Math.abs(shape.eye - 1.5) <= grid.cell + 1e-9, `eye ${shape.eye}`);
    assert.ok(Math.abs(shape.eye / shape.height - EYE_SHARE) < 1e-12);
    assert.ok(shape.gravity > 0 && shape.jump > 0 && shape.fallbackFloor !== null);
    // With no grid there's nothing to stand on: flying only.
    const free = bodyFor(null, path, length);
    assert.equal(free.gravity, null);
    assert.equal(createBody(free, [0, 1, 0]).flying, true);
});

function settle(shape, body, grid, seconds = 2) {
    for (let t = 0; t < seconds; t += 1 / 60) {
        stepBody(grid, shape, body, [0, 0, 0], 1 / 60);
    }
}

test('dropped from up high, the body falls onto the floor and stands there', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    const body = createBody(shape, [-1.5, 2.9, -1.5]);
    settle(shape, body, grid);
    assert.equal(body.onGround, true);
    assert.ok(body.feet[1] >= 0 && body.feet[1] <= grid.cell + 1e-9, `feet ${body.feet[1]}`);
    assert.ok(Math.abs(eyeOf(shape, body)[1] - shape.eye - body.feet[1]) < 1e-12);
});

test('walking into a wall stops at it and slides along it', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    const body = createBody(shape, [2, 1.5, -1.5]);
    settle(shape, body, grid, 0.5);
    for (let n = 0; n < 60; n++) {
        stepBody(grid, shape, body, [0.05, 0, 0.02], 1 / 60);
        assert.equal(bodyFits(grid, shape, body.feet, false), true);
    }
    assert.ok(body.feet[0] < 3 - shape.radius * 0.9, `x ${body.feet[0]}`);
    // The z part of the move went on along the wall.
    assert.ok(body.feet[2] > -1.5 + 0.9, `z ${body.feet[2]}`);
});

test('a fast move never passes through a thin wall', () => {
    const points = boxSurface([], [-3, 0, -3], [3, 3, 3]);
    for (let y = 0; y <= 3; y += 0.04) {
        for (let z = -3; z <= 3; z += 0.04) {
            points.push([1, y, z]);
        }
    }
    const positions = [[-1.5, 1.5, 0], [-0.5, 1.5, 0]];
    const grid = buildOpenGrid(new Float32Array(points.flat()), identity, { lo: [-4, -1, -4], hi: [4, 4, 4] }, { cells: CELLS });
    const shape = bodyFor(grid, { positions, forwards: [[0, 0, -1], [0, 0, -1]], rate: 30 }, 3);
    const body = createBody(shape, [0, 1.5, 0]);
    settle(shape, body, grid, 0.5);
    stepBody(grid, shape, body, [5, 0, 0], 1 / 60);
    assert.ok(body.feet[0] < 1, `x ${body.feet[0]}`);
});

test('the body steps up onto something low and is stopped by something tall', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    assert.ok(0.3 < STEP_SHARE * shape.height && 1.0 > STEP_SHARE * shape.height);
    const low = createBody(shape, [1.2, 1.5, 0.2]);
    settle(shape, low, grid, 0.5);
    for (let n = 0; n < 60; n++) {
        stepBody(grid, shape, low, [0, 0, 0.03], 1 / 60);
    }
    assert.ok(low.feet[2] > 1.4, `walked onto the step: z ${low.feet[2]}`);
    assert.ok(low.feet[1] >= 0.29 && low.feet[1] <= 0.3 + grid.cell, `stands on it: y ${low.feet[1]}`);
    const tall = createBody(shape, [-1.8, 1.5, 0.2]);
    settle(shape, tall, grid, 0.5);
    for (let n = 0; n < 60; n++) {
        stepBody(grid, shape, tall, [0, 0, 0.03], 1 / 60);
    }
    assert.ok(tall.feet[2] < 1.2 - shape.radius * 0.5, `stopped by the crate: z ${tall.feet[2]}`);
    assert.ok(tall.feet[1] < grid.cell + 1e-9);
});

test('a jump goes up and comes back down, and flying ignores gravity but not the ceiling', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    const body = createBody(shape, [-1.5, 1.5, -1.5]);
    settle(shape, body, grid, 0.5);
    const floor = body.feet[1];
    stepBody(grid, shape, body, [0, 0, 0], 1 / 60, true);
    let top = floor;
    for (let t = 0; t < 1.5; t += 1 / 60) {
        stepBody(grid, shape, body, [0, 0, 0], 1 / 60);
        top = Math.max(top, body.feet[1]);
    }
    assert.ok(top - floor > 0.3, `jumped ${top - floor}`);
    assert.ok(Math.abs(body.feet[1] - floor) < 1e-9 && body.onGround);
    body.flying = true;
    stepBody(grid, shape, body, [0, 0.5, 0], 1 / 60);
    const flown = body.feet[1];
    settle(shape, body, grid, 0.5);
    assert.equal(body.feet[1], flown, 'no falling while flying');
    stepBody(grid, shape, body, [0, 10, 0], 1 / 60);
    assert.ok(eyeOf(shape, body)[1] + shape.radius <= 3 + 1e-9, 'under the ceiling');
});

test('property: walking at random never leaves the body inside a wall or the floor', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    const random = seeded(29);
    for (let run = 0; run < 6; run++) {
        const body = createBody(shape, [-1.5 + random() * 3, 1.5, -1.5]);
        settle(shape, body, grid, 0.3);
        for (let n = 0; n < 250; n++) {
            const dt = 0.005 + random() * 0.05;
            const a = random() * 2 * Math.PI;
            const speed = random() * 3;
            stepBody(grid, shape, body, [Math.cos(a) * speed * dt, 0, Math.sin(a) * speed * dt], dt, random() < 0.05);
            assert.equal(bodyFits(grid, shape, body.feet, false), true, `run ${run}, step ${n}`);
            assert.ok(body.feet[1] >= -1e-9, `run ${run}, step ${n}: under the floor`);
        }
    }
});

test('a body that starts inside something is put back on the filmed path', () => {
    const { grid, path, length } = world;
    const shape = bodyFor(grid, path, length);
    // Half in the pillar's side. (Inside a hollow thing counts as open: splats only show surfaces.)
    const body = createBody(shape, [0.3, 1.5, 0]);
    assert.equal(rescueBody(grid, shape, body, path), true);
    assert.ok(bodyFits(grid, shape, body.feet, true) || bodyFits(grid, shape, body.feet, false));
    assert.equal(rescueBody(grid, shape, body, path), false);
    assert.ok(floorUnder(grid, [-1.5, 1.5, -1.5], 0, 3) <= grid.cell + 1e-9);
});

// ---- the chase ----

test('a pumpkin chases round the pillar to reach you, and never goes into it', () => {
    const { grid } = world;
    const p = [0, 1.2, -1.5];
    const target = [0, 1.2, 1.2];
    const memory = { side: 1 };
    assert.equal(canSee(grid, p, target, 0.15), false);
    assert.equal(canSee(grid, [-1.5, 1.2, -1.5], [-1.5, 1.2, 1.0], 0.15), true);
    for (let n = 0; n < 900 && dist(p, target) > 0.05; n++) {
        chaseStep(grid, p, target, 0.6, 0.15, 1 / 30, memory);
        assert.equal(isOpen(grid, p, 0.15 * 0.999), true, `step ${n}`);
    }
    assert.ok(dist(p, target) < 0.1, `still ${dist(p, target)} away`);
});

test('in the open, a chasing pumpkin closes in at its speed', () => {
    const { grid } = world;
    const p = [-2, 1.5, -2];
    const target = [-2, 1.5, -0.5];
    chaseStep(grid, p, target, 0.6, 0.15, 0.5, { side: 1 });
    assert.ok(Math.abs(dist(p, target) - 1.2) < 1e-9);
    assert.ok(chaseSpeed(0) < chaseSpeed(30) && chaseSpeed(30) < chaseSpeed(60));
    assert.ok(chaseSpeed(60) < 0.4 && chaseSpeed(1000) === chaseSpeed(60));
});

test('a throw at a pumpkin behind the pillar hits the pillar instead', () => {
    const { grid } = world;
    const origin = [0, 1.5, -2];
    const direction = [0, 0, 1];
    const wall = openDepth(grid, origin, direction, 5);
    const pumpkin = [{ centre: [0, 1.5, 1], radius: 0.2 }];
    assert.ok(wall < 2);
    assert.equal(throwHit(origin, direction, pumpkin, wall), null);
    assert.notEqual(throwHit(origin, direction, pumpkin, 5), null);
});

// ---- looking and the camera ----

test('moving the mouse right turns right and moving it down looks down', () => {
    const view = { yaw: 0, pitch: 0 };
    mouseLook(view, 10, 0, 0.12);
    const right = [1, 0, 0];
    const f = forwardOf(view.yaw, view.pitch);
    assert.ok(f[0] * right[0] > 0);
    mouseLook(view, 0, 10, 0.12);
    assert.ok(forwardOf(view.yaw, view.pitch)[1] < 0);
    mouseLook(view, 0, -100000, 0.12);
    assert.equal(view.pitch, 89);
});

test('the camera behind you stays out of the wall at your back', () => {
    const { grid } = world;
    const free = followCamera(grid, [-1.5, 1.5, 1.0], { yaw: 0, pitch: 0 }, 0.8, 0.2);
    assert.ok(dist(free, [-1.3, 1.5, 1.8]) < 1e-9, `free ${free}`);
    const backed = followCamera(grid, [2.4, 1.5, -1.5], { yaw: 90, pitch: 0 }, 1.2, 0.2);
    assert.ok(backed[0] < 3 && !solidAt(grid, backed), `backed ${backed}`);
    // A thin, sparse wall the player could walk through still stops the camera.
    // Two splats in every cube of a patch of wall: below the walls' density, at the camera's.
    const sparse = [];
    const { lo, cell } = grid;
    const iz = Math.floor((2.4 - lo[2]) / cell);
    for (let iy = Math.floor((1 - lo[1]) / cell); iy <= Math.floor((2 - lo[1]) / cell); iy++) {
        for (let ix = Math.floor((-2.5 - lo[0]) / cell); ix <= Math.floor((-0.5 - lo[0]) / cell); ix++) {
            const centre = [lo[0] + (ix + 0.5) * cell, lo[1] + (iy + 0.5) * cell, lo[2] + (iz + 0.5) * cell];
            sparse.push(centre, [centre[0] + cell * 0.1, centre[1], centre[2]]);
        }
    }
    const thin = room(sparse).grid;
    assert.ok(Math.abs(thin.cell - cell) < 1e-12);
    assert.equal(solidAt(thin, [-1.3, 1.5, 2.4]), false);
    const stopped = followCamera(thin, [-1.5, 1.5, 1.0], { yaw: 0, pitch: 0 }, 2, 0.2);
    assert.ok(stopped[2] < 2.4, `camera at ${stopped}`);
    // Looking up or down doesn't lift the camera out of the room.
    const pitched = followCamera(grid, [-1.5, 1.5, 1.0], { yaw: 0, pitch: -60 }, 0.8, 0.2);
    assert.ok(dist(pitched, free) < 1e-9);
});

test('Q and E go down and up in the viewer, and WASD stays level', () => {
    assert.equal(WALK_KEYS.KeyQ, 'down');
    assert.equal(WALK_KEYS.KeyE, 'up');
    const view = { position: [0, 1, 0], yaw: 30, pitch: -20, sceneLength: 2 };
    assert.equal(walk(view, new Set(['up']), 0.5), true);
    assert.ok(Math.abs(view.position[1] - 1.4) < 1e-12 && view.position[0] === 0 && view.position[2] === 0);
    walk(view, new Set(['down']), 0.5);
    assert.ok(Math.abs(view.position[1] - 1) < 1e-12);
    walk(view, new Set(['forward', 'left']), 0.5);
    assert.ok(Math.abs(view.position[1] - 1) < 1e-12);
    assert.equal(walk(view, new Set(), 0.5), false);
});

// ---- scoring and the light ----

test('a lantern costs 2, being caught costs 1, and the score can go below 0', () => {
    const round = createRound();
    assert.equal(round.breakLantern(), false);
    round.start();
    assert.equal(round.breakLantern(), true);
    assert.equal(round.score, -LANTERN_PENALTY);
    round.smash();
    round.smash();
    round.smash();
    assert.equal(round.caughtBy(), true);
    assert.equal(round.score, 3 - LANTERN_PENALTY - CAUGHT_PENALTY);
    assert.deepEqual([round.smashed, round.broken, round.caught], [3, 1, 1]);
    assert.deepEqual(resultLines(round), [
        'You smashed 3 pumpkins with candy corn.',
        'You broke 1 lantern: −2.',
        'Pumpkins caught you 1 time: −1.',
        'Score: 0',
    ]);
    round.step(60);
    assert.equal(round.caughtBy(), false);
    round.start();
    assert.deepEqual([round.score, round.smashed, round.broken, round.caught], [0, 0, 0, 0]);
});

test('the more fired up you are, the brighter and warmer the room, and it calms down again', () => {
    let previous = null;
    for (let f = 0; f <= 1.0001; f += 0.1) {
        const light = lightingFor(f);
        if (previous !== null) {
            assert.ok(light.keep[0] >= previous.keep[0] && light.keep[1] >= previous.keep[1]);
            assert.ok(light.pumpkinGain >= previous.pumpkinGain && light.playerStrength >= previous.playerStrength);
            assert.ok(light.playerReach >= previous.playerReach);
        }
        previous = light;
    }
    const calm = lightingFor(0);
    const fired = lightingFor(1);
    // Calm is a cool, dim room, though far from black; fired up is warm.
    assert.ok(calm.keep[2] > calm.keep[0] && calm.keep[0] >= 0.35);
    assert.ok(fired.keep[0] > fired.keep[2]);
    assert.equal(furyKick(0.95, 0.2), 1);
    assert.equal(furyKick(0.1, -0.5), 0);
    assert.ok(furyAfter(1, 5) < 0.4 && furyAfter(1, 5) > 0.3);
    assert.equal(furyAfter(0.5, 0), 0.5);
});

test('a broken lantern bursts into shards that fall and are gone within 1.5 s', () => {
    const pool = createPool(MAX_PARTICLES, blankBit);
    launchShards([0, 1, 0], 0.2, seeded(6), 0, (now) => pool.spawn(now));
    const alive = pool.items.filter((bit) => bit.alive);
    assert.ok(alive.length > 10 && alive.some((bit) => bit.kind === 'shard'));
    const out = blankPose();
    for (const bit of alive) {
        assert.ok(bit.gravity > 0 && bit.life <= PIECE_SECONDS);
        assert.equal(bitAt(bit, PIECE_SECONDS, out).size, 0);
    }
});

// ---- the human model ----

const parts = (pose, name) => pose.filter((item) => item.part === name);

function worldVertices(item, height) {
    const shape = humanShapes()[item.part];
    const out = [];
    for (let i = 0; i < shape.positions.length; i += 3) {
        const v = turnVector(item.turn, [shape.positions[i] * height, shape.positions[i + 1] * height, shape.positions[i + 2] * height]);
        out.push([item.position[0] + v[0], item.position[1] + v[1], item.position[2] + v[2]]);
    }
    return out;
}

test('the person stands on the ground, is their height tall and faces the view', () => {
    const height = 1.7;
    for (const yaw of [0, 90, 217]) {
        const feet = [1, 0.4, -2];
        const pose = humanPose(feet, yaw, height, {});
        assert.equal(pose.length, 15);
        assert.ok(pose.every((item) => HUMAN_PARTS.includes(item.part)));
        const ys = pose.flatMap((item) => worldVertices(item, height).map((p) => p[1]));
        assert.ok(Math.abs(Math.min(...ys) - feet[1]) < 0.03 * height, `bottom ${Math.min(...ys)}`);
        assert.ok(Math.abs(Math.max(...ys) - (feet[1] + height)) < 0.03 * height, `top ${Math.max(...ys)}`);
        const front = turnVector(parts(pose, 'head')[0].turn, [0, 0, 1]);
        const forward = forwardOf(yaw, 0);
        assert.ok(dist(front, forward) < 1e-9, `yaw ${yaw}`);
    }
    for (const shape of Object.values(humanShapes())) {
        assert.ok(shape.indices.every((i) => i < shape.positions.length / 3));
    }
});

test('the arms and legs stay joined in every pose', () => {
    const height = 1.6;
    const random = seeded(12);
    for (let n = 0; n < 40; n++) {
        const pose = humanPose([0, 0, 0], random() * 360, height, {
            phase: random() * 10, stride: random(), air: random() < 0.3, flying: random() < 0.2,
            throwAge: random() < 0.5 ? random() * THROW_POSE : Infinity,
        });
        for (const [parent, child, size] of [['thigh', 'shin', THIGH], ['shin', 'shoe', SHIN], ['upperArm', 'foreArm', UPPER_ARM], ['foreArm', 'hand', FOREARM]]) {
            const a = parts(pose, parent);
            const b = parts(pose, child);
            for (let side = 0; side < 2; side++) {
                const end = turnVector(a[side].turn, [0, -size * height, 0]);
                const joint = a[side].position.map((value, c) => value + end[c]);
                assert.ok(dist(joint, b[side].position) < 1e-9, `${parent} to ${child}`);
            }
        }
    }
});

test('walking swings the legs, and a throw lifts the right hand and brings it over', () => {
    const height = 1.7;
    const forward = forwardOf(0, 0);
    const along = (p) => p[0] * forward[0] + p[2] * forward[2];
    const stride = humanPose([0, 0, 0], 0, height, { phase: Math.PI / 2, stride: 1 });
    const shoes = parts(stride, 'shoe');
    assert.ok(Math.abs(along(shoes[0].position) - along(shoes[1].position)) > 0.1 * height);
    const standing = humanPose([0, 0, 0], 0, height, {});
    const shoulder = parts(standing, 'upperArm')[1].position;
    const windUp = rightHand(humanPose([0, 0, 0], 0, height, { throwAge: 0.35 * THROW_POSE }));
    assert.ok(windUp[1] > shoulder[1], 'the hand goes up');
    const release = rightHand(humanPose([0, 0, 0], 0, height, { throwAge: 0.95 * THROW_POSE }));
    assert.ok(along(release) - along(shoulder) > 0.1 * height, 'and comes over in front');
});
