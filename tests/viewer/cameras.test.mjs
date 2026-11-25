// The viewer's reading of out/cameras.json and its starting camera.
// Run by tests/test_viewer.py with Node's own test runner, so no npm package is needed.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
    CamerasError, fallbackView, forwardOf, matrixToQuaternion, parseCameras, quaternionToMatrix, startView,
} from '../../app/viewer/cameras.js';

const CLOSE = 1e-9;

function near(actual, expected, message, tolerance = CLOSE) {
    assert.equal(actual.length, expected.length, message);
    actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < tolerance, `${message}: ${actual} vs ${expected}`));
}

// The example in docs/pipeline.md, "Files in out/" of the cameras step, plus a second frame.
function example(overrides = {}) {
    return {
        camera: { width: 1552, height: 873, fx: 1290.2, fy: 1290.2, cx: 776.1, cy: 436.4 },
        levelled: true,
        frames: [
            { name: '000000.jpg', quaternion_wxyz: [1, 0, 0, 0], translation: [0, 0, 0] },
            { name: '000001.jpg', quaternion_wxyz: [1, 0, 0, 0], translation: [-2, 0, 0] },
        ],
        ...overrides,
    };
}

// A quaternion (w, x, y, z) for a turn of `degrees` about a unit axis.
function about(axis, degrees) {
    const half = (degrees * Math.PI) / 360;
    return [Math.cos(half), ...axis.map((value) => value * Math.sin(half))];
}

// Hamilton product of two (w, x, y, z) quaternions.
function times([aw, ax, ay, az], [bw, bx, by, bz]) {
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ];
}

// Turns a viewer-frame vector by a PlayCanvas quaternion [x, y, z, w].
function rotate([x, y, z, w], v) {
    const m = quaternionToMatrix([w, x, y, z]);
    return m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
}

// A small seeded random source, so the property tests below are the same on every run.
function seeded(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

test('the docs example parses, with each quaternion normalised', () => {
    const doc = example({ frames: [{ name: '000000.jpg', quaternion_wxyz: [0.99, 0.01, -0.12, 0.0], translation: [0.31, -0.02, 1.84] }] });
    const cameras = parseCameras(doc);
    assert.equal(cameras.levelled, true);
    assert.equal(cameras.camera.fy, 1290.2);
    assert.equal(cameras.frames.length, 1);
    assert.ok(Math.abs(Math.hypot(...cameras.frames[0].quaternion) - 1) < CLOSE);
    assert.deepEqual(cameras.frames[0].translation, [0.31, -0.02, 1.84]);
});

test('a broken cameras.json is refused, naming the field', () => {
    const cases = [
        [null, /JSON object/],
        [[], /JSON object/],
        [example({ camera: undefined }), /camera must be an object/],
        [example({ camera: { width: 10, height: 10, fx: 0, fy: 10, cx: 5, cy: 5 } }), /camera\.fx must be above 0/],
        [example({ camera: { width: 10, height: 10, fx: 10, fy: 10, cx: '5', cy: 5 } }), /camera\.cx must be a number/],
        [example({ levelled: 'yes' }), /levelled must be true or false/],
        [example({ frames: [] }), /at least one frame/],
        [example({ frames: [{ quaternion_wxyz: [1, 0, 0], translation: [0, 0, 0] }] }), /frames\[0\]\.quaternion_wxyz must be a list of 4/],
        [example({ frames: [{ quaternion_wxyz: [0, 0, 0, 0], translation: [0, 0, 0] }] }), /zero length/],
        [example({ frames: [{ quaternion_wxyz: [1, 0, 0, 0], translation: [0, null, 0] }] }), /frames\[0\]\.translation\[1\] must be a number/],
    ];
    for (const [value, message] of cases) {
        assert.throws(() => parseCameras(value), (error) => error instanceof CamerasError && message.test(error.message), String(message));
    }
});

test('a camera with no rotation starts at its centre, looking along the file\'s +z with up up', () => {
    // COLMAP's camera centre is -R^T t; with R = I and t = (1, 2, 3) it's (-1, -2, -3), which the
    // turn about z shows at (1, 2, -3).
    const cameras = parseCameras(example({ frames: [{ name: 'a', quaternion_wxyz: [1, 0, 0, 0], translation: [1, 2, 3] }] }));
    const view = startView(cameras);
    near(view.position, [1, 2, -3], 'position');
    near(forwardOf(view.yaw, view.pitch), [0, 0, 1], 'forward');
    assert.ok(Math.abs(view.pitch) < CLOSE);
    // The splat turns by half a turn about z: the levelled up (0, -1, 0) shows as +y.
    near(rotate(view.sceneRotation, [0, -1, 0]), [0, 1, 0], 'up');
    near(rotate(view.sceneRotation, [1, 0, 0]), [-1, 0, 0], 'x');
});

test("the field of view is the picture's vertical one", () => {
    const view = startView(parseCameras(example()));
    const expected = (2 * Math.atan(873 / 2 / 1290.2) * 180) / Math.PI;
    assert.ok(Math.abs(view.fov - expected) < CLOSE);
    const tall = startView(parseCameras(example({ camera: { width: 900, height: 1600, fx: 300, fy: 300, cx: 450, cy: 800 } })));
    assert.equal(tall.fov, 90);
});

test('the start uses the first frame, not a later one', () => {
    const cameras = parseCameras(example({
        frames: [
            { name: 'first', quaternion_wxyz: about([0, 1, 0], 90), translation: [0, 0, 0] },
            { name: 'second', quaternion_wxyz: [1, 0, 0, 0], translation: [5, 5, 5] },
        ],
    }));
    const view = startView(cameras);
    near(view.position, [0, 0, 0], 'position');
    // R about y by 90 degrees: the camera looks along R^T (0, 0, 1) = (-1, 0, 0) in the file,
    // which the turn about z shows as (1, 0, 0).
    near(forwardOf(view.yaw, view.pitch), [1, 0, 0], 'forward');
});

test("a rolled camera starts with the roll taken out, so the horizon is level", () => {
    // R = Rx(-10) Rz(25): rolled 25 degrees, then tilted 10 degrees about its own x. Its view
    // direction, R's third row, is (-sin 10 sin 25, -sin 10 cos 25, cos 10): a little up in the
    // file's y-down world, so the start looks up by asin(sin 10 cos 25) degrees.
    const q = times(about([1, 0, 0], -10), about([0, 0, 1], 25));
    const view = startView(parseCameras(example({ frames: [{ name: 'a', quaternion_wxyz: q, translation: [0, 0, 0] }] })));
    const m = quaternionToMatrix(q);
    const fileForward = [m[2][0], m[2][1], m[2][2]];
    near(forwardOf(view.yaw, view.pitch), [-fileForward[0], -fileForward[1], fileForward[2]], 'forward');
    const up = (Math.asin(Math.sin(Math.PI / 18) * Math.cos((25 * Math.PI) / 180)) * 180) / Math.PI;
    assert.ok(Math.abs(view.pitch - up) < 1e-6, `pitch ${view.pitch}`);
});

test('property: for any levelled first camera, the start looks its way from its centre, with no roll', () => {
    const random = seeded(7);
    for (let i = 0; i < 500; i++) {
        const q = [random() - 0.5, random() - 0.5, random() - 0.5, random() - 0.5];
        const t = [random() * 10 - 5, random() * 10 - 5, random() * 10 - 5];
        const cameras = parseCameras(example({ frames: [{ name: 'a', quaternion_wxyz: q, translation: t }] }));
        const view = startView(cameras);
        const m = quaternionToMatrix(cameras.frames[0].quaternion);
        // R^T t and R^T (0, 0, 1), from the columns of R.
        const centre = [0, 1, 2].map((j) => -(m[0][j] * t[0] + m[1][j] * t[1] + m[2][j] * t[2]));
        const forward = [m[2][0], m[2][1], m[2][2]];
        near(view.position, [-centre[0], -centre[1], centre[2]], `position ${i}`, 1e-6);
        if (Math.hypot(forward[0], forward[2]) > 1e-3) {
            near(forwardOf(view.yaw, view.pitch), [-forward[0], -forward[1], forward[2]], `forward ${i}`, 1e-6);
        }
        assert.ok(Math.abs(view.pitch) <= 90);
        near(rotate(view.sceneRotation, [0, -1, 0]), [0, 1, 0], `up ${i}`, 1e-9);
    }
});

test('property: an unlevelled scene is turned so the cameras\' mean up shows as +y', () => {
    const random = seeded(11);
    for (let i = 0; i < 200; i++) {
        // Every camera shares one random tilt of the world, plus a small wobble of its own.
        const tilt = [random() - 0.5, random() - 0.5, random() - 0.5, random() - 0.5];
        const n = Math.hypot(...tilt);
        const base = tilt.map((value) => value / n);
        const frames = [0, 1, 2, 3].map((k) => ({
            name: `${k}`,
            quaternion_wxyz: times(about([0, 0, 1], (random() - 0.5) * 10), times(about([0, 1, 0], k * 40), base)),
            translation: [k, 0, 0],
        }));
        const cameras = parseCameras(example({ levelled: false, frames }));
        const view = startView(cameras);
        const ups = cameras.frames.map((frame) => {
            const m = quaternionToMatrix(frame.quaternion);
            return [-m[1][0], -m[1][1], -m[1][2]];
        });
        const mean = [0, 1, 2].map((j) => ups.reduce((sum, up) => sum + up[j], 0) / ups.length);
        const length = Math.hypot(...mean);
        near(rotate(view.sceneRotation, mean.map((value) => value / length)), [0, 1, 0], `mean up ${i}`, 1e-9);
    }
});

test('a levelled scene is never turned by more than the half turn about z', () => {
    const rolled = parseCameras(example({ frames: [{ name: 'a', quaternion_wxyz: about([0, 0, 1], 90), translation: [0, 0, 0] }] }));
    near(startView(rolled).sceneRotation.map(Math.abs), [0, 0, 1, 0], 'rotation');
});

test('the scene length is the spread of the camera centres', () => {
    const view = startView(parseCameras(example()));
    // Centres at (0, 0, 0) and (2, 0, 0) in the file.
    assert.ok(Math.abs(view.sceneLength - 2) < CLOSE);
    const one = startView(parseCameras(example({ frames: [{ name: 'a', quaternion_wxyz: [1, 0, 0, 0], translation: [0, 0, 0] }] })));
    assert.equal(one.sceneLength, 1);
});

test('a camera looking straight down still gets a heading', () => {
    const view = startView(parseCameras(example({ frames: [{ name: 'a', quaternion_wxyz: about([1, 0, 0], 90), translation: [0, 0, 0] }] })));
    assert.ok(Number.isFinite(view.yaw));
    assert.ok(Math.abs(Math.abs(view.pitch) - 90) < 0.1);
});

test('with no cameras file, the start frames the splat from outside its box', () => {
    const view = fallbackView([1, 2, 3], [3, 0, 4]);
    near(view.position, [-1, -2, 3 + 5 * 2.5], 'position');
    assert.equal(view.yaw, 0);
    assert.equal(view.pitch, 0);
    near(rotate(view.sceneRotation, [0, -1, 0]), [0, 1, 0], 'up');
});

test('matrixToQuaternion undoes quaternionToMatrix', () => {
    const random = seeded(3);
    for (let i = 0; i < 200; i++) {
        const q = [random() - 0.5, random() - 0.5, random() - 0.5, random() - 0.5];
        const n = Math.hypot(...q);
        const [w, x, y, z] = q.map((value) => value / n);
        const back = matrixToQuaternion(quaternionToMatrix([w, x, y, z]));
        const sign = Math.sign(back[3] * w + back[0] * x + back[1] * y + back[2] * z);
        near(back.map((value) => value * sign), [x, y, z, w], `quaternion ${i}`, 1e-9);
    }
});
