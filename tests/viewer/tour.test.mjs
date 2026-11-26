// The tour's path maths and player, the ghost's screen maths and the loading screen's numbers.
// Run by tests/test_viewer.py with Node's own test runner, so no npm package is needed.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { forwardOf, parseCameras, startView, viewerPoses } from '../../app/viewer/cameras.js';
import { projectPoint, tapCounter } from '../../app/viewer/haunt.js';
import { loadingProgress } from '../../app/viewer/loading.js';
import {
    MAX_SECONDS, MIN_SECONDS, TURN_SPEED, angleDelta, buildTourPath, createTour, nearestTime, sampleTour,
} from '../../app/viewer/tour.js';

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const ahead = [0, 0, -1];

// Cameras walking along +x, n of them, step apart, all looking along -z.
function line(n, step = 0.1) {
    return Array.from({ length: n }, (_, i) => ({ centre: [i * step, 0, 0], forward: ahead }));
}

// The scene length cameras.js gives: the diagonal of the centres' bounding box.
function spread(poses) {
    const lo = [0, 1, 2].map((i) => Math.min(...poses.map((p) => p.centre[i])));
    const hi = [0, 1, 2].map((i) => Math.max(...poses.map((p) => p.centre[i])));
    return dist(lo, hi);
}

function pathOf(poses) {
    return buildTourPath(poses, spread(poses));
}

test('a short straight walk takes the shortest tour, at an even speed, from its start to its end', () => {
    const poses = line(41);
    const path = pathOf(poses);
    // Crossing the scene once would take PACE_SECONDS (8 s), so the walk is stretched to MIN_SECONDS.
    assert.ok(Math.abs(path.duration - MIN_SECONDS) < 1e-6, `duration ${path.duration}`);
    assert.ok(dist(sampleTour(path, 0).position, poses[0].centre) < 0.05);
    assert.ok(dist(sampleTour(path, path.duration).position, poses[40].centre) < 0.05);
    // Away from the ends the speed is steady.
    for (const t of [6, 9, 12]) {
        const moved = dist(sampleTour(path, t).position, sampleTour(path, t + 1).position);
        assert.ok(Math.abs(moved - path.speed) < path.speed * 0.02, `at ${t} s it moved ${moved}, not ${path.speed}`);
    }
    // And it only ever goes forwards.
    for (let t = 0; t < path.duration; t += 0.25) {
        assert.ok(sampleTour(path, t + 0.25).position[0] >= sampleTour(path, t).position[0] - 1e-9);
    }
});

test('a long walk is capped at MAX_SECONDS', () => {
    // Six laps of a circle, looking at its middle.
    const poses = Array.from({ length: 361 }, (_, i) => {
        const a = (i / 60) * 2 * Math.PI;
        return { centre: [Math.cos(a), 0, Math.sin(a)], forward: [-Math.cos(a), 0, -Math.sin(a)] };
    });
    const path = pathOf(poses);
    assert.ok(Math.abs(path.duration - MAX_SECONDS) < 0.5, `duration ${path.duration}`);
});

test('a pan on the spot still takes time, at no more than TURN_SPEED', () => {
    const poses = Array.from({ length: 31 }, (_, i) => ({ centre: [0, 0, 0], forward: forwardOf(i * 3, 0) }));
    const path = buildTourPath(poses, 1);
    // 90 degrees at 30 degrees a second, a little less because smoothing rounds off the ends.
    assert.ok(path.duration > 2.7 && path.duration <= 90 / TURN_SPEED + 1e-9, `duration ${path.duration}`);
    const turned = angleDelta(sampleTour(path, 0).yaw, sampleTour(path, path.duration).yaw);
    assert.ok(turned > 80 && turned <= 90, `turned ${turned}`);
    for (let t = 0; t < path.duration; t += 0.1) {
        const rate = Math.abs(angleDelta(sampleTour(path, t).yaw, sampleTour(path, t + 0.1).yaw)) / 0.1;
        assert.ok(rate <= TURN_SPEED * 1.05, `turning ${rate} degrees a second at ${t} s`);
    }
});

test('the hand\'s shake is smoothed out of the path', () => {
    // A straight walk with a 5 cm zigzag up and down at every pick.
    const poses = line(61).map((pose, i) => ({ ...pose, centre: [pose.centre[0], i % 2 ? 0.05 : -0.05, 0] }));
    const path = pathOf(poses);
    // The end points stay where they were filmed; everything a second or more inside is smooth.
    const inside = path.positions.slice(path.rate, -path.rate);
    const worst = Math.max(...inside.map((p) => Math.abs(p[1])));
    assert.ok(worst < 0.01, `the path still wobbles by ${worst}`);
});

test('there is no tour with fewer than two picks, or with every pick in the same pose', () => {
    assert.equal(buildTourPath([], 1), null);
    assert.equal(buildTourPath(line(1), 1), null);
    assert.equal(buildTourPath([{ centre: [1, 2, 3], forward: ahead }, { centre: [1, 2, 3], forward: ahead }], 1), null);
});

test('sampling clamps to the walk, and the samples are the path\'s, one every 1 / rate seconds', () => {
    const path = pathOf(line(41));
    assert.deepEqual(sampleTour(path, -5).position, sampleTour(path, 0).position);
    assert.deepEqual(sampleTour(path, path.duration + 5).position, path.positions[path.positions.length - 1]);
    assert.deepEqual(sampleTour(path, 3).position, path.positions[3 * path.rate]);
    const view = sampleTour(path, 4);
    assert.ok(dist(forwardOf(view.yaw, view.pitch), ahead) < 1e-9);
});

test('the nearest point is found along the path, and the earliest one on a tie', () => {
    const path = pathOf(line(41));
    const middle = sampleTour(path, 7.5).position;
    assert.ok(Math.abs(nearestTime(path, [middle[0], 0.5, 0.3]) - 7.5) < 1 / path.rate + 1e-9);
    assert.equal(nearestTime(path, [100, 0, 0]), path.duration);
    assert.equal(nearestTime(path, [-100, 0, 0]), 0);
    // A walk there and back passes every point twice: the first pass wins.
    const back = [...line(21), ...line(21).reverse().slice(1)];
    const loop = pathOf(back);
    const half = sampleTour(loop, loop.duration / 4).position;
    assert.ok(nearestTime(loop, half) < loop.duration / 2);
});

test('the player picks up from the nearest point, glides onto the path and stops at the end', () => {
    const path = pathOf(line(41));
    const tour = createTour(path);
    const near = sampleTour(path, 10).position;
    const view = { position: [near[0], 0.3, 0], yaw: 120, pitch: 20, fov: 60, sceneLength: 4 };
    tour.play(view);
    assert.ok(Math.abs(tour.time - 10) < 1 / path.rate + 1e-9);
    // The first frame hardly moves the view: the glide starts where the view was.
    tour.step(1 / 60, view);
    assert.ok(Math.abs(view.yaw - 120) < 1 && Math.abs(view.position[1] - 0.3) < 0.01);
    // After the glide the view is on the path, looking the path's way.
    for (let i = 0; i < 4 * 60; i++) {
        tour.step(1 / 60, view);
    }
    const pose = sampleTour(path, tour.time);
    assert.ok(dist(view.position, pose.position) < 1e-9);
    assert.ok(Math.abs(angleDelta(view.yaw, pose.yaw)) < 1e-6 && Math.abs(view.pitch - pose.pitch) < 1e-6);
    // It slows into the end and stops there.
    let ended = false;
    for (let i = 0; i < 60 * 60 && !ended; i++) {
        ended = tour.step(1 / 60, view);
    }
    assert.ok(ended && !tour.playing && tour.time === path.duration);
    // Play at the end starts again from the beginning.
    tour.play(view);
    assert.equal(tour.time, 0);
});

test('looking around while playing turns away from the path, then eases back, and pause keeps the view', () => {
    const path = pathOf(line(41));
    const tour = createTour(path);
    const view = { position: sampleTour(path, 0).position, yaw: 0, pitch: 0, fov: 60, sceneLength: 4 };
    tour.play(view);
    for (let i = 0; i < 60; i++) {
        tour.step(1 / 60, view);
    }
    // A drag by half the screen's height turns by half the field of view, 30 degrees.
    tour.look(-500, 0, 1000, 60);
    tour.step(1 / 60, view);
    assert.ok(Math.abs(angleDelta(sampleTour(path, tour.time).yaw, view.yaw) + 30) < 0.5, `yaw ${view.yaw}`);
    for (let i = 0; i < 5 * 60; i++) {
        tour.step(1 / 60, view);
    }
    assert.ok(Math.abs(angleDelta(sampleTour(path, tour.time).yaw, view.yaw)) < 1, `yaw ${view.yaw}`);
    tour.pause();
    const kept = structuredClone(view);
    assert.equal(tour.step(1 / 60, view), false);
    assert.deepEqual(view, kept);
});

test('a real cameras file gives a tour that starts at the starting camera', () => {
    const cameras = parseCameras({
        camera: { width: 1600, height: 900, fx: 1200, fy: 1200, cx: 800, cy: 450 },
        levelled: true,
        frames: Array.from({ length: 30 }, (_, i) => ({ name: `${i}`, quaternion_wxyz: [1, 0, 0, 0], translation: [-0.1 * i, 0, 0] })),
    });
    const start = startView(cameras);
    const path = buildTourPath(viewerPoses(cameras).poses, start.sceneLength);
    const first = sampleTour(path, 0);
    assert.ok(dist(first.position, start.position) < 0.05);
    assert.ok(Math.abs(angleDelta(first.yaw, start.yaw)) < 1e-6);
});

test('the ghost shows in the middle when straight ahead, to the right when right, and not when behind', () => {
    const view = { position: [0, 0, 0], yaw: 0, pitch: 0, fov: 60 };
    const middle = projectPoint(view, [0, 0, -2], 800, 600);
    assert.ok(Math.abs(middle.x - 400) < 1e-9 && Math.abs(middle.y - 300) < 1e-9 && Math.abs(middle.depth - 2) < 1e-9);
    const right = projectPoint(view, [1, 1, -2], 800, 600);
    assert.ok(right.x > 400 && right.y < 300);
    // A point at the top edge of a 60 degree view, tan(30 degrees) up per unit ahead.
    const top = projectPoint(view, [0, Math.tan(Math.PI / 6), -1], 800, 600);
    assert.ok(Math.abs(top.y) < 1e-9);
    assert.equal(projectPoint(view, [0, 0, 2], 800, 600), null);
    // Turned 90 degrees left (yaw +90 looks along -x), a point along -x is in the middle.
    const turned = projectPoint({ ...view, yaw: 90 }, [-3, 0, 0], 800, 600);
    assert.ok(Math.abs(turned.x - 400) < 1e-9);
});

test('three taps within a second turn haunted mode on, slower ones don\'t', () => {
    const taps = tapCounter();
    assert.deepEqual([0, 300, 600].map(taps), [false, false, true]);
    const slow = tapCounter();
    assert.deepEqual([0, 700, 1400, 2100].map(slow), [false, false, false, false]);
});

test('the loading screen shows the percent and megabytes, or only megabytes with no size', () => {
    assert.deepEqual(loadingProgress(12_345_678, 48_000_000), { fraction: 12_345_678 / 48_000_000, percent: '25%', detail: '12.3 of 48.0 MB' });
    assert.deepEqual(loadingProgress(3_200_000, 0), { fraction: null, percent: '', detail: '3.2 MB' });
    assert.equal(loadingProgress(60, 50).fraction, 1);
    assert.equal(loadingProgress(60, 50).percent, '100%');
});

// A small seeded random source, so the property test below is the same on every run.
function seeded(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

test('property: any filmed walk gives a tour from its first camera to its last, never faster than its pace', () => {
    const random = seeded(11);
    for (let run = 0; run < 40; run++) {
        // A wandering, shaky walk with a wandering heading, at a random scale.
        const scale = 0.1 + random() * 20;
        const n = 2 + Math.floor(random() * 80);
        let position = [0, 0, 0];
        let heading = random() * 360;
        const poses = [];
        for (let i = 0; i < n; i++) {
            poses.push({ centre: position, forward: forwardOf(heading, (random() - 0.5) * 30) });
            const a = random() * 2 * Math.PI;
            position = [position[0] + Math.cos(a) * 0.05 * scale, position[1] + (random() - 0.5) * 0.01 * scale, position[2] + Math.sin(a) * 0.05 * scale];
            heading += (random() - 0.5) * 20;
        }
        const path = pathOf(poses);
        if (path === null) {
            continue;
        }
        const where = `run ${run}, ${n} picks`;
        assert.ok(dist(sampleTour(path, 0).position, poses[0].centre) < 1e-9 * scale, where);
        assert.ok(dist(sampleTour(path, path.duration).position, poses[n - 1].centre) < 1e-9 * scale, where);
        assert.ok(Number.isFinite(path.duration) && path.duration > 0, where);
        for (let k = 1; k < path.positions.length; k++) {
            const step = dist(path.positions[k - 1], path.positions[k]);
            assert.ok(step <= (path.speed / path.rate) * 1.001 + 1e-12, `${where}: a step of ${step} at sample ${k}`);
            const f = path.forwards[k];
            assert.ok(Math.abs(Math.hypot(...f) - 1) < 1e-9, where);
        }
        // Every sample is its own nearest point, or an earlier one at the same place.
        for (let k = 0; k < path.positions.length; k += 7) {
            const p = path.positions[k];
            const t = nearestTime(path, p);
            assert.ok(t <= Math.min(k / path.rate, path.duration) + 1e-12, where);
            assert.ok(dist(sampleTour(path, t).position, p) < 1e-9 * scale, where);
        }
    }
});
