// The player's body in pumpkin smash, made in code with no model files: a person in an orange
// hoodie, jeans and white trainers, built from rounded parts joined at the shoulders, elbows,
// wrists, hips, knees and ankles, like an action figure. You see it from behind with V.
//
// humanShapes() gives each part's shape in units of the person's height, with the part's joint at
// the origin and the limb hanging down -y, the front looking along +z. humanPose() places every
// part for a walk, a jump, flying or a throw. Pure maths with no engine and no DOM, so the tests
// run it in Node.

import { Shape } from './smash-meshes.js';
import { quatAxis, quatMul, turnVector } from './smash-play.js';

const HOODIE = [0.93, 0.42, 0.08];
const HOODIE_DARK = [0.2, 0.17, 0.16];
const JEANS = [0.2, 0.3, 0.52];
const SKIN = [0.86, 0.66, 0.52];
const HAIR = [0.2, 0.12, 0.08];
const SHOE = [0.94, 0.94, 0.92];
const SOLE = [0.18, 0.18, 0.2];
const EYE = [0.08, 0.06, 0.06];

// The joints, in heights from the feet, with x to the person's left.
export const SHOULDER = [0.17, 0.79, 0];
export const HIP = [0.075, 0.5, 0];
export const NECK = 0.82;
export const WAIST = 0.55;
export const UPPER_ARM = 0.17;
export const FOREARM = 0.15;
export const THIGH = 0.245;
export const SHIN = 0.215;

const unit = (v) => {
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
};

// A rounded limb hanging from the origin down to -length, radius thick, in one colour, with
// optional bands of another colour at the ends.
function addLimb(shape, length, radius, color, glow = 0.05, endColor = null) {
    const sides = 10;
    const rings = [];
    // The profile: a half circle at each end and straight between.
    const profile = [];
    for (let k = 0; k <= 4; k++) {
        const a = (Math.PI / 2) * (k / 4);
        profile.push({ y: radius * (1 - Math.sin(a)) * 0.6, r: radius * Math.sin(a + 0.15) });
    }
    profile.push({ y: length * 0.5, r: radius });
    for (let k = 4; k >= 0; k--) {
        const a = (Math.PI / 2) * (k / 4);
        profile.push({ y: length - radius * (1 - Math.sin(a)) * 0.6, r: radius * Math.sin(a + 0.15) * 0.92 });
    }
    profile.forEach(({ y, r }, i) => {
        const ring = [];
        const band = endColor !== null && i >= profile.length - 3;
        for (let s = 0; s < sides; s++) {
            const a = (2 * Math.PI * s) / sides;
            const out = [Math.cos(a), 0, Math.sin(a)];
            const slope = i === 0 ? -1 : i === profile.length - 1 ? 1 : 0;
            ring.push(shape.vertex([out[0] * r, -y, out[2] * r], unit([out[0], -slope * 0.8, out[2]]), band ? endColor : color, glow));
        }
        rings.push(ring);
    });
    for (let i = 0; i + 1 < rings.length; i++) {
        for (let s = 0; s < sides; s++) {
            const n = (s + 1) % sides;
            shape.triangle(rings[i][s], rings[i + 1][s], rings[i + 1][n]);
            shape.triangle(rings[i][s], rings[i + 1][n], rings[i][n]);
        }
    }
    for (const [ring, y, ny] of [[rings[0], 0, 1], [rings[rings.length - 1], -length, -1]]) {
        const cap = shape.vertex([0, y, 0], [0, ny, 0], color, glow);
        for (let s = 0; s < sides; s++) {
            shape.triangle(cap, ring[s], ring[(s + 1) % sides]);
        }
    }
}

// An ellipsoid with its middle at centre and radii radii, coloured by colorAt(normal).
function addBlob(shape, centre, radii, colorAt, glow = 0.05, rows = 10, sides = 14) {
    const grid = [];
    for (let i = 0; i <= rows; i++) {
        const t = (Math.PI * i) / rows;
        const row = [];
        for (let s = 0; s < sides; s++) {
            const a = (2 * Math.PI * s) / sides;
            const n = [Math.sin(t) * Math.sin(a), Math.cos(t), Math.sin(t) * Math.cos(a)];
            const p = [centre[0] + n[0] * radii[0], centre[1] + n[1] * radii[1], centre[2] + n[2] * radii[2]];
            row.push(shape.vertex(p, n, colorAt(n), glow));
        }
        grid.push(row);
    }
    for (let i = 0; i < rows; i++) {
        for (let s = 0; s < sides; s++) {
            const n = (s + 1) % sides;
            shape.triangle(grid[i][s], grid[i + 1][s], grid[i + 1][n]);
            shape.triangle(grid[i][s], grid[i + 1][n], grid[i][n]);
        }
    }
}

function headShape() {
    const shape = new Shape();
    addLimb(shape, 0.05, 0.032, SKIN);
    addBlob(shape, [0, 0.09, 0], [0.068, 0.082, 0.074], (n) => (n[1] > 0.35 || (n[2] < -0.2 && n[1] > -0.5) ? HAIR : SKIN));
    for (const side of [-1, 1]) {
        addBlob(shape, [side * 0.026, 0.095, 0.067], [0.009, 0.012, 0.006], () => EYE, 0, 4, 6);
    }
    // The hood, bunched behind the neck.
    addBlob(shape, [0, 0.0, -0.045], [0.085, 0.04, 0.05], () => HOODIE, 0.05, 6, 10);
    return shape.build();
}

function torsoShape() {
    const shape = new Shape();
    const height = NECK - WAIST;
    const rows = 6;
    const sides = 16;
    const grid = [];
    for (let i = 0; i <= rows; i++) {
        const t = i / rows;
        const halfWidth = 0.125 + 0.04 * t - 0.03 * Math.max(0, t - 0.85) / 0.15;
        const halfDepth = 0.08 + 0.01 * Math.sin(Math.PI * t);
        const row = [];
        for (let s = 0; s < sides; s++) {
            const a = (2 * Math.PI * s) / sides;
            const n = [Math.sin(a), 0, Math.cos(a)];
            const band = t < 0.12 ? HOODIE_DARK : HOODIE;
            // A dark front pocket.
            const pocket = t > 0.15 && t < 0.4 && n[2] > 0.75 ? HOODIE_DARK : band;
            row.push(shape.vertex([n[0] * halfWidth, t * height, n[2] * halfDepth], n, pocket, 0.06));
        }
        grid.push(row);
    }
    for (let i = 0; i < rows; i++) {
        for (let s = 0; s < sides; s++) {
            const n = (s + 1) % sides;
            shape.triangle(grid[i][s], grid[i + 1][s], grid[i + 1][n]);
            shape.triangle(grid[i][s], grid[i + 1][n], grid[i][n]);
        }
    }
    for (const [row, y, ny] of [[grid[0], 0, -1], [grid[rows], height, 1]]) {
        const cap = shape.vertex([0, y, 0], [0, ny, 0], HOODIE, 0.06);
        for (let s = 0; s < sides; s++) {
            shape.triangle(cap, row[s], row[(s + 1) % sides]);
        }
    }
    // The shoulders, rounded.
    for (const side of [-1, 1]) {
        addBlob(shape, [side * 0.15, height - 0.035, 0], [0.05, 0.045, 0.07], () => HOODIE, 0.06, 6, 10);
    }
    return shape.build();
}

function pelvisShape() {
    const shape = new Shape();
    addBlob(shape, [0, 0, 0], [0.13, 0.075, 0.085], () => JEANS, 0.05, 6, 14);
    return shape.build();
}

function handShape() {
    const shape = new Shape();
    addBlob(shape, [0, -0.035, 0.005], [0.028, 0.04, 0.02], () => SKIN, 0.05, 6, 10);
    return shape.build();
}

function shoeShape() {
    const shape = new Shape();
    addBlob(shape, [0, -0.025, 0.035], [0.042, 0.03, 0.085], (n) => (n[1] < -0.45 ? SOLE : SHOE), 0.08, 6, 12);
    return shape.build();
}

function limbShape(length, radius, color, endColor = null) {
    const shape = new Shape();
    addLimb(shape, length, radius, color, 0.05, endColor);
    return shape.build();
}

let cache = null;

// Each part's shape, made once, in heights: head, torso, pelvis, upperArm, foreArm, hand, thigh,
// shin and shoe.
export function humanShapes() {
    cache ??= {
        head: headShape(),
        torso: torsoShape(),
        pelvis: pelvisShape(),
        upperArm: limbShape(UPPER_ARM, 0.042, HOODIE),
        foreArm: limbShape(FOREARM, 0.036, HOODIE, HOODIE_DARK),
        hand: handShape(),
        thigh: limbShape(THIGH, 0.06, JEANS),
        shin: limbShape(SHIN, 0.05, JEANS),
        shoe: shoeShape(),
    };
    return cache;
}

export const HUMAN_PARTS = ['head', 'torso', 'pelvis', 'upperArm', 'foreArm', 'hand', 'thigh', 'shin', 'shoe'];

const X = [1, 0, 0];
const Y = [0, 1, 0];
const Z = [0, 0, 1];

// The throw takes this long, in seconds: the arm goes up and back, then swings over.
export const THROW_POSE = 0.35;

// Where every part goes, for a person height tall standing at feet, facing the view's yaw in
// degrees. motion is { phase, stride, air, flying, throwAge }: the walk's phase in radians, how
// much of a full stride the walk has (0 standing, 1 walking), whether the feet are off the ground,
// whether flying, and the seconds since a throw began (Infinity for none).
//
// Returns [{ part, position, turn }], the joint of each part in the world and its turn, a
// quaternion [x, y, z, w]; each part is drawn at size height.
export function humanPose(feet, yaw, height, motion) {
    const { phase = 0, stride = 0, air = false, flying = false, throwAge = Infinity } = motion;
    const root = quatAxis(Y, (yaw * Math.PI) / 180 + Math.PI);
    const out = [];
    const local = (p) => {
        const w = turnVector(root, [p[0] * height, p[1] * height, p[2] * height]);
        return [feet[0] + w[0], feet[1] + w[1], feet[2] + w[2]];
    };
    const swing = Math.sin(phase) * 0.55 * stride;
    const bob = Math.abs(Math.sin(phase)) * 0.012 * stride;
    const lift = [0, bob, 0];
    const at = (p) => local([p[0] + lift[0], p[1] + lift[1], p[2] + lift[2]]);

    out.push({ part: 'pelvis', position: at([0, HIP[1] + 0.02, 0]), turn: [...root] });
    const lean = flying ? 0.25 : air ? 0.1 : 0.06 * stride;
    const torsoTurn = quatMul(root, quatAxis(X, lean));
    const waist = at([0, WAIST, 0]);
    out.push({ part: 'torso', position: waist, turn: torsoTurn });
    // The neck and shoulders ride on the torso's lean.
    const fromWaist = (p) => {
        const w = turnVector(torsoTurn, [p[0] * height, (p[1] - WAIST) * height, p[2] * height]);
        return [waist[0] + w[0], waist[1] + w[1], waist[2] + w[2]];
    };
    out.push({ part: 'head', position: fromWaist([0, NECK, 0]), turn: torsoTurn });

    // A limb chain from a joint: each turn is about the person's own x (forward is -angle) after a
    // small spread sideways about z.
    const chain = (start, turnBase, spread, angles, lengths, parts) => {
        let position = start;
        let turn = quatMul(turnBase, quatAxis(Z, spread));
        angles.forEach((angle, i) => {
            turn = quatMul(turn, quatAxis(X, angle));
            out.push({ part: parts[i], position, turn: [...turn] });
            if (i < lengths.length) {
                const d = turnVector(turn, [0, -lengths[i] * height, 0]);
                position = [position[0] + d[0], position[1] + d[1], position[2] + d[2]];
            }
        });
    };

    for (const side of [1, -1]) {
        // side 1 is the left, -1 the right.
        const legSwing = side * swing;
        let thigh = -legSwing;
        let knee = 0.1 + 0.55 * stride * Math.max(0, Math.sin(phase * side + 0.4) * -1 + 0.2);
        if (air) {
            thigh = -0.5;
            knee = 0.9;
        }
        if (flying) {
            thigh = 0.15 + side * 0.1;
            knee = 0.6;
        }
        const foot = -(thigh + knee) * 0.7;
        chain(at([side * HIP[0], HIP[1], 0]), root, side * 0.03, [thigh, knee, foot], [THIGH, SHIN], ['thigh', 'shin', 'shoe']);

        let shoulder = legSwing * 0.8;
        let elbow = -0.35 - 0.2 * stride;
        let spread = side * 0.12;
        if (flying) {
            shoulder = -0.3;
            elbow = -0.4;
            spread = side * 0.6;
        }
        if (side === -1 && throwAge < THROW_POSE) {
            // The right arm: up and back, then over and forward.
            const t = throwAge / THROW_POSE;
            const windUp = Math.min(1, t / 0.35);
            const over = Math.max(0, (t - 0.35) / 0.65);
            shoulder = -0.4 - 2.4 * windUp + 1.9 * over * over;
            elbow = -1.4 * windUp * (1 - over) - 0.15;
            spread = -0.15;
        }
        chain(fromWaist([side * SHOULDER[0], SHOULDER[1], 0]), torsoTurn, spread, [shoulder, elbow, 0], [UPPER_ARM, FOREARM], ['upperArm', 'foreArm', 'hand']);
    }
    return out;
}

// Where the right hand is in a pose from humanPose(), for the candy corn to start from.
export function rightHand(pose) {
    const hands = pose.filter((item) => item.part === 'hand');
    return hands.length === 2 ? [...hands[1].position] : null;
}
