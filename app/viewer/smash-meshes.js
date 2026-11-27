// Pumpkin smash's shapes, made in code with no model or image files: the pumpkin's 8 ribbed
// wedges, its stem and its carved face, the candy corn, a wrapped sweet and a sparkle. Each shape
// is plain arrays (positions, normals, colours with a glow in alpha, and triangle indices), so the
// tests can check them in Node and smash.js turns them into engine meshes.
//
// The pumpkin's parts are in units of its radius, with +y up and the face looking along +z. A
// wedge and the stem are centred on their own middle, so they spin about it when they fly apart,
// and each part says where its middle sits in the whole pumpkin. The candy corn is in units of its
// length, pointing along +z, and the sweet and the sparkle in units of their size.
//
// The alpha of a colour is how much that vertex glows: 1 is drawn at full colour whatever way it
// faces, 0 is shaded by which way it faces.

export const WEDGES = 8;
const WEDGE_ANGLE = (2 * Math.PI) / WEDGES;

// The pumpkin's shape: squat, with a dimple at the top and the bottom and a groove between wedges.
const HEIGHT = 0.8;
const DIMPLE = 0.16;
const GROOVE = 0.1;
const ROWS = 12;
const COLUMNS = 4;
// The face sits this far outside the skin, in radii, so the skin never covers it.
const FACE_LIFT = 0.035;

const ORANGE = [1.0, 0.46, 0.06];
const FLESH = [1.0, 0.8, 0.38];
const STEM_DARK = [0.3, 0.36, 0.1];
const STEM_LIGHT = [0.5, 0.55, 0.22];
const CARVED_GLOW = [1.0, 0.86, 0.3];
const CORN_BASE = [1.0, 0.78, 0.1];
const CORN_MIDDLE = [1.0, 0.45, 0.04];
const CORN_TIP = [1.0, 0.97, 0.88];

class Shape {
    constructor() {
        this.positions = [];
        this.normals = [];
        this.colors = [];
        this.indices = [];
    }

    vertex(p, n, color, glow) {
        this.positions.push(p[0], p[1], p[2]);
        this.normals.push(n[0], n[1], n[2]);
        this.colors.push(color[0], color[1], color[2], glow);
        return this.positions.length / 3 - 1;
    }

    triangle(a, b, c) {
        this.indices.push(a, b, c);
    }

    // The mean of the vertices, moved to the origin. Returns where the middle was.
    centre() {
        const n = this.positions.length / 3;
        const middle = [0, 0, 0];
        for (let i = 0; i < n; i++) {
            for (let c = 0; c < 3; c++) {
                middle[c] += this.positions[i * 3 + c] / n;
            }
        }
        for (let i = 0; i < n; i++) {
            for (let c = 0; c < 3; c++) {
                this.positions[i * 3 + c] -= middle[c];
            }
        }
        return middle;
    }

    build(extra = {}) {
        return {
            positions: new Float32Array(this.positions),
            normals: new Float32Array(this.normals),
            colors: new Float32Array(this.colors),
            indices: new Uint16Array(this.indices),
            ...extra,
        };
    }
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const scaled = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

function unit(v, fallback = [0, 1, 0]) {
    const n = Math.hypot(v[0], v[1], v[2]);
    return n > 1e-9 ? [v[0] / n, v[1] / n, v[2] / n] : fallback;
}

// A point on the pumpkin's skin at polar angle theta (0 at the top, pi at the bottom) and angle phi
// about +y from the front (+z), in radii.
export function skinPoint(theta, phi) {
    const s = Math.sin(theta);
    const c = Math.cos(theta);
    const lobe = 1 - (GROOVE * (1 - Math.cos(WEDGES * phi))) / 2;
    const radial = lobe * Math.pow(Math.max(s, 0), 0.85);
    const y = HEIGHT * c - DIMPLE * c * (1 - s) ** 2;
    return [radial * Math.sin(phi), y, radial * Math.cos(phi)];
}

function skinNormal(theta, phi) {
    if (theta < 1e-4 || theta > Math.PI - 1e-4) {
        return [0, Math.cos(theta) > 0 ? 1 : -1, 0];
    }
    const h = 1e-4;
    const a = skinPoint(theta + h, phi);
    const b = skinPoint(theta - h, phi);
    const c = skinPoint(theta, phi + h);
    const d = skinPoint(theta, phi - h);
    const n = unit(cross([c[0] - d[0], c[1] - d[1], c[2] - d[2]], [a[0] - b[0], a[1] - b[1], a[2] - b[2]]));
    const p = skinPoint(theta, phi);
    return n[0] * p[0] + n[1] * p[1] + n[2] * p[2] < 0 ? scaled(n, -1) : n;
}

// One wedge, the one in front (+z): its ribbed skin, darker in the grooves, and its two cut sides,
// pale and glowing like the inside of a carved pumpkin.
function wedgeShape() {
    const shape = new Shape();
    const half = WEDGE_ANGLE / 2;
    const grid = [];
    for (let i = 0; i <= ROWS; i++) {
        const theta = (Math.PI * i) / ROWS;
        const row = [];
        for (let j = 0; j <= COLUMNS; j++) {
            const phi = -half + (WEDGE_ANGLE * j) / COLUMNS;
            const groove = (1 - Math.cos(WEDGES * phi)) / 2;
            const shade = (1 - 0.4 * groove) * (0.8 + 0.2 * Math.sin(theta));
            row.push(shape.vertex(skinPoint(theta, phi), skinNormal(theta, phi), scaled(ORANGE, shade), 0.12));
        }
        grid.push(row);
    }
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLUMNS; j++) {
            shape.triangle(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1]);
            shape.triangle(grid[i][j], grid[i + 1][j + 1], grid[i][j + 1]);
        }
    }
    for (const side of [-1, 1]) {
        const phi = side * half;
        const normal = scaled([Math.cos(phi), 0, -Math.sin(phi)], side);
        let previous = null;
        for (let i = 0; i <= ROWS; i++) {
            const p = skinPoint((Math.PI * i) / ROWS, phi);
            const outer = shape.vertex(p, normal, FLESH, 0.55);
            const axis = shape.vertex([0, p[1], 0], normal, FLESH, 0.55);
            if (previous !== null) {
                shape.triangle(previous.axis, previous.outer, outer);
                shape.triangle(previous.axis, outer, axis);
            }
            previous = { outer, axis };
        }
    }
    const offset = shape.centre();
    return shape.build({ offset });
}

// The stem: a short, bent, six-sided stalk standing in the top dimple.
function stemShape() {
    const shape = new Shape();
    const sides = 6;
    const segments = 4;
    const bottom = 0.56;
    const height = 0.34;
    const rings = [];
    for (let s = 0; s <= segments; s++) {
        const t = s / segments;
        const radius = 0.11 - 0.04 * t;
        const lean = 0.1 * t * t;
        const color = STEM_DARK.map((value, c) => value + (STEM_LIGHT[c] - value) * t);
        const ring = [];
        for (let k = 0; k < sides; k++) {
            const a = (2 * Math.PI * k) / sides;
            const out = [Math.cos(a), 0, Math.sin(a)];
            ring.push(shape.vertex([lean + out[0] * radius, bottom + height * t, out[2] * radius], out, color, 0.1));
        }
        rings.push(ring);
    }
    for (let s = 0; s < segments; s++) {
        for (let k = 0; k < sides; k++) {
            const n = (k + 1) % sides;
            shape.triangle(rings[s][k], rings[s + 1][k], rings[s + 1][n]);
            shape.triangle(rings[s][k], rings[s + 1][n], rings[s][n]);
        }
    }
    const top = shape.vertex([0.1, bottom + height, 0], [0, 1, 0], STEM_LIGHT, 0.1);
    for (let k = 0; k < sides; k++) {
        shape.triangle(rings[segments][k], rings[segments][(k + 1) % sides], top);
    }
    const offset = shape.centre();
    return shape.build({ offset });
}

// A face shape's corner, given as angles on the front of the pumpkin: across (phi, + to the
// pumpkin's left as you look at it) and up (latitude), lifted just off the skin.
function onSkin(across, up) {
    const p = skinPoint(Math.PI / 2 - up, across);
    return scaled(p, 1 + FACE_LIFT);
}

// Splits a flat triangle given in face angles into 4 ** levels pieces before it goes on the skin,
// so it follows the curve instead of cutting into it.
function faceTriangle(shape, a, b, c, levels) {
    if (levels > 0) {
        const ab = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        const bc = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2];
        const ca = [(c[0] + a[0]) / 2, (c[1] + a[1]) / 2];
        faceTriangle(shape, a, ab, ca, levels - 1);
        faceTriangle(shape, ab, b, bc, levels - 1);
        faceTriangle(shape, ca, bc, c, levels - 1);
        faceTriangle(shape, ab, bc, ca, levels - 1);
        return;
    }
    const corners = [a, b, c].map(([across, up]) => {
        const p = onSkin(across, up);
        return shape.vertex(p, unit(p), CARVED_GLOW, 1);
    });
    shape.triangle(corners[0], corners[1], corners[2]);
}

// The carved face, glowing: two triangle eyes, a triangle nose and a grin with two teeth.
function faceShape() {
    const shape = new Shape();
    for (const side of [-1, 1]) {
        faceTriangle(shape, [side * 0.12, 0.16], [side * 0.42, 0.16], [side * 0.27, 0.42], 2);
    }
    faceTriangle(shape, [-0.07, 0.0], [0.07, 0.0], [0, 0.12], 1);
    const columns = 14;
    const width = 0.5;
    const teeth = new Set([4, 9]);
    for (let k = 0; k < columns; k++) {
        const left = -width + (2 * width * k) / columns;
        const right = -width + (2 * width * (k + 1)) / columns;
        const edge = (x) => {
            const curve = (x / width) ** 2;
            const middle = -0.26 + 0.14 * curve;
            const thick = 0.15 * (1 - curve) + 0.012;
            return { top: middle + thick / 2, bottom: middle - thick / 2 };
        };
        const l = edge(left);
        const r = edge(right);
        const bite = teeth.has(k) ? 0.055 : 0;
        faceTriangle(shape, [left, l.bottom], [right, r.bottom], [right, r.top - bite], 1);
        faceTriangle(shape, [left, l.bottom], [right, r.top - bite], [left, l.top - bite], 1);
    }
    return shape.build();
}

// A round-sided cone along +z from z = start to z = end, with the given radii, in one colour.
function band(shape, start, end, startRadius, endRadius, color, glow, sides) {
    const ring = (z, radius) => {
        const out = [];
        for (let k = 0; k < sides; k++) {
            const a = (2 * Math.PI * k) / sides;
            // A soft triangle in cross-section, like a real candy corn.
            const r = radius * (1 + 0.12 * Math.cos(3 * a));
            const normal = unit([Math.cos(a), Math.sin(a), (startRadius - endRadius) / Math.max(end - start, 1e-6)]);
            out.push(shape.vertex([Math.cos(a) * r, Math.sin(a) * r, z], normal, color, glow));
        }
        return out;
    };
    const a = ring(start, startRadius);
    const b = ring(end, endRadius);
    for (let k = 0; k < sides; k++) {
        const n = (k + 1) % sides;
        shape.triangle(a[k], b[k], b[n]);
        shape.triangle(a[k], b[n], a[n]);
    }
    return { a, b };
}

// The candy corn: a glowing cone with a yellow base, an orange middle and a white tip, 1 long.
function candyCornShape() {
    const shape = new Shape();
    const sides = 9;
    const radius = (z) => 0.3 * (0.5 - z) + 0.02;
    const { a: base } = band(shape, -0.5, -0.17, radius(-0.5), radius(-0.17), CORN_BASE, 0.75, sides);
    band(shape, -0.17, 0.17, radius(-0.17), radius(0.17), CORN_MIDDLE, 0.75, sides);
    const { b: tipRing } = band(shape, 0.17, 0.46, radius(0.17), radius(0.46), CORN_TIP, 0.75, sides);
    const tip = shape.vertex([0, 0, 0.5], [0, 0, 1], CORN_TIP, 0.75);
    const bottom = shape.vertex([0, 0, -0.5], [0, 0, -1], CORN_BASE, 0.75);
    for (let k = 0; k < sides; k++) {
        const n = (k + 1) % sides;
        shape.triangle(tipRing[k], tipRing[n], tip);
        shape.triangle(base[n], base[k], bottom);
    }
    return shape.build();
}

// A wrapped sweet, about 1 across: a round middle with a twist of wrapper at each end. It's white,
// so each sweet's own colour comes from its tint.
function sweetShape() {
    const shape = new Shape();
    const rings = 6;
    const sides = 8;
    const grid = [];
    for (let i = 0; i <= rings; i++) {
        const t = (Math.PI * i) / rings;
        const row = [];
        for (let k = 0; k < sides; k++) {
            const a = (2 * Math.PI * k) / sides;
            const n = [Math.cos(t), Math.sin(t) * Math.cos(a), Math.sin(t) * Math.sin(a)];
            row.push(shape.vertex([n[0] * 0.36, n[1] * 0.26, n[2] * 0.26], n, [0.85, 0.85, 0.85], 0.35));
        }
        grid.push(row);
    }
    for (let i = 0; i < rings; i++) {
        for (let k = 0; k < sides; k++) {
            const n = (k + 1) % sides;
            shape.triangle(grid[i][k], grid[i + 1][k], grid[i + 1][n]);
            shape.triangle(grid[i][k], grid[i + 1][n], grid[i][n]);
        }
    }
    for (const side of [-1, 1]) {
        const knot = shape.vertex([side * 0.32, 0, 0], [side, 0, 0], [1, 1, 1], 0.55);
        const rim = [];
        for (let k = 0; k < sides * 2; k++) {
            const a = (2 * Math.PI * k) / (sides * 2);
            const r = 0.26 * (k % 2 === 0 ? 1 : 0.7);
            const n = unit([side * 0.6, Math.cos(a), Math.sin(a)]);
            rim.push(shape.vertex([side * 0.62, Math.cos(a) * r, Math.sin(a) * r], n, [1, 1, 1], 0.55));
        }
        for (let k = 0; k < rim.length; k++) {
            shape.triangle(knot, rim[k], rim[(k + 1) % rim.length]);
        }
    }
    return shape.build();
}

// A sparkle: a little six-pointed star, 1 from tip to tip, glowing white.
function sparkleShape() {
    const shape = new Shape();
    const inner = 0.12;
    const tips = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    for (const tip of tips) {
        const axis = tip.findIndex((value) => value !== 0);
        const others = [0, 1, 2].filter((c) => c !== axis);
        const ring = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([u, v]) => {
            const p = [0, 0, 0];
            p[others[0]] = u * inner;
            p[others[1]] = v * inner;
            return p;
        });
        const top = shape.vertex(scaled(tip, 0.5), tip, [1, 1, 1], 1);
        const corners = ring.map((p) => shape.vertex(p, unit(p), [1, 1, 1], 1));
        for (let k = 0; k < 4; k++) {
            shape.triangle(top, corners[k], corners[(k + 1) % 4]);
        }
    }
    return shape.build();
}

// A box with its middle at centre and half sizes half, in one colour, with flat faces.
export function addBox(shape, centre, half, color, glow) {
    const faces = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    ];
    for (const [n, u, v] of faces) {
        const corner = (a, b) => [0, 1, 2].map((c) => centre[c] + half[c] * (n[c] + a * u[c] + b * v[c]));
        const ids = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => shape.vertex(corner(a, b), n, color, glow));
        shape.triangle(ids[0], ids[1], ids[2]);
        shape.triangle(ids[0], ids[2], ids[3]);
    }
}

const IRON = [0.13, 0.11, 0.1];
const LANTERN_GLASS = [1.0, 0.8, 0.38];

// The lantern you mustn't hit: a little iron lantern with four glowing amber panes, a pointed
// roof and a ring to hang it by, 1 tall and centred on its middle.
function lanternShape() {
    const shape = new Shape();
    const w = 0.22;
    addBox(shape, [0, -0.47, 0], [w + 0.03, 0.03, w + 0.03], IRON, 0.05);
    addBox(shape, [0, 0.3, 0], [w + 0.02, 0.025, w + 0.02], IRON, 0.05);
    for (const [x, z] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        addBox(shape, [x * w, -0.08, z * w], [0.03, 0.37, 0.03], IRON, 0.05);
    }
    // The glass, a little inside the posts, glowing.
    const g = w - 0.01;
    addBox(shape, [0, -0.08, 0], [g, 0.36, g], LANTERN_GLASS, 1);
    // The roof: four sloping triangles up to a point.
    const apex = [0, 0.47, 0];
    const corners = [[w + 0.02, 0.32, w + 0.02], [w + 0.02, 0.32, -w - 0.02], [-w - 0.02, 0.32, -w - 0.02], [-w - 0.02, 0.32, w + 0.02]];
    for (let k = 0; k < 4; k++) {
        const a = corners[k];
        const b = corners[(k + 1) % 4];
        const n = unit(cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [apex[0] - a[0], apex[1] - a[1], apex[2] - a[2]]));
        const out = n[0] * a[0] + n[2] * a[2] < 0 ? scaled(n, -1) : n;
        shape.triangle(shape.vertex(a, out, IRON, 0.05), shape.vertex(b, out, IRON, 0.05), shape.vertex(apex, out, IRON, 0.05));
        shape.triangle(shape.vertex(a, scaled(out, -1), IRON, 0.05), shape.vertex(apex, scaled(out, -1), IRON, 0.05), shape.vertex(b, scaled(out, -1), IRON, 0.05));
    }
    // The ring on top, as eight little bars.
    for (let k = 0; k < 8; k++) {
        const a = (2 * Math.PI * k) / 8;
        addBox(shape, [Math.cos(a) * 0.05, 0.53 + Math.sin(a) * 0.05, 0], [0.015, 0.015, 0.012], IRON, 0.05);
    }
    return shape.build();
}

// A shard of the lantern's glass: a thin glowing sliver, 1 long, seen from both sides.
function shardShape() {
    const shape = new Shape();
    const points = [[0, 0.5, 0], [-0.22, -0.4, 0.02], [0.25, -0.5, -0.02]];
    for (const side of [1, -1]) {
        const n = [0, 0, side];
        const ids = points.map((p) => shape.vertex(p, n, [1, 0.9, 0.6], 0.9));
        if (side > 0) {
            shape.triangle(ids[0], ids[1], ids[2]);
        } else {
            shape.triangle(ids[0], ids[2], ids[1]);
        }
    }
    return shape.build();
}

let cache = null;

// Every shape, made once: { wedge, stem, face, candyCorn, sweet, sparkle, lantern, shard }. The
// wedge and the stem also have offset, where their middle sits in the pumpkin.
export function shapes() {
    cache ??= {
        wedge: wedgeShape(),
        stem: stemShape(),
        face: faceShape(),
        candyCorn: candyCornShape(),
        sweet: sweetShape(),
        sparkle: sparkleShape(),
        lantern: lanternShape(),
        shard: shardShape(),
    };
    return cache;
}

export { Shape };

// The pumpkin's 9 parts that fly apart, as { kind, offset, turn }: where each part's middle sits
// in the pumpkin, in radii, and its turn as a quaternion [x, y, z, w]. Wedge k is the front wedge
// turned by k eighths of a turn about +y.
export function pumpkinParts() {
    const { wedge, stem } = shapes();
    const parts = [];
    for (let k = 0; k < WEDGES; k++) {
        const a = k * WEDGE_ANGLE;
        const c = Math.cos(a);
        const s = Math.sin(a);
        const o = wedge.offset;
        parts.push({
            kind: 'wedge',
            offset: [c * o[0] + s * o[2], o[1], -s * o[0] + c * o[2]],
            turn: [0, Math.sin(a / 2), 0, Math.cos(a / 2)],
        });
    }
    parts.push({ kind: 'stem', offset: [...stem.offset], turn: [0, 0, 0, 1] });
    return parts;
}
