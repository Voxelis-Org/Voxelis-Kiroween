// Where pumpkin smash puts its pumpkins: in the open space the phone moved through and looked at,
// near the filmed path, spread out, never inside a wall, and never so close to the view that one
// fills the screen. Pure maths with no engine and no DOM, so the tests run it in Node. Everything
// is in the viewer's frame (+y up), with distances in scene lengths, the spread of the cameras the
// tour already uses.
//
// The walls come from the splat itself. Every splat's centre is counted into a grid of cubes
// around the filmed path, and a cube holding a fair share of splats is solid: a wall, the floor or
// a thing in the room. A pumpkin goes at a random point of the path, some way along the direction
// the phone looked from there, short of the first solid cube in that direction and with no solid
// cube within its clearance. With no splat centres to count, it trusts the looking direction for
// half the furthest distance.

import { forwardOf, yawPitchOf } from './cameras.js';

// The grid: this many cubes along its longest side, over the cameras' box and this many scene
// lengths around it, so every place a pumpkin can go is inside it.
export const GRID_CELLS = 64;
export const GRID_MARGIN = 0.75;
// A cube is solid when it holds at least MIN_SOLID_COUNT splats and at least SOLID_SHARE of the
// mean count of the cubes that hold any. Stray floaters in the air hold far fewer.
export const MIN_SOLID_COUNT = 3;
export const SOLID_SHARE = 0.1;

// How far along the looking direction a pumpkin goes, in scene lengths.
export const MIN_AHEAD = 0.1;
export const MAX_AHEAD = 0.6;
// With no grid, how much of the furthest distance the looking direction is trusted for.
const NO_GRID_TRUST = 0.5;
// Pumpkins keep this far apart, in scene lengths, and this many of their radii from anything solid.
export const MIN_SPACING = 0.15;
export const CLEARANCE = 1.6;
// A new pumpkin is at most this share of the view's height across.
export const MAX_SCREEN_SHARE = 0.25;
// However close the camera comes, a pumpkin is drawn at most this share of the view's height.
export const SCREEN_SHARE_LIMIT = 0.5;
// How far the looking direction is turned at random, in degrees, and the steepest it may point.
const YAW_JITTER = 30;
const PITCH_JITTER = 12;
const MAX_PITCH = 35;
// Half the tries start from the path near the camera, within this many seconds of the tour.
const NEAR_SHARE = 0.5;
const NEAR_SECONDS = 6;
export const CANDIDATES = 80;

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// The box around a list of points, grown by margin on every side: { lo, hi }.
export function boxAround(points, margin) {
    const lo = [0, 1, 2].map((c) => Math.min(...points.map((p) => p[c])) - margin);
    const hi = [0, 1, 2].map((c) => Math.max(...points.map((p) => p[c])) + margin);
    return { lo, hi };
}

// Counts the splat centres into a grid over the box { lo, hi }. centres is a flat list of x, y, z
// in the splat's own frame, as the engine keeps them; toViewer { rows, offset } takes a centre to
// the viewer's frame as rows x centre + offset, rows being 3 rows of 3.
//
// Returns { lo, cell, dims, counts, solid }: the grid's low corner, its cube size, its cubes along
// x, y and z, the splats in each cube, and the count from which a cube is solid.
export function buildOpenGrid(centres, toViewer, box, cells = GRID_CELLS) {
    const { lo, hi } = box;
    const longest = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9);
    const cell = longest / cells;
    const dims = [0, 1, 2].map((c) => Math.max(1, Math.ceil((hi[c] - lo[c]) / cell)));
    const counts = new Uint16Array(dims[0] * dims[1] * dims[2]);
    const [r0, r1, r2] = toViewer.rows;
    const [o0, o1, o2] = toViewer.offset;
    for (let i = 0; i + 2 < centres.length; i += 3) {
        const x = centres[i];
        const y = centres[i + 1];
        const z = centres[i + 2];
        const ix = Math.floor((r0[0] * x + r0[1] * y + r0[2] * z + o0 - lo[0]) / cell);
        const iy = Math.floor((r1[0] * x + r1[1] * y + r1[2] * z + o1 - lo[1]) / cell);
        const iz = Math.floor((r2[0] * x + r2[1] * y + r2[2] * z + o2 - lo[2]) / cell);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= dims[0] || iy >= dims[1] || iz >= dims[2]) {
            continue;
        }
        const k = (iz * dims[1] + iy) * dims[0] + ix;
        if (counts[k] < 65535) {
            counts[k] += 1;
        }
    }
    let filled = 0;
    let total = 0;
    for (const count of counts) {
        if (count > 0) {
            filled += 1;
            total += count;
        }
    }
    const mean = filled > 0 ? total / filled : 0;
    const solid = Math.max(MIN_SOLID_COUNT, Math.ceil(mean * SOLID_SHARE));
    return { lo: [...lo], cell, dims, counts, solid };
}

// Whether the cube holding point p is solid. Outside the grid counts as solid, as nothing is known
// about it.
export function solidAt(grid, p) {
    const ix = Math.floor((p[0] - grid.lo[0]) / grid.cell);
    const iy = Math.floor((p[1] - grid.lo[1]) / grid.cell);
    const iz = Math.floor((p[2] - grid.lo[2]) / grid.cell);
    const [nx, ny, nz] = grid.dims;
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
        return true;
    }
    return grid.counts[(iz * ny + iy) * nx + ix] >= grid.solid;
}

// Whether no solid cube comes within clearance of point p. With no grid, everything is open.
export function isOpen(grid, p, clearance) {
    if (grid === null) {
        return true;
    }
    const { lo, cell, dims, counts, solid } = grid;
    const from = [0, 1, 2].map((c) => Math.floor((p[c] - clearance - lo[c]) / cell));
    const to = [0, 1, 2].map((c) => Math.floor((p[c] + clearance - lo[c]) / cell));
    if (from.some((value) => value < 0) || to.some((value, c) => value >= dims[c])) {
        return false;
    }
    const limit = clearance * clearance;
    for (let iz = from[2]; iz <= to[2]; iz++) {
        const dz = Math.max(lo[2] + iz * cell - p[2], 0, p[2] - (lo[2] + (iz + 1) * cell));
        for (let iy = from[1]; iy <= to[1]; iy++) {
            const dy = Math.max(lo[1] + iy * cell - p[1], 0, p[1] - (lo[1] + (iy + 1) * cell));
            for (let ix = from[0]; ix <= to[0]; ix++) {
                const dx = Math.max(lo[0] + ix * cell - p[0], 0, p[0] - (lo[0] + (ix + 1) * cell));
                if (dx * dx + dy * dy + dz * dz <= limit && counts[(iz * dims[1] + iy) * dims[0] + ix] >= solid) {
                    return false;
                }
            }
        }
    }
    return true;
}

// How far from origin along the unit direction the first solid cube is, up to maxDistance. With
// no grid, NO_GRID_TRUST of maxDistance.
export function openDepth(grid, origin, direction, maxDistance) {
    if (grid === null) {
        return maxDistance * NO_GRID_TRUST;
    }
    const step = grid.cell / 2;
    const p = [0, 0, 0];
    for (let d = step; d < maxDistance; d += step) {
        for (let c = 0; c < 3; c++) {
            p[c] = origin[c] + direction[c] * d;
        }
        if (solidAt(grid, p)) {
            return d;
        }
    }
    return maxDistance;
}

// The nearest a new pumpkin of this radius comes to the camera, so that it's at most
// MAX_SCREEN_SHARE of the view's height across with a vertical field of view of fov degrees.
export function minViewDistance(radius, fov) {
    return radius / (MAX_SCREEN_SHARE * Math.tan((fov * Math.PI) / 360));
}

// How much to shrink a pumpkin that's distance from the camera, so it's never more than
// SCREEN_SHARE_LIMIT of the view's height across: 1 when it's far enough.
export function screenFit(distance, radius, fov) {
    const allowed = distance * Math.tan((fov * Math.PI) / 360) * SCREEN_SHARE_LIMIT;
    return Math.min(1, Math.max(0, allowed / Math.max(radius, 1e-12)));
}

function nearestIndex(positions, point) {
    let best = 0;
    let bestDistance = Infinity;
    positions.forEach((p, k) => {
        const d = dist(p, point);
        if (d < bestDistance) {
            bestDistance = d;
            best = k;
        }
    });
    return best;
}

// Finds a place for a new pumpkin. path is tour.js's path (positions and forwards, rate samples a
// second), grid is buildOpenGrid's or null, view is the camera { position, fov, sceneLength },
// others are the middles of the pumpkins already out (and any spot to stay away from), and radius
// is the pumpkin's radius. Of CANDIDATES random tries that pass every rule, it keeps the one
// furthest from the others, so the pumpkins spread out.
//
// Returns [x, y, z], or null when no try passed.
export function placePumpkin({ path, grid, view, others = [], radius, random = Math.random, candidates = CANDIDATES }) {
    const length = view.sceneLength;
    const nearest = minViewDistance(radius, view.fov);
    const spacing = MIN_SPACING * length;
    const clearance = CLEARANCE * radius;
    const close = MIN_AHEAD * length;
    const count = path.positions.length;
    const home = nearestIndex(path.positions, view.position);
    const reach = Math.round(NEAR_SECONDS * path.rate);
    let best = null;
    let bestScore = -Infinity;
    for (let n = 0; n < candidates; n++) {
        const k = random() < NEAR_SHARE
            ? Math.max(0, Math.min(count - 1, home + Math.round((random() * 2 - 1) * reach)))
            : Math.min(count - 1, Math.floor(random() * count));
        const from = path.positions[k];
        const look = yawPitchOf(path.forwards[k]);
        const pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, look.pitch + (random() * 2 - 1) * PITCH_JITTER));
        const direction = forwardOf(look.yaw + (random() * 2 - 1) * YAW_JITTER, pitch);
        const depth = openDepth(grid, from, direction, MAX_AHEAD * length);
        const far = Math.min(depth - clearance - (grid === null ? 0 : grid.cell), MAX_AHEAD * length);
        if (far < close) {
            continue;
        }
        const ahead = close + random() * (far - close);
        const p = [0, 1, 2].map((c) => from[c] + direction[c] * ahead);
        if (dist(p, view.position) < nearest || !isOpen(grid, p, clearance)) {
            continue;
        }
        let gap = Infinity;
        for (const other of others) {
            gap = Math.min(gap, dist(p, other));
        }
        if (gap < spacing) {
            continue;
        }
        const score = Math.min(gap, 3 * spacing) + random() * 0.1 * spacing;
        if (score > bestScore) {
            bestScore = score;
            best = p;
        }
    }
    return best;
}
