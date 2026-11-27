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
// lengths around it (shrunk to the splats inside), so every place a pumpkin can go is inside it.
export const GRID_CELLS = 112;
export const GRID_MARGIN = 0.75;
// Every cube within PATH_OPEN scene lengths of the filmed path is open, and a clump of fewer than
// MIN_CLUMP solid cubes is a floater.
export const PATH_OPEN = 0.05;
export const MIN_CLUMP = 12;
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

// Counts the splat centres into a grid over the box { lo, hi }, and works out which cubes are
// solid. centres is a flat list of x, y, z in the splat's own frame, as the engine keeps them;
// toViewer { rows, offset } takes a centre to the viewer's frame as rows x centre + offset, rows
// being 3 rows of 3.
//
// Three filters keep floaters and fog out of the walls, so nothing stops you in the open air:
// - Density: a cube is solid only when it holds at least MIN_SOLID_COUNT splats and SOLID_SHARE of
//   the mean count of the cubes that hold any. A wall packs many splats into a cube; a floater or
//   a wisp of fog spreads a few over many.
// - The phone's own path: options.keepOpen lists points the phone moved through, and every cube
//   within options.openRadius of one is open, because the phone was there.
// - Clumps: a group of solid cubes touching each other face to face with fewer than
//   options.minClump cubes in it is a floater, not a thing in the room, so it's open.
// The box first shrinks to the splats' own bounds inside it, so its cubes are as small as they
// can be.
//
// Returns { lo, cell, dims, counts, solid, mask }: the grid's low corner, its cube size, its cubes
// along x, y and z, the splats in each cube, the count from which a cube is dense enough, and 1
// for each cube that's solid after the filters.
export function buildOpenGrid(centres, toViewer, box, options = {}) {
    const cells = options.cells ?? GRID_CELLS;
    const [r0, r1, r2] = toViewer.rows;
    const [o0, o1, o2] = toViewer.offset;
    const at = (i, out) => {
        const x = centres[i];
        const y = centres[i + 1];
        const z = centres[i + 2];
        out[0] = r0[0] * x + r0[1] * y + r0[2] * z + o0;
        out[1] = r1[0] * x + r1[1] * y + r1[2] * z + o1;
        out[2] = r2[0] * x + r2[1] * y + r2[2] * z + o2;
        return out;
    };
    const q = [0, 0, 0];
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i + 2 < centres.length; i += 3) {
        at(i, q);
        if (q[0] < box.lo[0] || q[1] < box.lo[1] || q[2] < box.lo[2] || q[0] > box.hi[0] || q[1] > box.hi[1] || q[2] > box.hi[2]) {
            continue;
        }
        for (let c = 0; c < 3; c++) {
            lo[c] = Math.min(lo[c], q[c]);
            hi[c] = Math.max(hi[c], q[c]);
        }
    }
    for (let c = 0; c < 3; c++) {
        if (!(lo[c] <= hi[c])) {
            lo[c] = box.lo[c];
            hi[c] = box.hi[c];
        }
    }
    const longest = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9);
    const cell = longest / cells;
    // One cube of room on every side, so a wall at the very edge is still inside.
    for (let c = 0; c < 3; c++) {
        lo[c] -= cell;
        hi[c] += cell;
    }
    const dims = [0, 1, 2].map((c) => Math.max(1, Math.ceil((hi[c] - lo[c]) / cell)));
    const [nx, ny, nz] = dims;
    const counts = new Uint16Array(nx * ny * nz);
    for (let i = 0; i + 2 < centres.length; i += 3) {
        at(i, q);
        const ix = Math.floor((q[0] - lo[0]) / cell);
        const iy = Math.floor((q[1] - lo[1]) / cell);
        const iz = Math.floor((q[2] - lo[2]) / cell);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
            continue;
        }
        const k = (iz * ny + iy) * nx + ix;
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
    const mask = new Uint8Array(counts.length);
    for (let k = 0; k < counts.length; k++) {
        mask[k] = counts[k] >= solid ? 1 : 0;
    }
    const grid = { lo, cell, dims, counts, solid, mask };
    if (options.keepOpen && options.openRadius > 0) {
        openAround(grid, options.keepOpen, options.openRadius);
    }
    dropClumps(grid, options.minClump ?? MIN_CLUMP);
    return grid;
}

// Opens every cube whose middle is within radius of one of the points.
function openAround(grid, points, radius) {
    const { lo, cell, dims, mask } = grid;
    const [nx, ny, nz] = dims;
    const limit = radius * radius;
    for (const p of points) {
        const from = [0, 1, 2].map((c) => Math.max(0, Math.floor((p[c] - radius - lo[c]) / cell)));
        const to = [0, 1, 2].map((c) => Math.min(dims[c] - 1, Math.floor((p[c] + radius - lo[c]) / cell)));
        for (let iz = from[2]; iz <= to[2]; iz++) {
            const dz = lo[2] + (iz + 0.5) * cell - p[2];
            for (let iy = from[1]; iy <= to[1]; iy++) {
                const dy = lo[1] + (iy + 0.5) * cell - p[1];
                for (let ix = from[0]; ix <= to[0]; ix++) {
                    const dx = lo[0] + (ix + 0.5) * cell - p[0];
                    if (dx * dx + dy * dy + dz * dz <= limit) {
                        mask[(iz * ny + iy) * nx + ix] = 0;
                    }
                }
            }
        }
    }
}

// Opens every group of solid cubes, touching face to face, that has fewer than minClump cubes.
function dropClumps(grid, minClump) {
    if (!(minClump > 1)) {
        return;
    }
    const { dims, mask } = grid;
    const [nx, ny] = dims;
    const plane = nx * ny;
    const seen = new Uint8Array(mask.length);
    const queue = new Int32Array(mask.length);
    for (let start = 0; start < mask.length; start++) {
        if (mask[start] !== 1 || seen[start]) {
            continue;
        }
        let head = 0;
        let tail = 0;
        queue[tail++] = start;
        seen[start] = 1;
        while (head < tail) {
            const k = queue[head++];
            const ix = k % nx;
            const iy = Math.floor(k / nx) % ny;
            const iz = Math.floor(k / plane);
            const next = [
                ix > 0 ? k - 1 : -1, ix < nx - 1 ? k + 1 : -1,
                iy > 0 ? k - nx : -1, iy < ny - 1 ? k + nx : -1,
                iz > 0 ? k - plane : -1, iz < dims[2] - 1 ? k + plane : -1,
            ];
            for (const j of next) {
                if (j >= 0 && mask[j] === 1 && !seen[j]) {
                    seen[j] = 1;
                    queue[tail++] = j;
                }
            }
        }
        if (tail < minClump) {
            for (let i = 0; i < tail; i++) {
                mask[queue[i]] = 0;
            }
        }
    }
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
    return grid.mask[(iz * ny + iy) * nx + ix] === 1;
}

// Whether no solid cube comes within clearance of point p. With no grid, everything is open.
export function isOpen(grid, p, clearance) {
    if (grid === null) {
        return true;
    }
    const { lo, cell, dims, mask } = grid;
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
                if (dx * dx + dy * dy + dz * dz <= limit && mask[(iz * dims[1] + iy) * dims[0] + ix] === 1) {
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
// is the pumpkin's radius. keepAway { point, distance } is a place it must be at least that far
// from, like the player, so a new pumpkin doesn't appear on top of them. Of CANDIDATES random
// tries that pass every rule, it keeps the one furthest from the others, so the pumpkins spread out.
//
// Returns [x, y, z], or null when no try passed.
export function placePumpkin({ path, grid, view, others = [], radius, keepAway = null, random = Math.random, candidates = CANDIDATES }) {
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
        if (dist(p, view.position) < nearest || !isOpen(grid, p, clearance)
            || (keepAway !== null && dist(p, keepAway.point) < keepAway.distance)) {
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
