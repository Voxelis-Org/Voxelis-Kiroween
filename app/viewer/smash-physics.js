// Pumpkin smash's physics, written by hand: the player's body bumping into walls and things in
// the room, standing on the floor with gravity and jumping, or flying; the pumpkins chasing the
// player around the walls; whether a pumpkin can see the player; and the camera behind the player.
// Pure maths with no engine and no DOM, so the tests run it in Node. Everything is in the
// viewer's frame (+y up), with lengths in the scene's own units.
//
// The walls are smash-place.js's grid of solid cubes, which already leaves out floaters and fog,
// and opens everything the phone moved through. Without a grid there are no walls and no floor,
// so the player flies freely.

import { isOpen, openDepth } from './smash-place.js';

// The body, as shares of the player's height: the phone was held at EYE_SHARE of it, the body is
// RADIUS_SHARE wide either side, a waist sphere sits at WAIST_SHARE and anything below STEP_SHARE
// is stepped up onto.
export const EYE_SHARE = 0.93;
export const RADIUS_SHARE = 0.13;
export const WAIST_SHARE = 0.45;
export const STEP_SHARE = 0.3;
// The head keeps this many body radii from anything, so walking up to a wall stops before the
// wall's splats fill the view.
export const HEAD_RADIUS = 1.8;
// The eye height is looked for between these shares of the scene length.
export const MIN_EYE = 0.12;
export const MAX_EYE = 0.6;
// Gravity pulls at GRAVITY_SHARE player heights a second squared (about 9.8 m/s² for 1.7 m), and
// a jump rises JUMP_SHARE of the height.
export const GRAVITY_SHARE = 5.8;
export const JUMP_SHARE = 0.25;
// A floor is looked for this many player heights below the feet, and at least this share of the
// path's samples must find one for gravity to be trusted.
const FLOOR_SEARCH = 2;
const FLOOR_FOUND = 0.3;
const PATH_SAMPLES = 60;
// Moves are cut into steps of at most this share of a grid cube, so nothing passes through a wall.
const SUB_STEP = 0.4;
const MAX_PITCH = 89;

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// The top of the highest solid cube under point p, from p's height down by at most maxDrop, as a
// height; null when there's none. It looks at the column under p and at four around it, radius
// away, so a missing cube in the floor doesn't make a hole to fall through.
export function floorUnder(grid, p, radius, maxDrop) {
    if (grid === null) {
        return null;
    }
    const { lo, cell, dims, mask } = grid;
    const [nx, ny, nz] = dims;
    let best = null;
    const offsets = [[0, 0], [radius, 0], [-radius, 0], [0, radius], [0, -radius]];
    const top = Math.floor((p[1] - lo[1]) / cell);
    const bottom = Math.floor((p[1] - maxDrop - lo[1]) / cell);
    for (const [ox, oz] of offsets) {
        const ix = Math.floor((p[0] + ox - lo[0]) / cell);
        const iz = Math.floor((p[2] + oz - lo[2]) / cell);
        if (ix < 0 || iz < 0 || ix >= nx || iz >= nz) {
            continue;
        }
        for (let iy = Math.min(top, ny - 1); iy >= Math.max(bottom, 0); iy--) {
            if (mask[(iz * ny + iy) * nx + ix] === 1) {
                const height = Math.min(lo[1] + (iy + 1) * cell, p[1]);
                if (best === null || height > best) {
                    best = height;
                }
                break;
            }
        }
    }
    return best;
}

// The player's body for a scene, from where the floor is under the filmed path: the phone was
// held at eye height, so the median drop from the path to the floor is the eye height.
//
// Returns { height, eye, radius, step, gravity, jump, fallbackFloor }, lengths in scene units and
// fallbackFloor the median floor height under the path, used where no floor is found. With no
// grid, or a floor under too few of the path's points, gravity is null and the player only flies.
export function bodyFor(grid, path, sceneLength) {
    const guessHeight = (0.3 * sceneLength) / EYE_SHARE;
    const plain = (height) => ({
        height,
        eye: height * EYE_SHARE,
        radius: height * RADIUS_SHARE,
        step: height * STEP_SHARE,
        gravity: null,
        jump: 0,
        fallbackFloor: null,
    });
    if (grid === null) {
        return plain(guessHeight);
    }
    const drops = [];
    const floors = [];
    const every = Math.max(1, Math.floor(path.positions.length / PATH_SAMPLES));
    let tried = 0;
    for (let k = 0; k < path.positions.length; k += every) {
        tried += 1;
        const p = path.positions[k];
        const floor = floorUnder(grid, p, 0, MAX_EYE * sceneLength);
        if (floor !== null && p[1] - floor >= MIN_EYE * sceneLength) {
            drops.push(p[1] - floor);
            floors.push(floor);
        }
    }
    if (drops.length < FLOOR_FOUND * tried || drops.length === 0) {
        return plain(guessHeight);
    }
    const eye = median(drops);
    const height = eye / EYE_SHARE;
    return {
        ...plain(height),
        gravity: GRAVITY_SHARE * height,
        jump: Math.sqrt(2 * GRAVITY_SHARE * height * JUMP_SHARE * height),
        fallbackFloor: median(floors),
    };
}

// A body standing with its eye at the given point.
export function createBody(shape, eye) {
    return {
        feet: [eye[0], eye[1] - shape.eye, eye[2]],
        vy: 0,
        onGround: false,
        flying: shape.gravity === null,
    };
}

export function eyeOf(shape, body, out = [0, 0, 0]) {
    out[0] = body.feet[0];
    out[1] = body.feet[1] + shape.eye;
    out[2] = body.feet[2];
    return out;
}

// Whether the body fits with its feet at feet: walking, a waist and a head sphere; flying, the head
// sphere alone.
export function bodyFits(grid, shape, feet, flying) {
    if (grid === null) {
        return true;
    }
    const head = [feet[0], feet[1] + shape.eye, feet[2]];
    if (flying) {
        return isOpen(grid, head, shape.radius * HEAD_RADIUS);
    }
    const waist = [feet[0], feet[1] + shape.height * WAIST_SHARE, feet[2]];
    return isOpen(grid, waist, shape.radius) && isOpen(grid, head, shape.radius * HEAD_RADIUS);
}

// Moves feet by (dx, dy, dz) in small steps, sliding along whatever is in the way: when a step
// doesn't fit, it tries each axis on its own. Returns how far it got, as [dx, dy, dz].
function slide(grid, shape, feet, move, flying) {
    const length = Math.hypot(move[0], move[1], move[2]);
    if (length === 0) {
        return [0, 0, 0];
    }
    const limit = grid === null ? Infinity : grid.cell * SUB_STEP;
    const steps = Math.max(1, Math.ceil(length / limit));
    const step = move.map((value) => value / steps);
    const start = [...feet];
    const tryAt = [0, 0, 0];
    for (let n = 0; n < steps; n++) {
        const options = [step, [step[0], 0, 0], [0, 0, step[2]], [0, step[1], 0]];
        let moved = false;
        for (const option of options) {
            if (option[0] === 0 && option[1] === 0 && option[2] === 0) {
                continue;
            }
            tryAt[0] = feet[0] + option[0];
            tryAt[1] = feet[1] + option[1];
            tryAt[2] = feet[2] + option[2];
            if (bodyFits(grid, shape, tryAt, flying)) {
                feet[0] = tryAt[0];
                feet[1] = tryAt[1];
                feet[2] = tryAt[2];
                moved = true;
                break;
            }
        }
        if (!moved) {
            break;
        }
    }
    return [feet[0] - start[0], feet[1] - start[1], feet[2] - start[2]];
}

// Moves the body on by dt seconds. move is [dx, dy, dz], how far the controls want to go this
// frame; walking, only its x and z count, and gravity, the floor and a jump decide the height.
// Flying, it moves on all three. Changes body in place and returns it.
export function stepBody(grid, shape, body, move, dt, jump = false) {
    const flying = body.flying || shape.gravity === null;
    if (flying) {
        slide(grid, shape, body.feet, move, true);
        body.vy = 0;
        body.onGround = false;
        return body;
    }
    slide(grid, shape, body.feet, [move[0], 0, move[2]], false);
    if (jump && body.onGround) {
        body.vy = shape.jump;
    }
    body.vy -= shape.gravity * dt;
    const fall = body.vy * dt;
    const above = [body.feet[0], body.feet[1] + shape.step, body.feet[2]];
    let floor = floorUnder(grid, above, shape.radius * 0.6, shape.step + FLOOR_SEARCH * shape.height);
    if (floor === null) {
        floor = shape.fallbackFloor;
    }
    if (fall <= 0) {
        const next = body.feet[1] + fall;
        if (floor !== null && next <= floor) {
            body.feet[1] = floor;
            body.vy = 0;
            body.onGround = true;
        } else {
            body.feet[1] = next;
            body.onGround = false;
        }
    } else {
        const tryAt = [body.feet[0], body.feet[1] + fall, body.feet[2]];
        if (bodyFits(grid, shape, tryAt, false)) {
            body.feet[1] = tryAt[1];
        } else {
            body.vy = 0;
        }
        body.onGround = false;
    }
    return body;
}

// Puts a body that doesn't fit where it is at the nearest point of the path where it does, with
// its eye there. Returns whether it moved.
export function rescueBody(grid, shape, body, path) {
    if (bodyFits(grid, shape, body.feet, body.flying || shape.gravity === null)) {
        return false;
    }
    const eye = eyeOf(shape, body);
    const order = path.positions.map((p, k) => [dist(p, eye), k]).sort((a, b) => a[0] - b[0]);
    for (const [, k] of order) {
        const p = path.positions[k];
        const feet = [p[0], p[1] - shape.eye, p[2]];
        if (bodyFits(grid, shape, feet, true) || bodyFits(grid, shape, feet, false)) {
            body.feet = feet;
            body.vy = 0;
            return true;
        }
    }
    return false;
}

// ---- the pumpkins' chase ----

// Whether a pumpkin of this radius at p can see the eye: nothing solid on the line between them.
export function canSee(grid, eye, p, radius) {
    if (grid === null) {
        return true;
    }
    const d = dist(eye, p);
    if (d < 1e-9) {
        return true;
    }
    const direction = [(p[0] - eye[0]) / d, (p[1] - eye[1]) / d, (p[2] - eye[2]) / d];
    return openDepth(grid, eye, direction, d) >= d - radius - grid.cell;
}

// The turns, in degrees about +y, a chasing pumpkin tries when the straight way is blocked: first
// all of them to the side it's going round by, then to the other side.
const DETOURS = [35, 70, 110, 150];

// Moves a chasing pumpkin of this radius at p (changed in place) towards target at speed for dt
// seconds, around anything solid. memory { side } is the side it goes round things by, 1 or -1. It
// keeps to that side, following the edge of what's in the way, and changes only when that side is
// blocked all round, so it doesn't dither. Returns whether it moved.
export function chaseStep(grid, p, target, speed, radius, dt, memory) {
    const d = dist(p, target);
    const reach = Math.min(speed * dt, d);
    if (reach <= 0) {
        return false;
    }
    const limit = grid === null ? Infinity : grid.cell * SUB_STEP;
    const steps = Math.max(1, Math.ceil(reach / limit));
    const length = reach / steps;
    let movedAny = false;
    for (let n = 0; n < steps; n++) {
        const to = [target[0] - p[0], target[1] - p[1], target[2] - p[2]];
        const flat = Math.hypot(to[0], to[2]);
        const total = Math.hypot(to[0], to[1], to[2]);
        if (total < 1e-9) {
            break;
        }
        const yaw = Math.atan2(to[0], to[2]);
        const climb = to[1] / total;
        let moved = false;
        const turns = [0, ...DETOURS.map((d) => d * memory.side), ...DETOURS.map((d) => -d * memory.side)];
        for (const detour of turns) {
            const a = yaw + (detour * Math.PI) / 180;
            for (const lift of [climb, climb + 0.6, climb - 0.6]) {
                const horizontal = flat > 1e-9 ? Math.sqrt(Math.max(0, 1 - Math.min(1, lift * lift))) : 0;
                const dir = [Math.sin(a) * horizontal, Math.max(-1, Math.min(1, lift)), Math.cos(a) * horizontal];
                const n2 = Math.hypot(dir[0], dir[1], dir[2]) || 1;
                const next = [p[0] + (dir[0] / n2) * length, p[1] + (dir[1] / n2) * length, p[2] + (dir[2] / n2) * length];
                if (isOpen(grid, next, radius)) {
                    p[0] = next[0];
                    p[1] = next[1];
                    p[2] = next[2];
                    moved = true;
                    if (detour * memory.side < 0) {
                        memory.side = -memory.side;
                    }
                    break;
                }
            }
            if (moved) {
                break;
            }
        }
        if (!moved) {
            break;
        }
        movedAny = true;
    }
    return movedAny;
}

// ---- looking and the camera ----

// Mouse look while the pointer is locked: moving the mouse right turns right and moving it down
// looks down, by degrees for each unit the mouse moves.
export function mouseLook(view, dx, dy, degrees) {
    view.yaw = (view.yaw - dx * degrees) % 360;
    view.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, view.pitch - dy * degrees));
}

// A cube holding this many splats stops the camera behind you, even when it isn't solid enough to
// stop you: a thinly filmed wall, the back of one, or a wisp of fog right in front of the lens
// would cover the view.
export const CAMERA_COUNT = 1;

// How far the camera can go from origin along a unit direction before it meets a solid cube, a
// cube with CAMERA_COUNT splats or the edge of the grid, up to maxDistance. Cubes within skip of
// origin don't count, so splats right at your head don't keep the camera in it.
export function cameraDepth(grid, origin, direction, maxDistance, skip = 0) {
    const { lo, cell, dims, mask, counts } = grid;
    const [nx, ny, nz] = dims;
    const step = cell / 2;
    for (let d = step; d < maxDistance; d += step) {
        const ix = Math.floor((origin[0] + direction[0] * d - lo[0]) / cell);
        const iy = Math.floor((origin[1] + direction[1] * d - lo[1]) / cell);
        const iz = Math.floor((origin[2] + direction[2] * d - lo[2]) / cell);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
            return d;
        }
        const k = (iz * ny + iy) * nx + ix;
        if (mask[k] === 1 || (d > skip && counts[k] >= CAMERA_COUNT)) {
            return d;
        }
    }
    return maxDistance;
}

// The camera behind and a little to the right of the player's head, level with it whatever the
// pitch, pulled in when a wall, or a thin wall or fog the player could walk into, is closer than
// distance. Staying level keeps it inside the room: a phone's video rarely shows the ceiling, so
// above the walls there's often nothing to stop it. Returns [x, y, z].
export function followCamera(grid, head, view, distance, side) {
    const y = (view.yaw * Math.PI) / 180;
    const right = [Math.cos(y), 0, -Math.sin(y)];
    const back = [Math.sin(y), 0, Math.cos(y)];
    const sideRoom = grid === null ? side : Math.min(side, cameraDepth(grid, head, right, side + grid.cell) - grid.cell);
    const from = [head[0] + right[0] * Math.max(0, sideRoom), head[1], head[2] + right[2] * Math.max(0, sideRoom)];
    const room = grid === null ? distance : cameraDepth(grid, from, back, distance + grid.cell, side) - grid.cell;
    const reach = Math.max(0, Math.min(distance, room));
    return [from[0] + back[0] * reach, from[1], from[2] + back[2] * reach];
}

