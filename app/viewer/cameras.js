// The job's out/cameras.json and the camera the viewer starts at. Pure maths with no engine and
// no DOM, so the tests run it in Node.
//
// Frames:
// - The file's frame is COLMAP's levelled frame (docs/pipeline.md, "Step: cameras"). A pose is
//   cam_from_world, x_cam = R x_world + t, and a camera looks along its +z with +y down its
//   picture. After levelling, world up is (0, -1, 0).
// - The viewer's frame is PlayCanvas's: +y up, and a camera with no rotation looks along -z.
// - The viewer turns the splat by the scene rotation, so file point p shows at S p. S is a turn of
//   180 degrees about z, (x, y, z) -> (-x, -y, z), which takes the levelled up (0, -1, 0) to +y.
//   It's the same turn the SuperSplat viewer gives every .ply and .sog. When the cameras file says
//   the levelling didn't move the scene, S first turns the cameras' mean up onto (0, -1, 0).
// - The starting camera has only a yaw and a pitch, no roll, so up stays up.

export class CamerasError extends Error {}

const FLIP_Z = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]];
const LEVELLED_UP = [0, -1, 0];

// The starting camera's field of view is the picture's vertical one, kept in this range so a
// tall portrait video or a very wide one still gives a usable view, in degrees.
const MIN_FOV = 30;
const MAX_FOV = 90;
// With no cameras file, the camera starts this many bounding radii from the splat's centre.
const FALLBACK_DISTANCE = 2.5;

// ---- small vector and matrix helpers, 3-vectors as arrays and 3 by 3 matrices as rows ----

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a) => Math.hypot(a[0], a[1], a[2]);
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const normalise = (a) => scale(a, 1 / length(a));
const mulMat = (m, n) => m.map((row) => [0, 1, 2].map((j) => row[0] * n[0][j] + row[1] * n[1][j] + row[2] * n[2][j]));
const mulVec = (m, v) => m.map((row) => dot(row, v));
const transpose = (m) => [0, 1, 2].map((i) => [m[0][i], m[1][i], m[2][i]]);

// The rotation matrix of a unit quaternion (w, x, y, z), Hamilton's convention as COLMAP writes it.
export function quaternionToMatrix([w, x, y, z]) {
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ];
}

// A rotation matrix as a unit quaternion [x, y, z, w], the order PlayCanvas's Quat takes.
export function matrixToQuaternion(m) {
    const trace = m[0][0] + m[1][1] + m[2][2];
    let q;
    if (trace > 0) {
        const s = 2 * Math.sqrt(trace + 1);
        q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4];
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const s = 2 * Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
        q = [s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s];
    } else if (m[1][1] > m[2][2]) {
        const s = 2 * Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]);
        q = [(m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s];
    } else {
        const s = 2 * Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]);
        q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4, (m[1][0] - m[0][1]) / s];
    }
    const n = Math.hypot(...q);
    return q.map((value) => value / n);
}

// The shortest rotation that turns unit vector a onto unit vector b.
function rotationBetween(a, b) {
    const c = dot(a, b);
    if (c > 1 - 1e-12) {
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    }
    if (c < -1 + 1e-12) {
        // Half a turn about any axis at right angles to a.
        const other = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        const k = normalise(cross(a, other));
        return [0, 1, 2].map((i) => [0, 1, 2].map((j) => 2 * k[i] * k[j] - (i === j ? 1 : 0)));
    }
    const v = cross(a, b);
    const vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]];
    const vx2 = mulMat(vx, vx);
    const f = 1 / (1 + c);
    return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + vx[i][j] + vx2[i][j] * f));
}

// ---- reading cameras.json ----

function finiteNumber(value, where) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new CamerasError(`cameras.json: ${where} must be a number`);
    }
    return value;
}

function numbers(value, count, where) {
    if (!Array.isArray(value) || value.length !== count) {
        throw new CamerasError(`cameras.json: ${where} must be a list of ${count} numbers`);
    }
    return value.map((item, i) => finiteNumber(item, `${where}[${i}]`));
}

// Checks the parsed JSON of out/cameras.json and gives it back with each quaternion normalised.
// Throws CamerasError, naming the field, for anything the viewer can't use.
export function parseCameras(value) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
        throw new CamerasError('cameras.json must hold a JSON object');
    }
    const cam = value.camera;
    if (cam === null || typeof cam !== 'object' || Array.isArray(cam)) {
        throw new CamerasError('cameras.json: camera must be an object');
    }
    const camera = {};
    for (const key of ['width', 'height', 'fx', 'fy', 'cx', 'cy']) {
        camera[key] = finiteNumber(cam[key], `camera.${key}`);
    }
    for (const key of ['width', 'height', 'fx', 'fy']) {
        if (camera[key] <= 0) {
            throw new CamerasError(`cameras.json: camera.${key} must be above 0`);
        }
    }
    if (typeof value.levelled !== 'boolean') {
        throw new CamerasError('cameras.json: levelled must be true or false');
    }
    if (!Array.isArray(value.frames) || value.frames.length === 0) {
        throw new CamerasError('cameras.json: frames must be a list with at least one frame');
    }
    const frames = value.frames.map((frame, i) => {
        if (frame === null || typeof frame !== 'object' || Array.isArray(frame)) {
            throw new CamerasError(`cameras.json: frames[${i}] must be an object`);
        }
        const q = numbers(frame.quaternion_wxyz, 4, `frames[${i}].quaternion_wxyz`);
        const n = Math.hypot(...q);
        if (n < 1e-9) {
            throw new CamerasError(`cameras.json: frames[${i}].quaternion_wxyz has zero length`);
        }
        return {
            name: typeof frame.name === 'string' ? frame.name : '',
            quaternion: q.map((item) => item / n),
            translation: numbers(frame.translation, 3, `frames[${i}].translation`),
        };
    });
    return { camera, levelled: value.levelled, frames };
}

// ---- the starting camera ----

// A frame's camera centre, view direction and picture up, in the file's frame.
function framePose(frame) {
    const rt = transpose(quaternionToMatrix(frame.quaternion));
    return {
        centre: scale(mulVec(rt, frame.translation), -1),
        forward: mulVec(rt, [0, 0, 1]),
        up: mulVec(rt, [0, -1, 0]),
    };
}

// The yaw and pitch, in degrees, of a camera that looks along the viewer-frame direction d with
// +y up. The camera's rotation is yaw about +y, then pitch about its own +x, so with both at 0
// it looks along -z, and its forward is (-sin yaw cos pitch, sin pitch, -cos yaw cos pitch).
export function yawPitchOf(d) {
    const n = normalise(d);
    const pitch = Math.asin(Math.max(-1, Math.min(1, n[1])));
    const yaw = Math.atan2(-n[0], -n[2]);
    return { yaw: (yaw * 180) / Math.PI, pitch: (pitch * 180) / Math.PI };
}

export function forwardOf(yaw, pitch) {
    const y = (yaw * Math.PI) / 180;
    const p = (pitch * Math.PI) / 180;
    return [-Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p)];
}

// A length that fits the scene, from the spread of the camera centres, for walking speed and the
// clip planes. COLMAP's scale is arbitrary, so the viewer can't use fixed metres.
function sceneLength(centres) {
    const lo = [0, 1, 2].map((i) => Math.min(...centres.map((c) => c[i])));
    const hi = [0, 1, 2].map((i) => Math.max(...centres.map((c) => c[i])));
    const spread = length(sub(hi, lo));
    if (spread > 1e-6) {
        return spread;
    }
    const reach = length(centres[0]);
    return reach > 1e-6 ? reach : 1;
}

// The scene rotation S as a matrix: the levelled frame needs only the turn about z. An unlevelled
// one first turns the cameras' mean picture up onto the levelled up, or the first camera's up
// when the mean is too short to trust.
function sceneMatrix(cameras, poses) {
    if (cameras.levelled) {
        return FLIP_Z;
    }
    const sum = poses.reduce((acc, pose) => [acc[0] + pose.up[0], acc[1] + pose.up[1], acc[2] + pose.up[2]], [0, 0, 0]);
    const mean = scale(sum, 1 / poses.length);
    const up = length(mean) > 0.2 ? normalise(mean) : normalise(poses[0].up);
    return mulMat(FLIP_Z, rotationBetween(up, LEVELLED_UP));
}

// Every frame's camera in the viewer's frame, in the file's order, from parseCameras' result.
//
// Returns:
//   sceneMatrix  S, the splat's turn, as a 3 by 3 matrix of rows
//   poses        [{ centre: [x, y, z], forward: unit [x, y, z] }, ...]
//   sceneLength  a length that fits the scene, from the camera centres
export function viewerPoses(cameras) {
    const filePoses = cameras.frames.map(framePose);
    const s = sceneMatrix(cameras, filePoses);
    const poses = filePoses.map((pose) => {
        let forward = mulVec(s, pose.forward);
        // A camera looking straight up or down has no heading. Lean it a little towards its own
        // picture's top, so the yaw is still well defined.
        if (Math.hypot(forward[0], forward[2]) < 1e-6) {
            const up = mulVec(s, pose.up);
            forward = normalise([forward[0] + up[0] * 1e-3, forward[1], forward[2] + up[2] * 1e-3]);
        }
        return { centre: mulVec(s, pose.centre), forward };
    });
    return { sceneMatrix: s, poses, sceneLength: sceneLength(poses.map((pose) => pose.centre)) };
}

// Where the viewer starts, from parseCameras' result: at the first frame's camera centre, looking
// its way with no roll, with its picture's vertical field of view.
//
// Returns, in the viewer's frame:
//   position       [x, y, z]
//   yaw, pitch     degrees, as yawPitchOf gives them
//   fov            vertical field of view in degrees
//   sceneRotation  the splat entity's rotation, a quaternion [x, y, z, w]
//   sceneLength    a length that fits the scene, from the camera centres
export function startView(cameras) {
    const { sceneMatrix: s, poses, sceneLength: fit } = viewerPoses(cameras);
    const { yaw, pitch } = yawPitchOf(poses[0].forward);
    const vertical = (2 * Math.atan(cameras.camera.height / 2 / cameras.camera.fy) * 180) / Math.PI;
    return {
        position: poses[0].centre,
        yaw,
        pitch,
        fov: Math.max(MIN_FOV, Math.min(MAX_FOV, vertical)),
        sceneRotation: matrixToQuaternion(s),
        sceneLength: fit,
    };
}

// Where the viewer starts with no usable cameras file: outside the splat's bounding box, looking
// at its centre along -z. centre and halfExtents are the box in the file's frame, as the engine
// reads it, and the scene rotation is the plain turn about z.
export function fallbackView(centre, halfExtents) {
    const radius = Math.max(length(halfExtents), 1e-3);
    const middle = mulVec(FLIP_Z, centre);
    return {
        position: [middle[0], middle[1], middle[2] + radius * FALLBACK_DISTANCE],
        yaw: 0,
        pitch: 0,
        fov: 60,
        sceneRotation: matrixToQuaternion(FLIP_Z),
        sceneLength: radius * 2,
    };
}
