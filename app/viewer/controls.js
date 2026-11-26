// How input moves the camera: turning, zooming and walking. Pure functions on a view object
// { position: [x, y, z], yaw, pitch, fov, sceneLength }, in the viewer's frame (+y up), with no
// engine and no DOM.

import { forwardOf } from './cameras.js';

// Looking further up or down than this, in degrees, would flip the view over.
const MAX_PITCH = 89;
// Walking speed, in scene lengths a second: crossing the cameras' path takes about 2.5 seconds.
const WALK_SPEED = 0.4;
// How far one wheel notch (100 pixels of scrolling) moves, in scene lengths.
const WHEEL_STEP = 0.08;
// How far a pinch moves, in scene lengths per screen height of change in the fingers' spread.
const PINCH_SPEED = 1.5;

// Turning: the picture follows the finger or the mouse, so dragging by a screen height turns by
// the field of view.
export function turn(view, dx, dy, screenHeight) {
    const degreesPerPixel = view.fov / Math.max(screenHeight, 1);
    view.yaw = (view.yaw + dx * degreesPerPixel) % 360;
    view.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, view.pitch + dy * degreesPerPixel));
}

// Moving along the view direction by a distance in scene lengths, forward when positive.
export function dolly(view, lengths) {
    const f = forwardOf(view.yaw, view.pitch);
    const d = lengths * view.sceneLength;
    view.position = [view.position[0] + f[0] * d, view.position[1] + f[1] * d, view.position[2] + f[2] * d];
}

// A wheel event's scroll in pixels, from its deltaY and deltaMode (0 pixels, 1 lines, 2 pages),
// turned into a dolly: scrolling up (a negative deltaY) moves forward.
export function wheelLengths(deltaY, deltaMode, screenHeight) {
    const pixels = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * screenHeight : deltaY;
    return (-pixels / 100) * WHEEL_STEP;
}

// A pinch: the fingers spreading apart by a screen height moves forward PINCH_SPEED lengths.
export function pinchLengths(spreadChange, screenHeight) {
    return (spreadChange / Math.max(screenHeight, 1)) * PINCH_SPEED;
}

// The keys that walk, by KeyboardEvent.code, so WASD works whatever the keyboard layout.
export const WALK_KEYS = {
    KeyW: 'forward', ArrowUp: 'forward',
    KeyS: 'back', ArrowDown: 'back',
    KeyA: 'left', ArrowLeft: 'left',
    KeyD: 'right', ArrowRight: 'right',
};

// Walking for dt seconds with the given set of held directions. Walking stays level: forward is
// the heading with the pitch left out, so looking down while walking doesn't sink into the floor.
export function walk(view, held, dt) {
    const ahead = (held.has('forward') ? 1 : 0) - (held.has('back') ? 1 : 0);
    const side = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
    if (ahead === 0 && side === 0) {
        return false;
    }
    const y = (view.yaw * Math.PI) / 180;
    const forward = [-Math.sin(y), 0, -Math.cos(y)];
    const right = [Math.cos(y), 0, -Math.sin(y)];
    const n = Math.hypot(ahead, side);
    const d = (WALK_SPEED * view.sceneLength * dt) / n;
    view.position = [
        view.position[0] + (forward[0] * ahead + right[0] * side) * d,
        view.position[1],
        view.position[2] + (forward[2] * ahead + right[2] * side) * d,
    ];
    return true;
}
