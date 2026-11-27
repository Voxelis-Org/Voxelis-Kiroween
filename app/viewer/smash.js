// Pumpkin smash, the viewer's Halloween game: glowing pumpkins float in the filmed place, and the
// ones that see you come after you. Throw glowing candy corn to smash them, and don't hit the
// lanterns. A round lasts 60 seconds: a smashed pumpkin scores 1, a broken lantern costs 2 and a
// pumpkin that catches you costs 1.
//
// createPumpkinSmash() adds the game's button to the viewer and runs it. The rules and motion are
// in smash-play.js, the walls and placement in smash-place.js, the body, chase and camera physics
// in smash-physics.js, the shapes in smash-meshes.js and smash-human.js, and the sounds in
// smash-sound.js. This file joins them to the engine and the page.
//
// - Like a PC game on a laptop: the mouse is locked to the view while a round runs, so moving it
//   looks around and a click throws at the aim dot in the middle. Esc frees the mouse and pauses;
//   Esc again, or End smash, ends the game. Phones keep drag to look and tap to throw.
// - Walls: you walk on the floor with gravity (Space jumps, G flies with Q and E), and you bump
//   into walls and things in the room. The pumpkins chase you around them. The walls come from
//   the splat, leaving out floaters and fog.
// - The light: the splat itself is dimmed and lit through the engine's gsplatModifyVS chunk on the
//   gsplat template material (app.scene.gsplat.material). Its modifySplatColor(center, color) gets
//   each splat's centre in the world. Ours keeps some of every splat's colour, more the more fired
//   up you are, and adds warm light from you, the pumpkins and the lanterns, falling off with
//   distance and flickering. Haunted mode's canvas filter would dim the pumpkins too, so it's off
//   while a round runs.
// - The pumpkins, lanterns, flying bits and your body are real meshes in the world layer, drawn
//   before the splat, which is drawn depth-tested over them, so a wall in front hides them. Every
//   shape is one instanced draw, and the bits come from pools made once, at most MAX_PARTICLES
//   sweets, sparkles and shards at a time.

import * as pc from './vendor/playcanvas.min.mjs';
import { projectPoint } from './haunt.js';
import { GRID_MARGIN, PATH_OPEN, boxAround, buildOpenGrid, isOpen, openDepth, placePumpkin, screenFit } from './smash-place.js';
import { WEDGES, pumpkinParts, shapes } from './smash-meshes.js';
import { HUMAN_PARTS, humanPose, humanShapes, rightHand } from './smash-human.js';
import { bodyFor, canSee, chaseStep, createBody, eyeOf, followCamera, mouseLook, rescueBody, stepBody } from './smash-physics.js';
import {
    CANDY_SIZE, FURY_CAUGHT, FURY_LANTERN, FURY_SMASH, FURY_THROW, LANTERN_COUNT, LANTERN_HIT,
    LANTERN_RESPAWN, LANTERN_SIZE, MAX_PARTICLES, MISS_DISTANCE, PUMPKIN_COUNT, PUMPKIN_RADIUS,
    RESPAWN_SECONDS, blankBit, blankPose, bitAt, chaseSpeed, clockText, createPool, createRound,
    createThrow, flashOf, flicker, furyAfter, furyKick, launchParts, launchShards, launchSweets,
    lightingFor, partPose, pointAlong, quatAxis, resultLines, screenRay, smoothstep, throwAt, throwHit,
    throwStart,
} from './smash-play.js';
import { createSounds } from './smash-sound.js';

// The room darkens over DARK_IN seconds and comes back over DARK_OUT. The pumpkins' light reaches
// LIGHT_REACH scene lengths and the lanterns' LANTERN_REACH.
const DARK_IN = 1.2;
const DARK_OUT = 0.8;
const LIGHT_REACH = 0.3;
const LANTERN_REACH = 0.22;
const LANTERN_LIGHT = 0.55;

// The pumpkins bob by BOB radii BOB_RATE times a second, sway SWAY degrees either side of facing
// the filmed path, and pop up over POP_IN seconds, one after another at the start.
const BOB = 0.12;
const BOB_RATE = 0.4;
const SWAY = 40;
const SWAY_RATE = 0.11;
const POP_IN = 0.45;
const POP_STAGGER = 0.12;
// A pumpkin starts chasing once it's on screen with nothing in the way, looked at SEE_EVERY
// seconds, or once you're within HEAR scene lengths. New ones appear at least KEEP_AWAY scene
// lengths from you.
const SEE_EVERY = 0.2;
const HEAR = 0.25;
const KEEP_AWAY = 0.35;
// A miss fades over MISS_FADE seconds once it's flown its way. A placement that finds no spot is
// tried again after RETRY seconds.
const MISS_FADE = 0.15;
const RETRY = 0.25;
const CANDY_SPIN = 14;
// A tap or a click with a free mouse moves at most TAP_SLOP pixels and lasts at most TAP_MS
// milliseconds.
const TAP_SLOP = 10;
const TAP_MS = 600;
// Mouse look, in degrees for each unit the mouse moves. An Esc this soon after the mouse was freed
// is the one that freed it, in milliseconds.
const MOUSE_DEGREES = 0.12;
const UNLOCK_ESC_MS = 600;
// The camera over your shoulder, in your heights: THIRD_BACK behind, THIRD_SIDE to the right, so
// you're to the left of the aim dot, and THIRD_UP above your eyes. Close, so it stays in the room.
const THIRD_BACK = 0.9;
const THIRD_SIDE = 0.3;
const THIRD_UP = 0.12;
// A pumpkin that's seen you waits this long, in seconds, before it comes after you.
const NOTICE = 0.8;
// The controls line shows for this long at the start, in seconds, and a score change for POP.
const HINT_SECONDS = 9;
const POP = 1.1;
// The pools: pieces for 8 bursts at a time, and 8 candy corns in the air.
const PIECES = 8 * (WEDGES + 1);
const THROWS = 8;
const STYLE_ID = 'splat-viewer-smash-style';

const PUMPKIN_LIGHTS = Array.from({ length: PUMPKIN_COUNT }, (_, i) => `uSmashLight${i}`);
const LANTERN_LIGHTS = Array.from({ length: LANTERN_COUNT }, (_, i) => `uSmashLantern${i}`);
const sum = (names, reach) => names.map((name) => `smashLight(${name}, center, ${reach})`).join('\n        + ');
const ROOM_GLSL = `
uniform vec4 uSmashRoom;
uniform vec4 uSmashReach;
uniform vec4 uSmashPlayer;
${[...PUMPKIN_LIGHTS, ...LANTERN_LIGHTS].map((name) => `uniform vec4 ${name};`).join('\n')}
const vec3 SMASH_PUMPKIN = vec3(1.0, 0.55, 0.2);
const vec3 SMASH_LANTERN = vec3(1.0, 0.82, 0.45);
const vec3 SMASH_PLAYER = vec3(1.0, 0.86, 0.68);

// One light at a splat's centre: xyz is where it is, w how bright, and inv is one over its reach
// squared.
float smashLight(vec4 light, vec3 p, float inv) {
    vec3 d = p - light.xyz;
    float x = clamp(1.0 - dot(d, d) * inv, 0.0, 1.0);
    return light.w * x * x;
}

void modifySplatCenter(inout vec3 center) {
}

void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
}

void modifySplatColor(vec3 center, inout vec4 color) {
    float pumpkins = ${sum(PUMPKIN_LIGHTS, 'uSmashRoom.a')};
    float lanterns = ${sum(LANTERN_LIGHTS, 'uSmashReach.y')};
    float player = smashLight(uSmashPlayer, center, uSmashReach.x);
    color.rgb *= uSmashRoom.rgb + SMASH_PUMPKIN * pumpkins + SMASH_LANTERN * lanterns + SMASH_PLAYER * player;
}
`;

// The parts' shader: each instance is placed by aPlace (xyz and a size), turned by aTurn (a
// quaternion) and tinted by aTint (rgb, and a in extra glow). A vertex that glows (its colour's
// alpha) is drawn at full colour; the rest are shaded by how much they face the camera.
const PART_VS = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec4 aColor;
attribute vec4 aPlace;
attribute vec4 aTurn;
attribute vec4 aTint;
uniform mat4 matrix_viewProjection;
uniform vec3 view_position;
varying vec3 vColor;

vec3 turnBy(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main(void) {
    vec3 world = aPlace.xyz + turnBy(aTurn, aPosition * aPlace.w);
    vec3 normal = turnBy(aTurn, aNormal);
    float facing = max(dot(normal, normalize(view_position - world)), 0.0);
    float shade = 0.35 + 0.5 * facing + 0.15 * max(normal.y, 0.0);
    float glow = clamp(aColor.a + aTint.a, 0.0, 1.0);
    vColor = aColor.rgb * aTint.rgb * mix(shade, 1.0, glow);
    gl_Position = matrix_viewProjection * vec4(world, 1.0);
}
`;
const PART_FS = `
varying vec3 vColor;

void main(void) {
    gl_FragColor = vec4(vColor, 1.0);
}
`;
// Floats per instance: aPlace, aTurn and aTint.
const FLOATS = 12;
const WHITE = [1, 1, 1];
const ANGRY_FACE = [1, 0.72, 0.5];

function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text) {
        node.textContent = text;
    }
    return node;
}

// The game's stylesheet, linked once per page from beside this module, so the page's
// Content-Security-Policy (style-src 'self') allows it.
function addStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }
    const link = document.createElement('link');
    link.id = STYLE_ID;
    link.rel = 'stylesheet';
    link.href = new URL('./smash.css', import.meta.url).href;
    document.head.append(link);
}

// One shape drawn as one instanced mesh, with room for capacity instances a frame.
function createGroup(device, material, node, format, shape, capacity) {
    const mesh = new pc.Mesh(device);
    mesh.setPositions(shape.positions);
    mesh.setNormals(shape.normals);
    mesh.setColors(shape.colors, 4);
    mesh.setIndices(shape.indices);
    mesh.update(pc.PRIMITIVE_TRIANGLES);
    const data = new Float32Array(capacity * FLOATS);
    const buffer = new pc.VertexBuffer(device, format, capacity, { usage: pc.BUFFER_DYNAMIC, data });
    const instance = new pc.MeshInstance(mesh, material, node);
    instance.setInstancing(buffer);
    instance.instancingCount = 0;
    instance.visible = false;
    let count = 0;
    return {
        instance,
        add(position, size, turn, tint = WHITE, glow = 0) {
            if (count >= capacity || !(size > 0)) {
                return;
            }
            const o = count * FLOATS;
            data[o] = position[0];
            data[o + 1] = position[1];
            data[o + 2] = position[2];
            data[o + 3] = size;
            data[o + 4] = turn[0];
            data[o + 5] = turn[1];
            data[o + 6] = turn[2];
            data[o + 7] = turn[3];
            data[o + 8] = tint[0];
            data[o + 9] = tint[1];
            data[o + 10] = tint[2];
            data[o + 11] = glow;
            count += 1;
        },
        // Sends this frame's instances to the GPU and draws that many.
        flush() {
            if (count > 0) {
                buffer.unlock(0, count * FLOATS * 4);
            }
            instance.instancingCount = count;
            instance.visible = count > 0;
            count = 0;
        },
        destroy() {
            instance.destroy();
            buffer.destroy();
        },
    };
}

// Adds pumpkin smash to a viewer whose splat has loaded and whose cameras gave a tour path.
//
// options:
//   app, container, canvas  the engine app, the viewer's element and its canvas
//   view                    the camera's view object, which the viewer keeps up to date
//   place()                 puts the engine's camera where view says
//   path                    tour.js's path of the filmed walk
//   splat, resource         the splat's entity and its loaded resource, whose centres show the walls
//   touch                   whether this is a touch screen
//   announce(text)          shows a short message
//   stopTour()              pauses the tour if it's playing
//   haunt                   haunted mode, { on, toggle(view) }
//   before                  the element the game's button goes in front of
//
// Returns { active, destroy() }. active is true while a round or its score is on screen.
export function createPumpkinSmash({ app, container, canvas, view, place, path, splat, resource, touch, announce, stopTour, haunt, before }) {
    addStyles();
    const device = app.graphicsDevice;
    const world = app.scene.layers.getLayerById(pc.LAYERID_WORLD);
    const template = app.scene.gsplat.material;
    const still = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    const sounds = createSounds();
    const round = createRound();
    const parts = pumpkinParts();
    const random = Math.random;
    const length = view.sceneLength;
    const radius = PUMPKIN_RADIUS * length;
    const lanternSize = LANTERN_SIZE * length;
    const wantLock = !touch && typeof canvas.requestPointerLock === 'function';

    // ---- the page ----

    const button = element('button', 'splat-viewer-smash');
    button.type = 'button';
    const hud = element('div', 'splat-viewer-smash-hud');
    hud.hidden = true;
    const scoreText = element('span', 'splat-viewer-smash-score');
    const fury = element('span', 'splat-viewer-smash-fury');
    const furyFill = element('span', 'splat-viewer-smash-fury-fill');
    fury.append(furyFill);
    fury.setAttribute('role', 'meter');
    fury.setAttribute('aria-label', 'How fired up you are');
    fury.setAttribute('aria-valuemin', '0');
    fury.setAttribute('aria-valuemax', '100');
    const clock = element('span', 'splat-viewer-smash-clock');
    clock.setAttribute('role', 'timer');
    const mute = element('button', 'splat-viewer-smash-mute');
    mute.type = 'button';
    hud.append(scoreText, fury, clock, mute);
    const pop = element('p', 'splat-viewer-smash-pop');
    pop.setAttribute('aria-live', 'polite');
    const aim = element('div', 'splat-viewer-smash-aim');
    aim.hidden = true;
    const flash = element('div', 'splat-viewer-smash-flash');
    flash.hidden = true;
    const keys = element('p', 'splat-viewer-smash-keys', touch
        ? "Drag to look, tap to throw, pinch to move. Don't hit the lanterns!"
        : "Mouse to look, click to throw, WASD to walk, Space to jump, G to fly (Q and E), V to see yourself, M for sound, Esc to pause. Don't hit the lanterns!");
    keys.hidden = true;
    const results = element('div', 'splat-viewer-smash-results');
    results.hidden = true;
    results.setAttribute('role', 'status');
    const lines = element('div', 'splat-viewer-smash-lines');
    const again = element('button', 'splat-viewer-smash-again', 'Play again');
    again.type = 'button';
    const note = element('p', 'splat-viewer-smash-note', touch
        ? 'Tap End smash to put the room back.'
        : 'Esc or End smash puts the room back.');
    results.append(element('p', 'splat-viewer-smash-title', "Time's up!"), lines, again, note);
    const paused = element('div', 'splat-viewer-smash-paused');
    paused.hidden = true;
    paused.append(
        element('p', 'splat-viewer-smash-title', 'Paused'),
        element('p', 'splat-viewer-smash-note', 'Click to carry on. Esc or End smash ends the game.'),
    );
    container.insertBefore(button, before ?? null);
    container.append(flash, aim, hud, pop, keys, results, paused);

    const cleanups = [];
    const listen = (target, type, handler, options) => {
        target.addEventListener(type, handler, options);
        cleanups.push(() => target.removeEventListener(type, handler, options));
    };

    // ---- state ----

    // 'off', 'on' while a round or its score shows, and 'leaving' while the room comes back.
    let mode = 'off';
    let dark = 0;
    let now = 0;
    let hauntWas = false;
    let grid;
    let shape = null;
    let body = null;
    let lastCam = [0, 0, 0];
    let scene = null;
    let roomOn = false;
    let level = 0;
    let thirdPerson = false;
    let jumpWanted = false;
    let locked = false;
    let everLocked = false;
    let lockFailed = false;
    let releasing = false;
    let lockLostAt = -Infinity;
    let isPaused = false;
    let hintUntil = 0;
    let popUntil = -Infinity;
    let caughtAt = -Infinity;
    let shownScore = null;
    let shownClock = '';
    let shownFury = -1;
    const avatar = { phase: 0, stride: 0, throwAt: -Infinity, pose: null, lastFeet: null };
    const eye = [0, 0, 0];
    const chest = [0, 0, 0];

    const room = new Float32Array([1, 1, 1, 1 / (LIGHT_REACH * length) ** 2]);
    const reach = new Float32Array([1, 1 / (LANTERN_REACH * length) ** 2, 0, 0]);
    const playerLight = new Float32Array(4);
    const pumpkinLights = PUMPKIN_LIGHTS.map(() => new Float32Array(4));
    const lanternLights = LANTERN_LIGHTS.map(() => new Float32Array(4));
    const newSlot = () => ({
        alive: false,
        id: 0,
        home: [0, 0, 0],
        last: null,
        yaw: 0,
        phase: 0,
        born: 0,
        respawnAt: -1,
        burstAt: -Infinity,
        burstCentre: [0, 0, 0],
        chasing: false,
        chaseFrom: 0,
        nextLook: 0,
        memory: { side: 1 },
        pose: { centre: [0, 0, 0], turn: [0, 0, 0, 1], radius: 0 },
    });
    const slots = Array.from({ length: PUMPKIN_COUNT }, newSlot);
    const lanterns = Array.from({ length: LANTERN_COUNT }, newSlot);
    const pieces = createPool(PIECES, blankBit);
    const particles = createPool(MAX_PARTICLES, blankBit);
    const throws = createPool(THROWS, () => ({ flight: null, kind: 'miss', index: -1, id: 0, roll: 0 }));
    const pose = blankPose();
    const flight = { position: [0, 0, 0], direction: [0, 0, 1], done: false };
    const candyTurn = [0, 0, 0, 1];
    const up = [0, 1, 0];

    // ---- the engine parts, made on the first round ----

    function buildScene() {
        const material = new pc.ShaderMaterial({
            uniqueName: 'PumpkinSmashParts',
            vertexGLSL: PART_VS,
            fragmentGLSL: PART_FS,
            attributes: {
                aPosition: pc.SEMANTIC_POSITION,
                aNormal: pc.SEMANTIC_NORMAL,
                aColor: pc.SEMANTIC_COLOR,
                aPlace: pc.SEMANTIC_ATTR12,
                aTurn: pc.SEMANTIC_ATTR13,
                aTint: pc.SEMANTIC_ATTR14,
            },
        });
        material.cull = pc.CULLFACE_NONE;
        material.update();
        const node = new pc.GraphNode('pumpkin-smash');
        const format = new pc.VertexFormat(device, [
            { semantic: pc.SEMANTIC_ATTR12, components: 4, type: pc.TYPE_FLOAT32 },
            { semantic: pc.SEMANTIC_ATTR13, components: 4, type: pc.TYPE_FLOAT32 },
            { semantic: pc.SEMANTIC_ATTR14, components: 4, type: pc.TYPE_FLOAT32 },
        ]);
        const s = shapes();
        const group = (shape, capacity) => createGroup(device, material, node, format, shape, capacity);
        const groups = {
            wedge: group(s.wedge, PUMPKIN_COUNT * WEDGES + PIECES),
            stem: group(s.stem, PUMPKIN_COUNT + PIECES),
            face: group(s.face, PUMPKIN_COUNT),
            candy: group(s.candyCorn, THROWS),
            sweet: group(s.sweet, MAX_PARTICLES),
            sparkle: group(s.sparkle, MAX_PARTICLES),
            shard: group(s.shard, MAX_PARTICLES),
            lantern: group(s.lantern, LANTERN_COUNT),
        };
        const human = humanShapes();
        for (const part of HUMAN_PARTS) {
            groups[part] = group(human[part], 2);
        }
        const instances = Object.values(groups).map((g) => g.instance);
        world.addMeshInstances(instances, true);
        return { material, groups, instances };
    }

    // The splat's centres counted into the grid of open and solid space, or null with no centres.
    function buildGrid() {
        const centres = resource?.centers ?? null;
        if (!centres || centres.length < 3) {
            return null;
        }
        const m = splat.getWorldTransform().data;
        const toViewer = {
            rows: [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]],
            offset: [m[12], m[13], m[14]],
        };
        return buildOpenGrid(centres, toViewer, boxAround(path.positions, GRID_MARGIN * length), {
            keepOpen: path.positions,
            openRadius: PATH_OPEN * length,
        });
    }

    // ---- the light ----

    function setRoom(on) {
        if (on === roomOn) {
            return;
        }
        roomOn = on;
        const chunks = template.getShaderChunks(pc.SHADERLANGUAGE_GLSL);
        if (on) {
            chunks.set('gsplatModifyVS', ROOM_GLSL);
            template.setParameter('uSmashRoom', room);
            template.setParameter('uSmashReach', reach);
            template.setParameter('uSmashPlayer', playerLight);
            PUMPKIN_LIGHTS.forEach((name, i) => template.setParameter(name, pumpkinLights[i]));
            LANTERN_LIGHTS.forEach((name, i) => template.setParameter(name, lanternLights[i]));
        } else {
            chunks.delete('gsplatModifyVS');
        }
        template.update();
    }

    function writeRoom() {
        const k = smoothstep(dark);
        const light = lightingFor(level);
        for (let c = 0; c < 3; c++) {
            room[c] = 1 + (light.keep[c] - 1) * k;
        }
        reach[0] = 1 / (light.playerReach * length) ** 2;
        playerLight[0] = chest[0];
        playerLight[1] = chest[1];
        playerLight[2] = chest[2];
        playerLight[3] = light.playerStrength * k;
        slots.forEach((slot, i) => {
            const out = pumpkinLights[i];
            const burst = flashOf(now - slot.burstAt);
            const glowing = slot.alive && now >= slot.born;
            const at = glowing ? slot.pose.centre : slot.burstCentre;
            const steady = still ? 0.95 : flicker(now, slot.phase);
            const own = glowing ? steady * Math.min(1, (now - slot.born) / POP_IN) * light.pumpkinGain : 0;
            out[0] = at[0];
            out[1] = at[1];
            out[2] = at[2];
            out[3] = (own + burst) * k;
        });
        lanterns.forEach((lantern, i) => {
            const out = lanternLights[i];
            const burst = flashOf(now - lantern.burstAt);
            const glowing = lantern.alive && now >= lantern.born;
            const at = glowing ? lantern.pose.centre : lantern.burstCentre;
            const steady = still ? 1 : 0.92 + 0.08 * Math.sin(now * 5.3 + lantern.phase * 9);
            out[0] = at[0];
            out[1] = at[1];
            out[2] = at[2];
            out[3] = ((glowing ? LANTERN_LIGHT * steady : 0) + burst) * k;
        });
        // The template material's changes reach the splat's own material on its next update.
        template.update();
    }

    // ---- you ----

    // Puts the camera at your eye, or behind you, and tells the engine.
    function setCamera() {
        eyeOf(shape, body, eye);
        chest[0] = eye[0];
        chest[1] = eye[1] - 0.25 * shape.height;
        chest[2] = eye[2];
        let camera = [...eye];
        if (thirdPerson) {
            const above = [eye[0], eye[1] + THIRD_UP * shape.height, eye[2]];
            const roomAbove = grid === null || openDepth(grid, eye, up, THIRD_UP * shape.height + grid.cell) > THIRD_UP * shape.height;
            camera = followCamera(grid, roomAbove ? above : eye, view, THIRD_BACK * shape.height, THIRD_SIDE * shape.height);
        }
        view.position = camera;
        lastCam = [...camera];
        place();
    }

    // Takes how far the viewer's controls moved the camera this frame and moves your body that
    // far, through the walls' physics. Paused, it puts the camera back.
    function moveBody(step) {
        const move = [view.position[0] - lastCam[0], view.position[1] - lastCam[1], view.position[2] - lastCam[2]];
        if (step > 0) {
            const before = [...body.feet];
            stepBody(grid, shape, body, move, step, jumpWanted);
            const moved = Math.hypot(body.feet[0] - before[0], body.feet[2] - before[2]);
            const target = Math.min(1, moved / step / (0.4 * length));
            avatar.stride += (target - avatar.stride) * Math.min(1, step * 8);
            avatar.phase += (moved / (0.9 * shape.height)) * 2 * Math.PI;
        }
        jumpWanted = false;
        setCamera();
    }

    // ---- the pumpkins and the lanterns ----

    function nearestOnPath(point) {
        let best = path.positions[0];
        let bestDistance = Infinity;
        for (const p of path.positions) {
            const d = (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2 + (p[2] - point[2]) ** 2;
            if (d < bestDistance) {
                bestDistance = d;
                best = p;
            }
        }
        return best;
    }

    function homesBut(skip) {
        const homes = [];
        for (const other of [...slots, ...lanterns]) {
            if (other !== skip && other.alive) {
                homes.push(other.home);
            }
        }
        if (skip.last) {
            homes.push(skip.last);
        }
        return homes;
    }

    // Puts a thing in slot, at a new place away from the others, its own last place and you.
    function putSlot(slot, delay, size, away) {
        const spot = placePumpkin({
            path, grid, view, others: homesBut(slot), radius: size, random,
            keepAway: { point: chest, distance: away * length },
        });
        if (spot === null) {
            slot.respawnAt = now + RETRY;
            return;
        }
        const facing = nearestOnPath(spot);
        slot.home = spot;
        slot.yaw = Math.atan2(facing[0] - spot[0], facing[2] - spot[2]);
        slot.phase = random();
        slot.born = now + delay;
        slot.alive = true;
        slot.chasing = false;
        slot.nextLook = now + delay + POP_IN;
        slot.id += 1;
        slot.respawnAt = -1;
    }
    const placePumpkinSlot = (slot, delay) => putSlot(slot, delay, radius, KEEP_AWAY);
    const placeLantern = (lantern, delay) => putSlot(lantern, delay, lanternSize * 0.5, HEAR);

    // Where a slot's pumpkin is this frame, written into slot.pose: bobbing, swaying, popping up,
    // and shrinking away while the room comes back. One that isn't chasing is never drawn filling
    // the screen; one that is comes right up to you.
    function posePumpkin(slot) {
        const age = now - slot.born;
        const t = Math.min(1, Math.max(0, age / POP_IN));
        // Pops up past its size and settles back.
        const popIn = t <= 0 ? 0 : 1 + 2.7 * (t - 1) ** 3 + 1.7 * (t - 1) ** 2;
        const bob = still ? 0 : Math.sin(2 * Math.PI * (BOB_RATE * now + slot.phase)) * BOB * radius;
        const swayAmount = slot.chasing ? 0.25 : 1;
        const sway = still ? 0 : swayAmount * (SWAY * Math.PI / 180) * Math.sin(2 * Math.PI * (SWAY_RATE * now + slot.phase * 3.1));
        const centre = slot.pose.centre;
        centre[0] = slot.home[0];
        centre[1] = slot.home[1] + bob;
        centre[2] = slot.home[2];
        quatAxis(up, slot.yaw + sway, slot.pose.turn);
        const away = Math.hypot(centre[0] - view.position[0], centre[1] - view.position[1], centre[2] - view.position[2]);
        const leaving = mode === 'leaving' ? smoothstep(dark) : 1;
        const fit = slot.chasing ? 1 : screenFit(away, radius, view.fov);
        slot.pose.radius = radius * Math.max(0, popIn) * fit * leaving;
    }

    function poseLantern(lantern) {
        const t = Math.min(1, Math.max(0, (now - lantern.born) / POP_IN));
        const bob = still ? 0 : Math.sin(2 * Math.PI * (0.3 * now + lantern.phase)) * 0.06 * lanternSize;
        lantern.pose.centre[0] = lantern.home[0];
        lantern.pose.centre[1] = lantern.home[1] + bob;
        lantern.pose.centre[2] = lantern.home[2];
        quatAxis(up, lantern.yaw + (still ? 0 : 0.25 * Math.sin(0.7 * now + lantern.phase * 5)), lantern.pose.turn);
        const leaving = mode === 'leaving' ? smoothstep(dark) : 1;
        lantern.pose.radius = lanternSize * smoothstep(t) * leaving;
    }

    // Shows a score change for a moment by the score.
    function showPop(text, good) {
        pop.textContent = text;
        pop.classList.toggle('is-good', good);
        popUntil = now + POP;
    }

    function burstPumpkin(slot, lightFlash = true) {
        slot.alive = false;
        slot.last = [...slot.home];
        slot.burstAt = lightFlash ? now : -Infinity;
        slot.burstCentre = [...slot.pose.centre];
        slot.respawnAt = now + RESPAWN_SECONDS;
        launchParts(slot.pose, parts, random, now, (at) => pieces.spawn(at));
        launchSweets(slot.pose.centre, Math.max(slot.pose.radius, radius * 0.5), random, now, (at) => particles.spawn(at));
    }

    function smash(slot) {
        if (round.smash()) {
            showPop('+1', true);
        }
        level = furyKick(level, FURY_SMASH);
        sounds.smashSound();
        burstPumpkin(slot);
    }

    function caught(slot) {
        if (round.caughtBy()) {
            showPop('−1 caught!', false);
        }
        level = furyKick(level, FURY_CAUGHT);
        caughtAt = now;
        sounds.caughtSound();
        // The edges of the view glow instead: a flash of its light right by your eyes would only
        // wash the view out.
        burstPumpkin(slot, false);
    }

    function breakLantern(lantern) {
        if (round.breakLantern()) {
            showPop('−2 lantern!', false);
        }
        level = furyKick(level, FURY_LANTERN);
        sounds.lanternSound();
        lantern.alive = false;
        lantern.last = [...lantern.home];
        lantern.burstAt = now;
        lantern.burstCentre = [...lantern.pose.centre];
        lantern.respawnAt = now + LANTERN_RESPAWN;
        launchShards(lantern.pose.centre, Math.max(lantern.pose.radius, lanternSize * 0.5), random, now, (at) => particles.spawn(at));
    }

    // ---- throwing ----

    function throwCandy(x, y, width, height) {
        if (round.state !== 'playing' || isPaused) {
            return;
        }
        const ray = screenRay(view, x, y, width, height);
        const targets = [
            ...slots.map((slot) => (slot.alive && slot.pose.radius > 0 ? { centre: slot.pose.centre, radius: slot.pose.radius } : null)),
            ...lanterns.map((lantern) => (lantern.alive && lantern.pose.radius > 0
                ? { centre: lantern.pose.centre, radius: lantern.pose.radius * LANTERN_HIT } : null)),
        ];
        const furthest = MISS_DISTANCE * length;
        const wall = grid ? openDepth(grid, ray.origin, ray.direction, furthest) : Infinity;
        const hit = throwHit(ray.origin, ray.direction, targets, Math.min(furthest, wall));
        const candy = throws.spawn(now);
        if (hit !== null) {
            const pumpkin = hit.index < slots.length;
            candy.kind = pumpkin ? 'pumpkin' : 'lantern';
            candy.index = pumpkin ? hit.index : hit.index - slots.length;
            candy.id = (pumpkin ? slots[candy.index] : lanterns[candy.index]).id;
        } else {
            candy.kind = wall < furthest ? 'wall' : 'miss';
            candy.index = -1;
            candy.id = 0;
        }
        const distance = hit !== null ? hit.distance : Math.min(furthest, wall);
        const target = ray.origin.map((value, c) => value + ray.direction[c] * distance);
        const hand = thirdPerson && avatar.pose ? rightHand(avatar.pose) : null;
        candy.flight = createThrow(hand ?? throwStart(view), target);
        candy.roll = random() * Math.PI * 2;
        avatar.throwAt = now;
        level = furyKick(level, FURY_THROW);
        sounds.throwSound();
    }

    function throwAtAim() {
        throwCandy(canvas.clientWidth / 2, canvas.clientHeight / 2, canvas.clientWidth, canvas.clientHeight);
    }

    // ---- the mouse ----

    function requestLock() {
        if (!wantLock || lockFailed || document.pointerLockElement === canvas) {
            return;
        }
        try {
            const asked = canvas.requestPointerLock();
            asked?.catch?.(() => {});
        } catch {
            lockFailed = !everLocked;
        }
    }

    function releaseLock() {
        if (document.pointerLockElement === canvas) {
            releasing = true;
            document.exitPointerLock();
        }
    }

    function pause() {
        isPaused = true;
        paused.hidden = false;
        keys.hidden = false;
        showAim();
    }

    function resume() {
        isPaused = false;
        paused.hidden = true;
        hintUntil = Math.max(hintUntil, now + 3);
        showAim();
    }

    function showAim() {
        aim.hidden = !(locked && mode === 'on' && !isPaused && round.state === 'playing');
    }

    // ---- the round ----

    function showHud() {
        if (round.score !== shownScore) {
            shownScore = round.score;
            scoreText.textContent = `🎃 ${round.score}`;
            scoreText.setAttribute('aria-label', `Score: ${round.score}`);
        }
        const text = clockText(round.timeLeft);
        if (text !== shownClock) {
            shownClock = text;
            clock.textContent = text;
            clock.classList.toggle('is-low', round.state === 'playing' && round.timeLeft <= 10);
        }
        const percent = Math.round(level * 100);
        if (percent !== shownFury) {
            shownFury = percent;
            furyFill.style.width = `${percent}%`;
            fury.setAttribute('aria-valuenow', String(percent));
        }
    }

    function showButton() {
        const on = mode === 'on';
        button.textContent = on ? '■ End smash' : '🎃 Pumpkin smash';
        button.setAttribute('aria-pressed', String(on));
        button.setAttribute('aria-label', on
            ? 'End pumpkin smash and put the room back'
            : 'Play pumpkin smash: throw candy corn at glowing pumpkins for 60 seconds');
    }

    function showMute() {
        mute.textContent = sounds.muted ? '🔇' : '🔊';
        mute.setAttribute('aria-pressed', String(sounds.muted));
        mute.setAttribute('aria-label', sounds.muted ? 'Turn the sounds on' : 'Mute the sounds');
    }

    function toggleMute() {
        sounds.setMuted(!sounds.muted);
        showMute();
    }

    function begin() {
        sounds.wake();
        stopTour();
        const fresh = mode === 'off' || mode === 'leaving';
        if (fresh) {
            hauntWas = haunt.on;
            if (haunt.on) {
                haunt.toggle(view);
            }
        }
        scene ??= buildScene();
        if (grid === undefined) {
            grid = buildGrid();
            shape = bodyFor(grid, path, length);
        }
        if (fresh || body === null) {
            thirdPerson = false;
            body = createBody(shape, view.position);
            rescueBody(grid, shape, body, path);
            avatar.stride = 0;
        }
        setRoom(true);
        mode = 'on';
        round.start();
        level = 0;
        isPaused = false;
        paused.hidden = true;
        pieces.clear();
        particles.clear();
        throws.clear();
        setCamera();
        for (const thing of [...slots, ...lanterns]) {
            thing.alive = false;
            thing.last = null;
            thing.burstAt = -Infinity;
        }
        slots.forEach((slot, i) => placePumpkinSlot(slot, i * POP_STAGGER));
        lanterns.forEach((lantern, i) => placeLantern(lantern, (PUMPKIN_COUNT + i) * POP_STAGGER));
        results.hidden = true;
        hud.hidden = false;
        keys.hidden = false;
        // The viewer's own hint would sit under the game's controls line.
        const viewerHint = container.querySelector('.splat-viewer-hint');
        if (viewerHint) {
            viewerHint.hidden = true;
        }
        hintUntil = now + HINT_SECONDS;
        popUntil = -Infinity;
        showHud();
        showButton();
        // Space jumps, so it mustn't press a button that still has focus, like End smash.
        container.focus({ preventScroll: true });
        requestLock();
        showAim();
        if (shape.gravity === null && grid === null) {
            announce('This splat has no walls to find, so you fly through it.');
        }
    }

    function finish() {
        if (mode !== 'on') {
            return;
        }
        round.end();
        mode = 'leaving';
        isPaused = false;
        releaseLock();
        throws.clear();
        hud.hidden = true;
        results.hidden = true;
        paused.hidden = true;
        keys.hidden = true;
        aim.hidden = true;
        flash.hidden = true;
        pop.textContent = '';
        showButton();
        if (hauntWas && !haunt.on) {
            haunt.toggle(view);
        }
        hauntWas = false;
        if (container.contains(document.activeElement) && document.activeElement !== button) {
            container.focus({ preventScroll: true });
        }
    }

    function timeUp() {
        releaseLock();
        lines.replaceChildren(...resultLines(round).map((line, i, all) => element('p', i === all.length - 1 ? 'splat-viewer-smash-total' : '', line)));
        results.hidden = false;
        keys.hidden = true;
        showHud();
        showAim();
        again.focus({ preventScroll: true });
    }

    function toggle() {
        if (mode === 'on') {
            finish();
        } else {
            begin();
        }
    }

    // Chasing pumpkins that come closer than two radii are pushed apart, each only as far as it
    // fits, so they don't overlap while they bunch up behind you.
    function keepApart() {
        for (let i = 0; i < slots.length; i++) {
            const a = slots[i];
            if (!a.alive || !a.chasing) {
                continue;
            }
            for (let j = i + 1; j < slots.length; j++) {
                const b = slots[j];
                if (!b.alive || !b.chasing) {
                    continue;
                }
                const d = [a.home[0] - b.home[0], a.home[1] - b.home[1], a.home[2] - b.home[2]];
                const gap = Math.hypot(d[0], d[1], d[2]);
                if (gap >= 2 * radius || gap < 1e-9) {
                    continue;
                }
                const push = (2 * radius - gap) / 2 / gap;
                for (const [thing, sign] of [[a, 1], [b, -1]]) {
                    const next = [0, 1, 2].map((c) => thing.home[c] + sign * d[c] * push);
                    if (isOpen(grid, next, radius * 0.9)) {
                        thing.home = next;
                    }
                }
            }
        }
    }

    // ---- every frame ----

    function update(dt) {
        if (mode === 'off') {
            return;
        }
        const real = Math.min(dt, 0.1);
        const step = isPaused ? 0 : real;
        now += step;
        dark = mode === 'on' ? Math.min(1, dark + real / DARK_IN) : Math.max(0, dark - real / DARK_OUT);
        level = furyAfter(level, step);
        if (round.step(step)) {
            timeUp();
        }
        if (mode === 'on') {
            showHud();
            moveBody(step);
        }
        const { groups } = scene;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const playing = mode === 'on' && round.state === 'playing' && step > 0;

        if (playing) {
            keepApart();
        }
        for (const slot of slots) {
            if (!slot.alive) {
                if (mode === 'on' && slot.respawnAt >= 0 && now >= slot.respawnAt) {
                    placePumpkinSlot(slot, 0);
                }
                continue;
            }
            if (playing && now >= slot.born) {
                if (!slot.chasing && now >= slot.nextLook) {
                    slot.nextLook = now + SEE_EVERY;
                    const spot = projectPoint(view, slot.home, width, height);
                    const onScreen = spot !== null && spot.x >= 0 && spot.x <= width && spot.y >= 0 && spot.y <= height;
                    const near = Math.hypot(slot.home[0] - chest[0], slot.home[1] - chest[1], slot.home[2] - chest[2]) < HEAR * length;
                    if (near || (onScreen && canSee(grid, view.position, slot.home, radius))) {
                        slot.chasing = true;
                        slot.chaseFrom = now + NOTICE;
                    }
                }
                if (slot.chasing && now >= slot.chaseFrom) {
                    chaseStep(grid, slot.home, chest, chaseSpeed(round.elapsed) * length, radius * 0.9, step, slot.memory);
                    slot.yaw = Math.atan2(chest[0] - slot.home[0], chest[2] - slot.home[2]);
                    const gap = Math.hypot(slot.home[0] - chest[0], slot.home[1] - chest[1], slot.home[2] - chest[2]);
                    if (gap < radius + shape.radius) {
                        posePumpkin(slot);
                        caught(slot);
                        continue;
                    }
                }
            }
            posePumpkin(slot);
            if (slot.pose.radius <= 0) {
                continue;
            }
            const glow = slot.chasing ? 0.22 : 0;
            for (const part of parts) {
                partPose(slot.pose, part, pose);
                (part.kind === 'stem' ? groups.stem : groups.wedge).add(pose.position, pose.size, pose.turn, WHITE, glow);
            }
            groups.face.add(slot.pose.centre, slot.pose.radius, slot.pose.turn, slot.chasing ? ANGRY_FACE : WHITE);
        }

        for (const lantern of lanterns) {
            if (!lantern.alive) {
                if (mode === 'on' && lantern.respawnAt >= 0 && now >= lantern.respawnAt) {
                    placeLantern(lantern, 0);
                }
                continue;
            }
            poseLantern(lantern);
            groups.lantern.add(lantern.pose.centre, lantern.pose.radius, lantern.pose.turn);
        }

        for (const candy of throws.items) {
            if (!candy.alive) {
                continue;
            }
            const age = now - candy.born;
            throwAt(candy.flight, age, flight);
            if (flight.done && (candy.kind === 'pumpkin' || candy.kind === 'lantern')) {
                const thing = candy.kind === 'pumpkin' ? slots[candy.index] : lanterns[candy.index];
                if (thing.alive && thing.id === candy.id && round.state === 'playing') {
                    if (candy.kind === 'pumpkin') {
                        smash(thing);
                    } else {
                        breakLantern(thing);
                    }
                }
                throws.retire(candy);
                continue;
            }
            if (flight.done && candy.kind === 'wall') {
                launchSweets(flight.position, radius * 0.4, random, now, (at) => particles.spawn(at), 0, 6);
                candy.kind = 'miss';
            }
            const fade = flight.done ? 1 - (age - candy.flight.seconds) / MISS_FADE : 1;
            if (fade <= 0) {
                throws.retire(candy);
                continue;
            }
            pointAlong(flight.direction, candy.roll + CANDY_SPIN * age, candyTurn);
            groups.candy.add(flight.position, CANDY_SIZE * radius * fade, candyTurn);
        }

        for (const bit of pieces.items) {
            if (!bit.alive) {
                continue;
            }
            const age = now - bit.born;
            if (age >= bit.life) {
                pieces.retire(bit);
                continue;
            }
            bitAt(bit, age, pose);
            (bit.kind === 'stem' ? groups.stem : groups.wedge).add(pose.position, pose.size, pose.turn);
        }
        for (const bit of particles.items) {
            if (!bit.alive) {
                continue;
            }
            const age = now - bit.born;
            if (age >= bit.life) {
                particles.retire(bit);
                continue;
            }
            bitAt(bit, age, pose);
            const g = bit.kind === 'sweet' ? groups.sweet : bit.kind === 'shard' ? groups.shard : groups.sparkle;
            g.add(pose.position, pose.size, pose.turn, bit.tint);
        }

        if (mode === 'on' && thirdPerson && body !== null) {
            avatar.pose = humanPose(body.feet, view.yaw, shape.height, {
                phase: avatar.phase,
                stride: avatar.stride,
                air: !body.onGround && !body.flying && shape.gravity !== null,
                flying: body.flying || shape.gravity === null,
                throwAge: now - avatar.throwAt,
            });
            for (const item of avatar.pose) {
                groups[item.part].add(item.position, shape.height, item.turn);
            }
        } else {
            avatar.pose = null;
        }

        writeRoom();
        Object.values(groups).forEach((g) => g.flush());

        flash.hidden = now - caughtAt > 0.6;
        if (!flash.hidden) {
            flash.style.opacity = String(0.85 * (1 - (now - caughtAt) / 0.6));
        }
        const popLeft = popUntil - now;
        pop.style.opacity = String(Math.max(0, Math.min(1, popLeft / 0.4)));
        if (mode === 'on' && !isPaused && now > hintUntil && round.state === 'playing') {
            keys.hidden = true;
        }

        if (mode === 'leaving' && dark <= 0) {
            mode = 'off';
            setRoom(false);
            for (const thing of [...slots, ...lanterns]) {
                thing.alive = false;
            }
            pieces.clear();
            particles.clear();
            Object.values(groups).forEach((g) => g.flush());
        }
    }

    // ---- input ----

    const presses = new Map();
    let crowded = false;
    // With the mouse locked, a click throws at the aim dot. It's taken on its way down to the
    // canvas, so the viewer's own drag handling never sees it: browsers refuse setPointerCapture
    // while the pointer is locked.
    listen(container, 'pointerdown', (event) => {
        if (!locked || event.target !== canvas) {
            return;
        }
        event.stopPropagation();
        event.preventDefault();
        if (event.button === 0 && mode === 'on' && !isPaused) {
            throwAtAim();
        }
    }, { capture: true });
    listen(canvas, 'pointerdown', (event) => {
        if (event.pointerType === 'mouse' && event.button !== 0) {
            return;
        }
        if (mode !== 'on' || locked) {
            return;
        }
        sounds.wake();
        if (isPaused || (wantLock && !lockFailed && round.state === 'playing' && event.pointerType === 'mouse')) {
            requestLock();
            return;
        }
        presses.set(event.pointerId, { x: event.clientX, y: event.clientY, time: event.timeStamp, moved: 0 });
        if (presses.size > 1) {
            crowded = true;
        }
    });
    listen(canvas, 'pointermove', (event) => {
        const press = presses.get(event.pointerId);
        if (press) {
            press.moved = Math.max(press.moved, Math.hypot(event.clientX - press.x, event.clientY - press.y));
        }
    });
    listen(canvas, 'pointerup', (event) => {
        const press = presses.get(event.pointerId);
        presses.delete(event.pointerId);
        const tap = press && !crowded && press.moved <= TAP_SLOP && event.timeStamp - press.time <= TAP_MS;
        if (presses.size === 0) {
            crowded = false;
        }
        if (tap && mode === 'on' && !locked) {
            const box = canvas.getBoundingClientRect();
            throwCandy(event.clientX - box.left, event.clientY - box.top, box.width, box.height);
        }
    });
    listen(canvas, 'pointercancel', (event) => {
        presses.delete(event.pointerId);
        if (presses.size === 0) {
            crowded = false;
        }
    });
    listen(document, 'mousemove', (event) => {
        if (locked && mode === 'on' && !isPaused) {
            mouseLook(view, event.movementX, event.movementY, MOUSE_DEGREES);
        }
    });
    listen(document, 'pointerlockchange', () => {
        const nowLocked = document.pointerLockElement === canvas;
        if (nowLocked) {
            locked = true;
            everLocked = true;
            if (isPaused) {
                resume();
            }
        } else if (locked) {
            locked = false;
            lockLostAt = performance.now();
            if (!releasing && mode === 'on' && round.state === 'playing') {
                pause();
            }
            releasing = false;
        }
        showAim();
    });
    listen(document, 'pointerlockerror', () => {
        // Before the mouse was ever locked this browser won't lock it, so the game keeps a free
        // mouse. After, it's the short wait browsers ask for after Esc; the next click tries again.
        if (!everLocked) {
            lockFailed = true;
            if (isPaused) {
                resume();
            }
        }
    });
    listen(paused, 'click', requestLock);
    listen(container, 'keydown', (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey) {
            return;
        }
        const key = event.key.toLowerCase();
        if (key === 'p' && !event.repeat) {
            event.preventDefault();
            toggle();
            return;
        }
        if (mode !== 'on') {
            return;
        }
        if (event.key === 'Escape') {
            // The Esc that frees a locked mouse pauses; the next one ends the game.
            if (locked || document.pointerLockElement === canvas || performance.now() - lockLostAt < UNLOCK_ESC_MS) {
                return;
            }
            event.preventDefault();
            finish();
        } else if (event.code === 'Space') {
            if (!(event.target instanceof HTMLButtonElement)) {
                event.preventDefault();
                jumpWanted = !event.repeat;
            }
        } else if (key === 'g' && !event.repeat) {
            if (shape.gravity === null) {
                announce('No floor was found in this splat, so you can only fly.');
            } else {
                body.flying = !body.flying;
                body.vy = 0;
                announce(body.flying ? 'Flying: Q and E go down and up. G to walk again.' : 'Walking, with gravity. G to fly.');
            }
        } else if (key === 'v' && !event.repeat) {
            thirdPerson = !thirdPerson;
            announce(thirdPerson ? 'Seeing yourself from behind. V to look through your eyes.' : 'Looking through your eyes. V to see yourself.');
        } else if (key === 'm' && !event.repeat) {
            toggleMute();
        }
    });
    listen(button, 'click', toggle);
    listen(again, 'click', begin);
    listen(mute, 'click', () => {
        toggleMute();
        container.focus({ preventScroll: true });
    });
    app.on('update', update);
    cleanups.push(() => app.off('update', update));

    showButton();
    showMute();

    return {
        get active() {
            return mode === 'on';
        },
        destroy() {
            cleanups.forEach((cleanup) => cleanup());
            releaseLock();
            if (roomOn) {
                setRoom(false);
            }
            if (scene !== null) {
                world.removeMeshInstances(scene.instances, true);
                Object.values(scene.groups).forEach((g) => g.destroy());
                scene.material.destroy();
                scene = null;
            }
            sounds.destroy();
            for (const node of [button, hud, pop, aim, flash, keys, results, paused]) {
                node.remove();
            }
        },
    };
}
