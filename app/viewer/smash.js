// Pumpkin smash, the viewer's Halloween game: glowing pumpkins float in the filmed place, and a
// tap or a click throws a glowing candy corn where it lands. A round lasts 60 seconds; smashing a
// pumpkin scores 1 and bursts it into its wedges, wrapped sweets and sparkles.
//
// createPumpkinSmash() adds the game's button to the viewer and runs it. The game's rules and
// motion are in smash-play.js, where pumpkins go in smash-place.js, their shapes in
// smash-meshes.js and the sounds in smash-sound.js. This file joins them to the engine and the page.
//
// - The dark room: the splat itself is dimmed through the engine's gsplatModifyVS chunk on the
//   gsplat template material (app.scene.gsplat.material). Its modifySplatColor(center, color) gets
//   each splat's centre in the world and its colour; ours darkens every splat and adds warm light
//   from each pumpkin, falling off with distance and flickering, so the pumpkins light the walls
//   around them. Haunted mode's canvas filter would dim the pumpkins too, so it's off while a
//   round runs.
// - The pumpkins and everything that flies are real meshes in the world layer, drawn before the
//   splat. The splat is drawn depth-tested over them, so a wall in front of a pumpkin hides it.
//   Every shape is one instanced draw, and the pieces, sweets and sparkles come from pools made
//   once, at most MAX_PARTICLES sweets and sparkles at a time.

import * as pc from './vendor/playcanvas.min.mjs';
import { GRID_MARGIN, boxAround, buildOpenGrid, placePumpkin, screenFit } from './smash-place.js';
import { WEDGES, pumpkinParts, shapes } from './smash-meshes.js';
import {
    CANDY_SIZE, MAX_PARTICLES, MISS_DISTANCE, PUMPKIN_COUNT, PUMPKIN_RADIUS, RESPAWN_SECONDS,
    blankBit, blankPose, bitAt, clockText, createPool, createRound, createThrow, flashOf, flicker,
    launchParts, launchSweets, partPose, pointAlong, quatAxis, screenRay, smoothstep, throwAt,
    throwHit, throwStart,
} from './smash-play.js';
import { createSounds } from './smash-sound.js';

// The dark room: how much of each colour a splat keeps, and the pumpkins' light, as warm orange
// over LIGHT_REACH scene lengths. The room darkens over DARK_IN seconds and comes back over DARK_OUT.
const DARK_ROOM = [0.1, 0.1, 0.15];
const LIGHT_REACH = 0.22;
const LIGHT_GAIN = 1.2;
const DARK_IN = 1.2;
const DARK_OUT = 0.8;

// The pumpkins bob by BOB radii BOB_RATE times a second, sway SWAY degrees either side of facing
// the filmed path, and pop up over POP_IN seconds, one after another at the start.
const BOB = 0.12;
const BOB_RATE = 0.4;
const SWAY = 40;
const SWAY_RATE = 0.11;
const POP_IN = 0.45;
const POP_STAGGER = 0.12;
// A miss fades over MISS_FADE seconds once it's flown its way. A placement that finds no spot is
// tried again after RETRY seconds.
const MISS_FADE = 0.15;
const RETRY = 0.25;
const CANDY_SPIN = 14;
// A tap or a click moves at most TAP_SLOP pixels and lasts at most TAP_MS milliseconds.
const TAP_SLOP = 10;
const TAP_MS = 600;
// The pools: pieces for 8 bursts at a time, and 8 candy corns in the air.
const PIECES = 8 * (WEDGES + 1);
const THROWS = 8;
const STYLE_ID = 'splat-viewer-smash-style';

const LIGHT_UNIFORMS = Array.from({ length: PUMPKIN_COUNT }, (_, i) => `uSmashLight${i}`);
const ROOM_GLSL = `
uniform vec4 uSmashRoom;
${LIGHT_UNIFORMS.map((name) => `uniform vec4 ${name};`).join('\n')}
const vec3 SMASH_GLOW = vec3(1.0, 0.55, 0.2);

// One pumpkin's light at a splat's centre: xyz is where it is, w how bright, and uSmashRoom.a is
// one over its reach squared.
float smashLight(vec4 light, vec3 p) {
    vec3 d = p - light.xyz;
    float x = clamp(1.0 - dot(d, d) * uSmashRoom.a, 0.0, 1.0);
    return light.w * x * x;
}

void modifySplatCenter(inout vec3 center) {
}

void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
}

void modifySplatColor(vec3 center, inout vec4 color) {
    float glow = ${LIGHT_UNIFORMS.map((name) => `smashLight(${name}, center)`).join('\n        + ')};
    color.rgb *= uSmashRoom.rgb + SMASH_GLOW * glow;
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
    float shade = 0.3 + 0.55 * facing + 0.15 * max(normal.y, 0.0);
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
//   path                    tour.js's path of the filmed walk
//   splat, resource         the splat's entity and its loaded resource, whose centres show the walls
//   touch                   whether this is a touch screen, for the wording
//   announce(text)          shows a short message
//   stopTour()              pauses the tour if it's playing
//   haunt                   haunted mode, { on, toggle(view) }
//   before                  the element the game's button goes in front of
//
// Returns { active, destroy() }. active is true while a round or its score is on screen.
export function createPumpkinSmash({ app, container, canvas, view, path, splat, resource, touch, announce, stopTour, haunt, before }) {
    addStyles();
    const device = app.graphicsDevice;
    const world = app.scene.layers.getLayerById(pc.LAYERID_WORLD);
    const template = app.scene.gsplat.material;
    const still = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    const sounds = createSounds();
    const round = createRound();
    const parts = pumpkinParts();
    const random = Math.random;
    const radius = PUMPKIN_RADIUS * view.sceneLength;

    // ---- the page ----

    const button = element('button', 'splat-viewer-smash');
    button.type = 'button';
    const hud = element('div', 'splat-viewer-smash-hud');
    hud.hidden = true;
    const scoreText = element('span', 'splat-viewer-smash-score');
    const clock = element('span', 'splat-viewer-smash-clock');
    clock.setAttribute('role', 'timer');
    const mute = element('button', 'splat-viewer-smash-mute');
    mute.type = 'button';
    hud.append(scoreText, clock, mute);
    const results = element('div', 'splat-viewer-smash-results');
    results.hidden = true;
    results.setAttribute('role', 'status');
    const total = element('p', 'splat-viewer-smash-total');
    const again = element('button', 'splat-viewer-smash-again', 'Play again');
    again.type = 'button';
    const note = element('p', 'splat-viewer-smash-note', touch
        ? 'Tap End smash to put the room back.'
        : 'Esc or End smash puts the room back.');
    results.append(element('p', 'splat-viewer-smash-title', "Time's up!"), total, again, note);
    container.insertBefore(button, before ?? null);
    container.append(hud, results);

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
    let scene = null;
    let roomOn = false;
    let shownScore = -1;
    let shownClock = '';

    const room = new Float32Array([1, 1, 1, 1 / (LIGHT_REACH * view.sceneLength) ** 2]);
    const lights = LIGHT_UNIFORMS.map(() => new Float32Array(4));
    const slots = Array.from({ length: PUMPKIN_COUNT }, () => ({
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
        pose: { centre: [0, 0, 0], turn: [0, 0, 0, 1], radius: 0 },
    }));
    const pieces = createPool(PIECES, blankBit);
    const particles = createPool(MAX_PARTICLES, blankBit);
    const throws = createPool(THROWS, () => ({ flight: null, slot: -1, id: 0, roll: 0 }));
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
        const groups = {
            wedge: createGroup(device, material, node, format, s.wedge, PUMPKIN_COUNT * WEDGES + PIECES),
            stem: createGroup(device, material, node, format, s.stem, PUMPKIN_COUNT + PIECES),
            face: createGroup(device, material, node, format, s.face, PUMPKIN_COUNT),
            candy: createGroup(device, material, node, format, s.candyCorn, THROWS),
            sweet: createGroup(device, material, node, format, s.sweet, MAX_PARTICLES),
            sparkle: createGroup(device, material, node, format, s.sparkle, MAX_PARTICLES),
        };
        const instances = Object.values(groups).map((group) => group.instance);
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
        return buildOpenGrid(centres, toViewer, boxAround(path.positions, GRID_MARGIN * view.sceneLength));
    }

    // ---- the dark room ----

    function setRoom(on) {
        if (on === roomOn) {
            return;
        }
        roomOn = on;
        const chunks = template.getShaderChunks(pc.SHADERLANGUAGE_GLSL);
        if (on) {
            chunks.set('gsplatModifyVS', ROOM_GLSL);
            template.setParameter('uSmashRoom', room);
            LIGHT_UNIFORMS.forEach((name, i) => template.setParameter(name, lights[i]));
        } else {
            chunks.delete('gsplatModifyVS');
        }
        template.update();
    }

    function writeRoom() {
        const k = smoothstep(dark);
        for (let c = 0; c < 3; c++) {
            room[c] = 1 + (DARK_ROOM[c] - 1) * k;
        }
        slots.forEach((slot, i) => {
            const light = lights[i];
            const flash = flashOf(now - slot.burstAt);
            const glowing = slot.alive && now >= slot.born;
            const at = glowing ? slot.pose.centre : slot.burstCentre;
            const steady = still ? 0.95 : flicker(now, slot.phase);
            const own = glowing ? steady * Math.min(1, (now - slot.born) / POP_IN) : 0;
            light[0] = at[0];
            light[1] = at[1];
            light[2] = at[2];
            light[3] = (own + flash) * k * LIGHT_GAIN;
        });
        // The template material's changes reach the splat's own material on its next update.
        template.update();
    }

    // ---- the pumpkins ----

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

    // Puts a pumpkin in slot, at a new place away from the others and from its own last place.
    function place(slot, delay) {
        const others = [];
        for (const other of slots) {
            if (other !== slot && other.alive) {
                others.push(other.home);
            }
        }
        if (slot.last) {
            others.push(slot.last);
        }
        const spot = placePumpkin({ path, grid, view, others, radius, random });
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
        slot.id += 1;
        slot.respawnAt = -1;
    }

    // Where a slot's pumpkin is this frame, written into slot.pose: bobbing, swaying, popping up,
    // never filling the screen, and shrinking away while the room comes back.
    function posePumpkin(slot) {
        const age = now - slot.born;
        const t = Math.min(1, Math.max(0, age / POP_IN));
        // Pops up past its size and settles back.
        const pop = t <= 0 ? 0 : 1 + 2.7 * (t - 1) ** 3 + 1.7 * (t - 1) ** 2;
        const bob = still ? 0 : Math.sin(2 * Math.PI * (BOB_RATE * now + slot.phase)) * BOB * radius;
        const sway = still ? 0 : (SWAY * Math.PI / 180) * Math.sin(2 * Math.PI * (SWAY_RATE * now + slot.phase * 3.1));
        const centre = slot.pose.centre;
        centre[0] = slot.home[0];
        centre[1] = slot.home[1] + bob;
        centre[2] = slot.home[2];
        quatAxis(up, slot.yaw + sway, slot.pose.turn);
        const away = Math.hypot(centre[0] - view.position[0], centre[1] - view.position[1], centre[2] - view.position[2]);
        const leaving = mode === 'leaving' ? smoothstep(dark) : 1;
        slot.pose.radius = radius * Math.max(0, pop) * screenFit(away, radius, view.fov) * leaving;
    }

    function smash(slot) {
        round.smash();
        sounds.smashSound();
        slot.alive = false;
        slot.last = [...slot.home];
        slot.burstAt = now;
        slot.burstCentre = [...slot.pose.centre];
        slot.respawnAt = now + RESPAWN_SECONDS;
        launchParts(slot.pose, parts, random, now, (at) => pieces.spawn(at));
        launchSweets(slot.pose.centre, slot.pose.radius, random, now, (at) => particles.spawn(at));
    }

    // ---- throwing ----

    function throwCandy(x, y, width, height) {
        if (round.state !== 'playing') {
            return;
        }
        const ray = screenRay(view, x, y, width, height);
        const targets = slots.map((slot) => (slot.alive && slot.pose.radius > 0
            ? { centre: slot.pose.centre, radius: slot.pose.radius }
            : null));
        const hit = throwHit(ray.origin, ray.direction, targets);
        const reach = hit ? hit.distance : MISS_DISTANCE * view.sceneLength;
        const target = ray.origin.map((value, c) => value + ray.direction[c] * reach);
        const candy = throws.spawn(now);
        candy.flight = createThrow(throwStart(view), target);
        candy.slot = hit ? hit.index : -1;
        candy.id = hit ? slots[hit.index].id : 0;
        candy.roll = random() * Math.PI * 2;
        sounds.throwSound();
    }

    // ---- the round ----

    function showHud() {
        if (round.score !== shownScore) {
            shownScore = round.score;
            scoreText.textContent = `🎃 ${round.score}`;
            scoreText.setAttribute('aria-label', `Pumpkins smashed: ${round.score}`);
        }
        const text = clockText(round.timeLeft);
        if (text !== shownClock) {
            shownClock = text;
            clock.textContent = text;
            clock.classList.toggle('is-low', round.state === 'playing' && round.timeLeft <= 10);
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

    function begin() {
        sounds.wake();
        stopTour();
        if (mode === 'off' || mode === 'leaving') {
            hauntWas = haunt.on;
            if (haunt.on) {
                haunt.toggle(view);
            }
        }
        scene ??= buildScene();
        if (grid === undefined) {
            grid = buildGrid();
        }
        setRoom(true);
        mode = 'on';
        round.start();
        pieces.clear();
        particles.clear();
        throws.clear();
        for (const slot of slots) {
            slot.alive = false;
            slot.last = null;
            slot.burstAt = -Infinity;
        }
        slots.forEach((slot, i) => place(slot, i * POP_STAGGER));
        results.hidden = true;
        hud.hidden = false;
        showHud();
        showButton();
        announce(touch ? 'Tap a pumpkin to throw candy corn at it.' : 'Click a pumpkin to throw candy corn at it. Esc stops.');
    }

    function finish() {
        if (mode !== 'on') {
            return;
        }
        round.end();
        mode = 'leaving';
        throws.clear();
        hud.hidden = true;
        results.hidden = true;
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
        total.textContent = `You smashed ${round.score} pumpkin${round.score === 1 ? '' : 's'} with candy corn.`;
        results.hidden = false;
        showHud();
        again.focus({ preventScroll: true });
    }

    function toggle() {
        if (mode === 'on') {
            finish();
        } else {
            begin();
        }
    }

    // ---- every frame ----

    function update(dt) {
        if (mode === 'off') {
            return;
        }
        const step = Math.min(dt, 0.1);
        now += step;
        dark = mode === 'on' ? Math.min(1, dark + step / DARK_IN) : Math.max(0, dark - step / DARK_OUT);
        if (round.step(step)) {
            timeUp();
        }
        if (mode === 'on') {
            showHud();
        }
        const { groups } = scene;

        for (const slot of slots) {
            if (!slot.alive) {
                if (mode === 'on' && slot.respawnAt >= 0 && now >= slot.respawnAt) {
                    place(slot, 0);
                }
                continue;
            }
            posePumpkin(slot);
            if (slot.pose.radius <= 0) {
                continue;
            }
            for (const part of parts) {
                partPose(slot.pose, part, pose);
                (part.kind === 'stem' ? groups.stem : groups.wedge).add(pose.position, pose.size, pose.turn);
            }
            groups.face.add(slot.pose.centre, slot.pose.radius, slot.pose.turn);
        }

        for (const candy of throws.items) {
            if (!candy.alive) {
                continue;
            }
            const age = now - candy.born;
            throwAt(candy.flight, age, flight);
            if (flight.done && candy.slot >= 0) {
                const slot = slots[candy.slot];
                if (slot.alive && slot.id === candy.id && round.state === 'playing') {
                    smash(slot);
                }
                throws.retire(candy);
                continue;
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
            (bit.kind === 'sweet' ? groups.sweet : groups.sparkle).add(pose.position, pose.size, pose.turn, bit.tint);
        }

        writeRoom();
        Object.values(groups).forEach((group) => group.flush());

        if (mode === 'leaving' && dark <= 0) {
            mode = 'off';
            setRoom(false);
            for (const slot of slots) {
                slot.alive = false;
            }
            pieces.clear();
            particles.clear();
            Object.values(groups).forEach((group) => group.flush());
        }
    }

    // ---- input ----

    const presses = new Map();
    let crowded = false;
    listen(canvas, 'pointerdown', (event) => {
        if (event.pointerType === 'mouse' && event.button !== 0) {
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
        if (tap && mode === 'on') {
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
    listen(container, 'keydown', (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey) {
            return;
        }
        if (event.key.toLowerCase() === 'p' && !event.repeat) {
            event.preventDefault();
            toggle();
        } else if (event.key === 'Escape' && mode === 'on') {
            event.preventDefault();
            finish();
        }
    });
    listen(button, 'click', toggle);
    listen(again, 'click', begin);
    listen(mute, 'click', () => {
        sounds.setMuted(!sounds.muted);
        showMute();
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
            if (roomOn) {
                setRoom(false);
            }
            if (scene !== null) {
                world.removeMeshInstances(scene.instances, true);
                Object.values(scene.groups).forEach((group) => group.destroy());
                scene.material.destroy();
                scene = null;
            }
            sounds.destroy();
            button.remove();
            hud.remove();
            results.remove();
        },
    };
}
