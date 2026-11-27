// Pumpkin smash's two sounds, made with Web Audio in code and no audio files: a soft whoosh when
// a candy corn is thrown, and a squelchy splat with a few sparkly pings when a pumpkin bursts.
//
// The audio context starts on the first sound, which always comes from a tap, a click or a key, so
// browsers that only allow sound after the visitor does something let it play. Muting turns the
// main volume down to nothing; it isn't saved.

const VOLUME = 0.6;
const NOISE_SECONDS = 0.5;
// The pings' notes, in hertz: a G, a C and an E two octaves up, for a cheerful sparkle.
const PINGS = [1568, 2093, 2637];

export function createSounds() {
    const Context = globalThis.AudioContext ?? globalThis.webkitAudioContext;
    let context = null;
    let main = null;
    let noise = null;
    let muted = false;

    const ready = () => {
        if (!Context) {
            return false;
        }
        if (context === null) {
            context = new Context();
            main = context.createGain();
            main.gain.value = muted ? 0 : VOLUME;
            main.connect(context.destination);
            // Half a second of white noise, made once and played through filters.
            noise = context.createBuffer(1, Math.ceil(context.sampleRate * NOISE_SECONDS), context.sampleRate);
            const samples = noise.getChannelData(0);
            for (let i = 0; i < samples.length; i++) {
                samples[i] = Math.random() * 2 - 1;
            }
        }
        if (context.state === 'suspended') {
            context.resume().catch(() => {});
        }
        return true;
    };

    // A gain that rises to peak at start and dies away over seconds, into the main volume.
    const envelope = (start, peak, seconds) => {
        const gain = context.createGain();
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(peak, start + 0.008);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + seconds);
        gain.connect(main);
        return gain;
    };

    const noiseBurst = (start, seconds, type, from, to, q, peak) => {
        const source = context.createBufferSource();
        source.buffer = noise;
        const filter = context.createBiquadFilter();
        filter.type = type;
        filter.Q.value = q;
        filter.frequency.setValueAtTime(from, start);
        filter.frequency.exponentialRampToValueAtTime(to, start + seconds);
        source.connect(filter).connect(envelope(start, peak, seconds));
        source.start(start);
        source.stop(start + seconds + 0.02);
    };

    const tone = (start, seconds, type, from, to, peak) => {
        const oscillator = context.createOscillator();
        oscillator.type = type;
        oscillator.frequency.setValueAtTime(from, start);
        oscillator.frequency.exponentialRampToValueAtTime(to, start + seconds);
        oscillator.connect(envelope(start, peak, seconds));
        oscillator.start(start);
        oscillator.stop(start + seconds + 0.02);
    };

    return {
        get muted() {
            return muted;
        },
        setMuted(value) {
            muted = value;
            if (main !== null) {
                main.gain.setTargetAtTime(muted ? 0 : VOLUME, context.currentTime, 0.02);
            }
        },
        // Starts the audio context from a tap, a click or a key, before the first sound.
        wake() {
            ready();
        },
        // The throw: a whoosh of noise sweeping down, with a soft falling whistle.
        throwSound() {
            if (muted || !ready()) {
                return;
            }
            const now = context.currentTime;
            noiseBurst(now, 0.2, 'bandpass', 2600, 700, 1.4, 0.35);
            tone(now, 0.16, 'sine', 900, 320, 0.08);
        },
        // The smash: a low thump, a wet crunch of noise, and three quick pings.
        smashSound() {
            if (muted || !ready()) {
                return;
            }
            const now = context.currentTime;
            tone(now, 0.2, 'sine', 170, 55, 0.7);
            noiseBurst(now, 0.28, 'lowpass', 2200, 260, 0.9, 0.55);
            PINGS.forEach((note, i) => tone(now + 0.05 + i * 0.055, 0.25, 'triangle', note, note * 0.98, 0.06));
        },
        destroy() {
            context?.close().catch(() => {});
            context = null;
        },
    };
}
