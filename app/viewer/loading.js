// The loading screen's numbers, from the engine's progress events: bytes received so far and the
// file's Content-Length, which is 0 when the host doesn't send one. Pure, so the tests run it in
// Node.

const MB = 1e6;

const megabytes = (bytes) => (bytes / MB).toFixed(1);

// Returns { fraction, percent, detail }: fraction from 0 to 1, or null when the size isn't known,
// the whole percent as text (or '' with no size), and the megabytes as text.
export function loadingProgress(received, total) {
    const got = Math.max(0, Number(received) || 0);
    const size = Math.max(0, Number(total) || 0);
    if (size === 0) {
        return { fraction: null, percent: '', detail: `${megabytes(got)} MB` };
    }
    const fraction = Math.min(1, got / size);
    return {
        fraction,
        percent: `${Math.floor(fraction * 100)}%`,
        detail: `${megabytes(Math.min(got, size))} of ${megabytes(size)} MB`,
    };
}
