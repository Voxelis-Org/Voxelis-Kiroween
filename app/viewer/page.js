// The standalone viewer page, index.html: ?url=<a job's out/clean.sog, out/train.ply or out/
// folder>, and optionally &cameras=<its out/cameras.json>. For testing on a laptop.

import { ViewerError, readViewerParams } from './params.js';

const container = document.getElementById('viewer');

function fail(text) {
    const message = document.createElement('p');
    message.className = 'splat-viewer-status';
    message.textContent = text;
    container.replaceChildren(message);
}

try {
    const sources = readViewerParams(location.search, location.href);
    import('./viewer.js')
        .then(({ mountViewer }) => {
            mountViewer(container, sources);
            // The page is only the viewer, so its keys (Space, H, WASD) work without a click first.
            container.focus({ preventScroll: true });
        })
        .catch((error) => {
            console.error(error);
            fail("The viewer's engine didn't load. Run uv run python scripts/fetch_viewer_engine.py, then reload.");
        });
} catch (error) {
    fail(error instanceof ViewerError ? error.message : String(error));
}
