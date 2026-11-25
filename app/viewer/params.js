// Where the viewer gets its files: the splat (a job's out/clean.sog, or out/train.ply when the
// job has no clean copy) and the job's out/cameras.json. Pure functions with no engine and no
// DOM, so the tests run them in Node.

// The file types the engine reads, by the last extension of the URL's path. Our clean step
// writes .sog and the train step .ply.
const SPLAT_KINDS = { sog: 'sog', ply: 'ply' };

// What a job's out/ folder holds, in the order the viewer tries them.
export const JOB_SPLATS = ['clean.sog', 'train.ply'];
export const JOB_CAMERAS = 'cameras.json';

export class ViewerError extends Error {}

// The splat type of a URL, 'sog' or 'ply', or null. Only the path counts, so a signed link's
// query string doesn't matter, the same way the engine picks its parser.
export function splatKind(url) {
    const path = new URL(url, 'https://viewer.invalid/').pathname;
    const name = path.slice(path.lastIndexOf('/') + 1).toLowerCase();
    const dot = name.lastIndexOf('.');
    return dot < 0 ? null : SPLAT_KINDS[name.slice(dot + 1)] ?? null;
}

function webUrl(value, base, what) {
    let url;
    try {
        url = new URL(value, base);
    } catch {
        throw new ViewerError(`The ${what} address isn't a URL: ${value}`);
    }
    if (url.protocol !== 'https:' && url.protocol !== 'http:') {
        throw new ViewerError(`The ${what} address must start with https:// or http://, not ${url.protocol}`);
    }
    return url;
}

// The files to load from a splat address and an optional cameras address, both resolved against
// the page's own URL.
//
// - A file's URL must end in .sog or .ply.
// - A folder's URL, ending in /, is a job's out/ folder: the viewer tries out/clean.sog, then
//   out/train.ply.
// - With no cameras address, the cameras are out/cameras.json beside the splat. A signed link
//   (one with a query string) signs only its own file, so then there's no cameras file to guess,
//   and the viewer frames the splat on its own.
//
// Returns { splatUrls: [string, ...], camerasUrl: string | null }.
export function viewerSources(splat, cameras, base) {
    if (typeof splat !== 'string' || splat.trim() === '') {
        throw new ViewerError("There's no splat to show. Give the address of a job's out/clean.sog or out/train.ply.");
    }
    const url = webUrl(splat.trim(), base, 'splat');
    let splatUrls;
    let folder = null;
    if (url.pathname.endsWith('/')) {
        folder = new URL(url.pathname, url);
        splatUrls = JOB_SPLATS.map((name) => new URL(name, folder).href);
    } else if (splatKind(url.href) === null) {
        throw new ViewerError(`The splat's address must end in .sog or .ply: ${url.pathname}`);
    } else {
        splatUrls = [url.href];
    }
    let camerasUrl = null;
    if (typeof cameras === 'string' && cameras.trim() !== '') {
        camerasUrl = webUrl(cameras.trim(), base, 'cameras').href;
    } else if (folder !== null) {
        camerasUrl = new URL(JOB_CAMERAS, folder).href;
    } else if (url.search === '') {
        camerasUrl = new URL(JOB_CAMERAS, url).href;
    }
    return { splatUrls, camerasUrl };
}

// The standalone page's parameters: ?url=<splat or out/ folder>&cameras=<cameras.json>.
export function readViewerParams(search, base) {
    const params = new URLSearchParams(search);
    return viewerSources(params.get('url'), params.get('cameras'), base);
}
