// Starts the viewer inside another page, like the website's Result_Page: every element with a
// data-splat-url attribute becomes a viewer. data-cameras-url is optional. See README.md.

import { ViewerError, viewerSources } from './params.js';
import { mountViewer } from './viewer.js';

for (const container of document.querySelectorAll('[data-splat-url]')) {
    try {
        const sources = viewerSources(container.dataset.splatUrl, container.dataset.camerasUrl, location.href);
        mountViewer(container, sources);
    } catch (error) {
        const message = document.createElement('p');
        message.className = 'splat-viewer-status';
        message.textContent = error instanceof ViewerError ? error.message : String(error);
        container.replaceChildren(message);
    }
}
