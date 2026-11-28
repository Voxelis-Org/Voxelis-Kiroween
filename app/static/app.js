// The Page_Script: the Start_Precheck, the part upload straight to storage, and the status
// refresh on a Result_Page that's still starting or running, plus the small things that make the
// pages nicer to use: the phone menu, the video drop zone, the upload percentage and the share
// link's Copy button. Plain JavaScript, no build step, no inline handlers or styles. Every request
// to the website carries the CSRF_Token in X-CSRF-Token.
'use strict';

(function () {
  const main = document.querySelector('main');
  const tokenTag = document.querySelector('meta[name="csrf-token"]');
  const csrf = tokenTag ? tokenTag.content : '';
  // Pages style the phone menu only once this script runs, so without it the links stay visible.
  document.documentElement.classList.add('js');

  class Refusal extends Error {}

  async function api(method, path, body) {
    const options = { method, headers: { 'X-CSRF-Token': csrf }, credentials: 'same-origin' };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    if (response.status === 401) {
      location.assign('/signin');
      throw new Refusal('You were signed out. Sign in again.');
    }
    let data = {};
    try {
      data = await response.json();
    } catch (error) {
      data = {};
    }
    if (!response.ok) {
      throw new Refusal(data.message || 'Something went wrong. Try again in a minute.');
    }
    return data;
  }

  // One part's PUT with XMLHttpRequest, which reports progress within the part. It sets no
  // header of its own, so the preflight asks for none. Gives the status and the ETag.
  function putPart(url, blob, onProgress) {
    return new Promise((resolve) => {
      const request = new XMLHttpRequest();
      request.open('PUT', url);
      request.upload.addEventListener('progress', (event) => onProgress(event.loaded));
      request.addEventListener('load', () => resolve({
        status: request.status, etag: request.getResponseHeader('ETag'),
      }));
      request.addEventListener('error', () => resolve({ status: 0, etag: null }));
      request.send(blob);
    });
  }

  function showMessage(text, kind) {
    const box = document.getElementById('message');
    if (box) {
      box.textContent = text;
      box.className = kind || '';
    }
  }

  // The bar and the percentage beside it, from a fraction between 0 and 1.
  function setProgress(progress, fraction) {
    const value = Math.max(0, Math.min(1, fraction || 0));
    progress.value = value;
    const percent = document.getElementById('progress-percent');
    if (percent) percent.textContent = `${Math.floor(value * 100)}%`;
  }

  async function uploadParts(file, upload, retries, progress) {
    const parts = [];
    let sent = 0;
    for (let n = 1; n <= upload.parts; n += 1) {
      const blob = file.slice((n - 1) * upload.part_bytes, Math.min(n * upload.part_bytes, file.size));
      let result = null;
      for (let attempt = 0; attempt <= retries; attempt += 1) {
        const link = await api('POST', `/api/uploads/${upload.upload}/parts/${n}`, {});
        result = await putPart(link.url, blob, (loaded) => { setProgress(progress, (sent + loaded) / file.size); });
        if (result.status !== 403) break;
      }
      if (!result || result.status < 200 || result.status >= 300 || !result.etag) {
        await api('POST', `/api/uploads/${upload.upload}/abort`, {}).catch(() => {});
        throw new Refusal('The upload stopped. Check your connection and start again.');
      }
      sent += blob.size;
      setProgress(progress, sent / file.size);
      parts.push([n, result.etag]);
    }
    return parts;
  }

  async function start(form) {
    const button = document.getElementById('start');
    const progress = document.getElementById('progress');
    const file = form.querySelector('#video').files[0];
    const chosen = form.querySelector('input[name="preset"]:checked');
    if (!file || !chosen) {
      showMessage('Pick a video and a preset first.', 'error');
      return;
    }
    button.disabled = true;
    try {
      showMessage('Checking that a job can start…');
      await api('POST', '/api/precheck', { preset: chosen.value });
      const upload = await api('POST', '/api/uploads', { name: file.name, size: file.size });
      setProgress(progress, 0);
      const box = document.getElementById('upload-progress');
      if (box) box.hidden = false;
      showMessage('Uploading…');
      const retries = Number(main.dataset.partRetries || '0');
      const parts = await uploadParts(file, upload, retries, progress);
      await api('POST', `/api/uploads/${upload.upload}/complete`, { parts });
      showMessage('Starting the job…');
      const job = await api('POST', '/api/jobs', { upload: upload.upload, preset: chosen.value });
      location.assign(`/jobs/${job.job}`);
    } catch (error) {
      showMessage(error instanceof Refusal ? error.message : 'Something went wrong. Try again.', 'error');
      button.disabled = false;
    }
  }

  function refresh() {
    const jobId = main && main.dataset.jobId;
    const seconds = Number(main && main.dataset.pollSeconds);
    if (!jobId || !seconds) return;
    const timer = setInterval(async () => {
      try {
        const status = await api('GET', `/api/jobs/${jobId}/status`);
        const text = document.getElementById('status');
        if (text) {
          text.textContent = status.text;
          text.className = `pill status-${status.status}`;
        }
        if (status.final) {
          clearInterval(timer);
          location.reload();
        }
      } catch (error) {
        clearInterval(timer);
      }
    }, seconds * 1000);
  }

  function megabytes(bytes) {
    return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
  }

  // The drop zone is the file input itself, stretched over the box, so a dropped file lands in it
  // natively. This only shows which file was picked and lights the box while a file is over it.
  function dropZone() {
    const zone = document.getElementById('dropzone');
    const input = document.getElementById('video');
    const name = document.getElementById('video-name');
    if (!zone || !input) return;
    const show = () => {
      const file = input.files[0];
      zone.classList.toggle('has-file', Boolean(file));
      if (name) name.textContent = file ? `${file.name}, ${megabytes(file.size)}` : '';
    };
    input.addEventListener('change', show);
    input.addEventListener('dragenter', () => zone.classList.add('is-over'));
    for (const type of ['dragleave', 'drop']) {
      input.addEventListener(type, () => zone.classList.remove('is-over'));
    }
    show();
  }

  // Under 600 pixels the nav folds behind the Menu button.
  function menu() {
    const button = document.querySelector('.menu-button');
    const nav = document.getElementById('site-nav');
    if (!button || !nav) return;
    const set = (open) => {
      nav.classList.toggle('open', open);
      button.setAttribute('aria-expanded', String(open));
    };
    button.addEventListener('click', () => set(!nav.classList.contains('open')));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && nav.classList.contains('open')) {
        set(false);
        button.focus();
      }
    });
    document.addEventListener('click', (event) => {
      if (!nav.contains(event.target) && !button.contains(event.target)) set(false);
    });
  }

  function copyButton(input) {
    const button = document.getElementById('copy-share');
    if (!button || !navigator.clipboard) return;
    button.hidden = false;
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(input.value);
        button.textContent = 'Copied';
      } catch (error) {
        input.select();
        button.textContent = 'Press Ctrl+C';
      }
    });
  }

  menu();
  dropZone();
  const form = document.getElementById('start-form');
  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      start(form);
    });
  }
  const shareLink = document.getElementById('share-link');
  if (shareLink) {
    shareLink.addEventListener('focus', () => shareLink.select());
    copyButton(shareLink);
  }
  refresh();
}());
