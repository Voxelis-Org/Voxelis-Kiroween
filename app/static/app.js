// The Page_Script: the Start_Precheck, the part upload straight to storage, and the status
// refresh on a Result_Page that's still starting or running. Plain JavaScript, no build step,
// no inline handlers. Every request to the website carries the CSRF_Token in X-CSRF-Token.
'use strict';

(function () {
  const main = document.querySelector('main');
  const tokenTag = document.querySelector('meta[name="csrf-token"]');
  const csrf = tokenTag ? tokenTag.content : '';

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

  async function uploadParts(file, upload, retries, progress) {
    const parts = [];
    let sent = 0;
    for (let n = 1; n <= upload.parts; n += 1) {
      const blob = file.slice((n - 1) * upload.part_bytes, Math.min(n * upload.part_bytes, file.size));
      let result = null;
      for (let attempt = 0; attempt <= retries; attempt += 1) {
        const link = await api('POST', `/api/uploads/${upload.upload}/parts/${n}`, {});
        result = await putPart(link.url, blob, (loaded) => { progress.value = (sent + loaded) / file.size; });
        if (result.status !== 403) break;
      }
      if (!result || result.status < 200 || result.status >= 300 || !result.etag) {
        await api('POST', `/api/uploads/${upload.upload}/abort`, {}).catch(() => {});
        throw new Refusal('The upload stopped. Check your connection and start again.');
      }
      sent += blob.size;
      progress.value = sent / file.size;
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
      progress.hidden = false;
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
        if (text) text.textContent = status.text;
        if (status.final) {
          clearInterval(timer);
          location.reload();
        }
      } catch (error) {
        clearInterval(timer);
      }
    }, seconds * 1000);
  }

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
  }
  refresh();
}());
