// The viewer's URL parameter: which splat and which cameras file it loads.
// Run by tests/test_viewer.py with Node's own test runner, so no npm package is needed.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ViewerError, readViewerParams, splatKind, viewerSources } from '../../app/viewer/params.js';

const PAGE = 'http://localhost:8000/app/viewer/index.html';
const SIGNED = 'https://voxelis-jobs.s3.us-east-1.amazonaws.com/jobs/20261119-142530-7f3a9c/out/clean.sog'
    + '?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc';

test('the splat kind comes from the path, whatever the query string says', () => {
    assert.equal(splatKind('/jobs/x/out/clean.sog'), 'sog');
    assert.equal(splatKind('/jobs/x/out/TRAIN.PLY'), 'ply');
    assert.equal(splatKind(SIGNED), 'sog');
    assert.equal(splatKind('https://h/a.ply?name=b.sog'), 'ply');
    assert.equal(splatKind('https://h/out/cameras.json'), null);
    assert.equal(splatKind('https://h/out/clean'), null);
});

test('a .sog file gives that file and the cameras.json beside it', () => {
    const sources = readViewerParams('?url=/data/jobs/j/out/clean.sog', PAGE);
    assert.deepEqual(sources, {
        splatUrls: ['http://localhost:8000/data/jobs/j/out/clean.sog'],
        camerasUrl: 'http://localhost:8000/data/jobs/j/out/cameras.json',
    });
});

test('a .ply file is used as it is', () => {
    const sources = readViewerParams('?url=https%3A%2F%2Ffiles.example%2Fout%2Ftrain.ply', PAGE);
    assert.deepEqual(sources.splatUrls, ['https://files.example/out/train.ply']);
    assert.equal(sources.camerasUrl, 'https://files.example/out/cameras.json');
});

test("a job's out/ folder tries clean.sog first, then train.ply", () => {
    const sources = readViewerParams('?url=/data/jobs/j/out/', PAGE);
    assert.deepEqual(sources.splatUrls, [
        'http://localhost:8000/data/jobs/j/out/clean.sog',
        'http://localhost:8000/data/jobs/j/out/train.ply',
    ]);
    assert.equal(sources.camerasUrl, 'http://localhost:8000/data/jobs/j/out/cameras.json');
});

test('a relative address resolves against the page', () => {
    const sources = readViewerParams('?url=../../data/jobs/j/out/train.ply', PAGE);
    assert.deepEqual(sources.splatUrls, ['http://localhost:8000/data/jobs/j/out/train.ply']);
});

test("a signed link keeps its query, and there's no cameras file to guess beside it", () => {
    const sources = viewerSources(SIGNED, undefined, PAGE);
    assert.deepEqual(sources, { splatUrls: [SIGNED], camerasUrl: null });
});

test('a cameras address, given, wins over the guess', () => {
    const cameras = 'https://voxelis-jobs.s3.us-east-1.amazonaws.com/jobs/j/out/cameras.json?X-Amz-Signature=def';
    const sources = readViewerParams(`?url=${encodeURIComponent(SIGNED)}&cameras=${encodeURIComponent(cameras)}`, PAGE);
    assert.deepEqual(sources, { splatUrls: [SIGNED], camerasUrl: cameras });
});

test('a missing or empty url is refused with a message saying what to give', () => {
    for (const search of ['', '?url=', '?url=%20%20', '?cameras=/out/cameras.json']) {
        assert.throws(() => readViewerParams(search, PAGE), (error) => error instanceof ViewerError && /out\/clean\.sog/.test(error.message));
    }
});

test('a file that is neither .sog nor .ply is refused', () => {
    assert.throws(() => readViewerParams('?url=/out/cameras.json', PAGE), (error) => error instanceof ViewerError && /\.sog or \.ply/.test(error.message));
    assert.throws(() => readViewerParams('?url=/out/clean.spz', PAGE), ViewerError);
});

test('only web addresses are taken, for the splat and for the cameras', () => {
    for (const bad of ['javascript:alert(1)//x.sog', 'data:text/plain,x.sog', 'file:///tmp/x.ply', 'blob:http://h/x.sog']) {
        assert.throws(() => viewerSources(bad, undefined, PAGE), ViewerError, bad);
        assert.throws(() => viewerSources('/out/clean.sog', bad, PAGE), ViewerError, bad);
    }
});
