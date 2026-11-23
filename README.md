# Voxelis-Kiroween

A pipeline that turns a phone video into a 3D Gaussian splat, built for Kiroween. Someone films a place, uploads the video and gets back a clean splat, plus a small copy for the web viewer. The app around the pipeline comes later.

The heavy work runs on a GPU machine in AWS, one machine per job. On a laptop you only run small CPU tests.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repo root:

```
uv sync
mkdir data
```

`uv sync` sets up `.venv/` with Python 3.11, pytest and Hypothesis. If you don't have Python 3.11, uv downloads it.

The clean step also needs Node 22 or newer. From the repo root:

```
npm ci
```

`npm ci` installs splat-transform 3.8.0 into `node_modules/`. The tests that run splat-transform are skipped until you've installed Node and run `npm ci`.

`data/` holds every video, frame, splat and weight. Git ignores it, so none of that ends up in the repo.

The laptop install leaves out the GPU packages (torch, gsplat and the depth model). Those only go in the cloud machine image.

## Run the tests

```
uv run pytest
```
