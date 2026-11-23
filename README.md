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

## AWS setup

To run jobs in your own AWS account, follow [docs/aws-setup.md](docs/aws-setup.md). The account owner deploys one CloudFormation stack from `cloud/aws-setup.yaml`, and each person who runs jobs sets up two AWS CLI profiles.

## Run your own copy

To run jobs in your own AWS account you need:

- an AWS account with GPU quota (G instances in us-east-1; a new account starts at 0 and has to ask AWS to raise it);
- the CloudFormation stack from the aws-setup spec deployed and the two AWS CLI profiles set up, per [docs/aws-setup.md](docs/aws-setup.md);
- the machine image built once. That's a paid run of about $2 and about 75 minutes.

Then, from the repo root, in order:

```
# Build the machine image once (paid, ~$2, ~75 min).
uv run python -m cloud.image

# Prove the fresh image works before any long job (paid).
uv run python -m cloud.smoke

# Start a job from a video (paid). Prints the job id and the machine it launched.
uv run python -m cloud.start path/to/video.mov --preset fast

# Check progress and fetch the results (free).
uv run python -m cloud.status <job-id>
uv run python -m cloud.download <job-id>
```

The paid commands (`cloud.image`, `cloud.smoke`, `cloud.start`) print the machine, the expected time and the rough cost first and start nothing unless `APPROVED_SPEND=1` is set on that one command. The $2 and 75 minutes are estimates; check them against the prices and times in `[cloud]` in `config.toml`. [docs/pipeline.md](docs/pipeline.md) describes the cloud run and all the commands.
