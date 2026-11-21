---
inclusion: always
---

# Rules

These apply to Kiro and to everyone on the team.

## Licences

- Our code and our Python and npm dependencies only use permissive licences:
  - MIT, including MIT-0 and MIT-CMU
  - BSD-2-Clause, BSD-3-Clause and 0BSD
  - Apache-2.0, including Apache-2.0 with the LLVM exception
  - ISC, PSF-2.0 and matplotlib's PSF-style licence, CNRI-Python (the licence of Python's early code), Zlib, BSL-1.0, the Unlicense and CC0

  torch, numpy, Pillow, typing-extensions, regex and splat-transform's npm packages use some of these.
- Never in our code or our Python and npm dependencies: GPL, AGPL, LGPL or any other copyleft licence, anything non-commercial, or anything without a clear licence. The only exceptions are the bundled libraries and named exceptions below.
- So no code from Inria's gaussian-splatting repo or anything copied from it, no nvdiffrast and no non-commercial model weights. Weights often have a different licence from their code, so check both.
- Any other licence: ask Samarth before adding it.
- ffmpeg and COLMAP are only command line tools on our private machine image. Don't vendor them and don't link them into our code. Never make the image public or share it, because that would mean distributing their GPL parts.
- Parts of COLMAP have licences we can't use: SiftGPU (non-commercial), CGAL (GPL) and LSD (AGPL). So we build COLMAP without them, and SIFT runs on the CPU. The exact flags are in tech.md.
- Every dependency, tool, model weight and dataset gets a line in `docs/third-party-licences.md`, with its licence and how we use it. Whoever adds one adds its line in the same commit.

### Bundled libraries

- We judge a package by its own licence. Shared libraries it bundles inside its wheel are OK if they're under a runtime exception (like the GCC runtime library exception) or LGPL, as long as we use them unmodified.
- The same goes for standard system runtimes that their maker lets programs ship, like Microsoft's Visual C++ runtime on Windows.
- That covers libgfortran and libquadmath in numpy and scipy, the Visual C++ runtime in numpy's Windows wheels, and FFmpeg inside `opencv-python-headless`. The other `opencv-python-headless` conditions below still apply.
- Note a package's bundled libraries on its line in `docs/third-party-licences.md`.

### Named exceptions

- Hypothesis (MPL-2.0): only as a test dependency.
- certifi (MPL-2.0) and tqdm (MPL-2.0 and MIT): used unmodified. certifi comes in through requests and tqdm through huggingface-hub.
- `opencv-python-headless`: its wheels ship FFmpeg under LGPL-2.1. Never plain `opencv-python`. All the OpenCV packages install the same cv2 module, and OpenCV's own README says to install only one of them per environment. We use the headless one because a server needs no GUI parts, and MoGe asks for it too. Never use cv2 to read or write video, frames always come from the ffmpeg command line.
- torch's CUDA wheels: torch plus the `nvidia-*` and `cuda-*` packages it pulls in, which are under NVIDIA's own licence. Only on the cloud machine image.

## Footage and other data

- Only our own phone videos, or footage whose licence allows it: CC0, CC BY, CC BY-SA or the Pexels licence.
- Anything made from CC BY or CC BY-SA footage, like a splat or a clip, has the author's credit next to it.
- Anything made from CC BY-SA footage is also shared under CC BY-SA 4.0 (share-alike).

## Our own code

- Write our code fresh. Use libraries, don't paste code from other repos.
- If you adapt something from a library's own examples (like gsplat's trainer example), keep its licence note.

## Money

- The owner of the AWS account that the jobs run on approves people in advance, by adding them to the project's AWS group. That is their OK to run jobs on that account. After that, whoever is in the chat says yes to each paid run (cloud machines, cloud runs, deploys), once Kiro has told them the machine, the expected time and the rough cost.
- The exception is jobs that people start through the app (the judges now, other users later). The account owner approves those in advance, as a daily job limit and a daily cost limit that the app enforces. The numbers come in the app spec.
- Kiro and the team still need an OK for every run they start by hand.
- It runs on AWS, in the account of whoever deploys it (for our team, Samarth's).
- Speed and reliability matter more than saving a few cents, so pick the faster machine when it helps.
- Once the person in the chat says yes to a paid run, Kiro runs our start or smoke-test command itself, with `APPROVED_SPEND=1` on that one command.
- The code has a spend guard: nothing paid starts unless `APPROVED_SPEND=1` is set. Kiro never sets it otherwise.
- One machine per job at most. No automatic retries or relaunches.
- Nothing stays running after a job. Every machine shuts itself down when its job ends.

## AWS access

- Everything runs in us-east-1.
- Kiro uses the read-only profile `kiroween-read`. It uses the runner profile `kiroween-runner` only inside our start, smoke-test and stop-all commands, and never any other AWS profile.
- Never rely on `AWS_PROFILE`. AWS keys saved in environment variables beat it, so `AWS_PROFILE=kiroween-read` can run as another account. Only a profile named in the command or in the code beats them.
- In aws commands, always pass `--profile kiroween-read`.
- In Python on our laptops, always name the profile with `boto3.Session(profile_name=...)`, never `AWS_PROFILE` or the default credential chain.
- Code running on the job machine uses its instance role.
- The AWS setup spec makes both profiles. Until `kiroween-read` exists, Kiro doesn't touch AWS at all.
- Our team's account also holds an older project's resources, in other regions and in IAM and S3. Leave them exactly as they are: never delete, change, reuse or depend on them.
- Never list, name or describe the older project's resources anywhere, not in code, docs, commits or chats. So don't run commands that would show them, like `aws s3 ls` or `aws iam list-roles`. Look up only our own resources, by their names or our tag.
- Instead, the AWS setup spec makes our own resources, with new names and our tag (`project=voxelis-kiroween`).
- Never put AWS keys or the account ID in the repo, in chats or in logs.

## Laptop

- Don't run heavy stuff on the laptop. Small CPU tests are fine.

## Git

- No videos, frames, splats, weights or other big files in git. They go in `data/`.
- Nothing over 1 MB in git without asking first.
- Work on feature branches (`feat/<topic>`), never straight on `main`.
- Small commits. Commit after every finished task.
- Stage files by name. Never `git add -A` or `git add .`.

## Hooks

- If a hook blocks something it shouldn't, say so and fix the hook. Never work around a hook or turn one off without the user's OK.

## Specs

- In every spec, the unit test and property test tasks are required. Never mark them optional.
- Keep every spec to 10 tasks or fewer, each task writing its own tests.
