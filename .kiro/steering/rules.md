---
inclusion: always
---

# Rules

These apply to Kiro and to everyone on the team.

## Licences

- Only MIT, BSD or Apache, in our code and in our dependencies. The named exceptions below are the only other licences allowed.
- Nothing non-commercial. So no code from Inria's gaussian-splatting repo or anything copied from it, no nvdiffrast and no non-commercial model weights. Weights often have a different licence from their code, so check both.
- ffmpeg and COLMAP are OK as command line tools we call on the server. Don't vendor them and don't link them into our code.
- COLMAP bundles SiftGPU, which is non-commercial. So we build COLMAP without CUDA, OpenGL or the GUI, which keeps SiftGPU out, and SIFT runs on the CPU. The exact flags are in tech.md.
- Anything else that isn't MIT, BSD or Apache: ask Samarth before adding it.

### Named exceptions

- Hypothesis (MPL-2.0): only as a test dependency.
- `opencv-python-headless`: its wheels ship FFmpeg under LGPL-2.1. Never plain `opencv-python`, whose Linux wheels also ship Qt under LGPL-3. Never use cv2 to read or write video, frames always come from the ffmpeg command line.
- torch's CUDA wheels: they come with NVIDIA's CUDA libraries under NVIDIA's own licence. Only on the cloud machine image.

## Our own code

- Write our code fresh. Use libraries, don't paste code from other repos.
- If you adapt something from a library's own examples (like gsplat's trainer example), keep its licence note.

## Money

- Anything that costs money (cloud machines, cloud runs, deploys) needs an OK for that exact run from whoever owns the AWS account it runs on.
- It runs on AWS, on a friend's account for now, so be careful with every dollar.
- Kiro never starts anything paid. It gives the command and the rough cost to the person in the chat, who runs it in their own terminal.
- The code has a spend guard: nothing paid starts unless `APPROVED_SPEND=1` is set. Kiro never sets it.
- One machine per job at most. No automatic retries or relaunches.
- Every machine shuts itself down when its job ends.

## AWS access

- Kiro only ever uses the read-only profile `kiroween-read`. Always name it (`--profile kiroween-read` or `AWS_PROFILE=kiroween-read`) so nothing falls back to the default profile.
- Kiro never uses the runner profile `kiroween-runner`, or any other AWS profile.
- The AWS setup spec makes both profiles. Until `kiroween-read` exists, Kiro doesn't touch AWS at all.
- Never put AWS keys or the account ID in the repo, in chats or in logs.

## Laptop

- Don't run heavy stuff on the laptop. Small CPU tests are fine.

## Git

- No videos, frames, splats, weights or other big files in git. They go in `data/`.
- Nothing over 1 MB in git without asking first.
- Work on feature branches (`feat/<topic>`), never straight on `main`.
- Small commits. Commit after every finished task.
- Stage files by name. Never `git add -A` or `git add .`.

## Specs

- In every spec, the unit test and property test tasks are required. Never mark them optional.
