---
inclusion: always
---

# Product

We're building a pipeline that turns a phone video into a 3D Gaussian splat.

Someone films a place with their phone and uploads the video. They get back a clean splat, plus a small copy for the web viewer.

The website is the app around the pipeline. A judge signs in with the shared account, picks a video and a preset and presses Start. The video goes straight to storage, the website starts the job on one GPU machine, and the job list and each job's result page show its status. A finished job's page has its 3D viewer and one download, and its owner can make a secret share link that shows that one result to anyone who has it. The showcase shows the finished jobs the team picks.
