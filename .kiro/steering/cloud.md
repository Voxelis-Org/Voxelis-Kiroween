---
inclusion: fileMatch
fileMatchPattern: "cloud/**"
---

# Cloud code

- The cloud code only wraps the pipeline functions. It never changes what they do.
- Every paid entry point calls the spend guard first, before it creates or starts anything.
- The website takes its reservation, a free write to its table, just before the spend guard, which still comes before any upload, copy or launch.
- Machine types, prices and time caps come from `config.toml`.
- Every instance gets our project tag from `config.toml`: `project=voxelis-kiroween`.
- Every instance is set to terminate when it shuts down (`InstanceInitiatedShutdownBehavior=terminate`).
