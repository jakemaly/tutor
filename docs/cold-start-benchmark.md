# Modal cold-start benchmark

This benchmark measures a genuine cold start: the time from the first `/api/chat`
request until the first nonempty answer text. It covers the real backend and the Modal
deployment in [deployment/modal_qwen.py](../deployment/modal_qwen.py). It splits the wait
into startup stages, so changes to the deployment can be compared using the same method.

**Deployment history.** The baseline, Triton-cache, and control runs measured the earlier
plain deployment: a single Modal function that started vLLM on every cold start. The
snapshot runs measured a separate snapshot app (`socratic-tutor-qwen-snapshot`). After the
experiment, the plain deployment was removed. `modal_qwen.py` now contains only the
snapshot server, deployed as `socratic-tutor-qwen`. Both sets of results are kept below as
measured.

**Settings since the measurements.** Every performance run below used
`--max-model-len 4096` and a 512-token output cap (`max_new_tokens`). The token-limit
change (#48) later removed the cap and switched to `--max-model-len auto`. Mentions of
"limit 512" below describe the runs as they were measured. The final configuration (#48's
settings in the consolidated snapshot server) was checked separately for correctness, not
re-benchmarked: see
[Validation with the token-limit settings](#validation-with-the-token-limit-settings-2026-11-18).

Raw results: [docs/cold-start-results/](cold-start-results/) (one JSON file per trial).

## Summary

Two changes were measured with this method on 2026-11-14:

| Deployment | Time to first answer text (median, range) | Trials |
|---|---|---|
| Baseline (kernels recompiled on every start) | 211.2 s (206.8–230.8) | 3 |
| Persistent Triton cache ([modal_qwen.py](../deployment/modal_qwen.py)) | 171.1 s (155.0–181.4); contemporaneous control 156.8 s (150.3–162.8) | 3 + 3 |
| GPU memory snapshot, **restored** (now [modal_qwen.py](../deployment/modal_qwen.py)) | **12.3 s** (6.6–98.6) | 6 |
| GPU memory snapshot, **snapshot being created** | 196.3 s (189.9–197.1) | 3 |

- The **persistent Triton cache** removed about 60 s of kernel compilation from every cold
  start ([details](#triton-kernel-cache)).
- A **GPU memory snapshot** restore removed everything between container start and a
  ready vLLM server: imports, GGUF parsing, model loading, CUDA and kernel setup, and API
  startup. It cut the contemporaneous control's median by 144.5 s (−92%). Cold starts that
  create a snapshot are 39.5 s slower than the control. In this run, 3 of the first 5 cold
  starts created one, and none of the 4 that followed did
  ([details](#memory-snapshot-experiment)).

The strongest conclusion the measurements support: once a snapshot exists for the worker,
a restored cold start reaches the first answer in about 7–20 s instead of about 150–160 s.
The exception was one run that waited 85 s for Modal to assign a GPU.
The rate at which Modal will need new snapshots in normal use was not measured. Snapshots
remain faster on average unless more than about 79% of cold starts are creations.

A re-test on 2026-11-18 added the route guard that hides vLLM's dev-mode routes. It
restored on an L40S in 6.2 s and 11.3 s, and on an L4 in 27.7 s from a separate L4
snapshot. One restore waited 313.9 s for Modal to assign a GPU
([details](#re-test-with-the-route-guard-2026-11-18)).

A final validation on 2026-11-18 deployed the consolidated server with #48's settings
(`--max-model-len auto`, no output cap). Snapshot creation and two restores worked on an
L40S, with no memory errors, and the feature and route-guard checks passed. It was a
correctness check, not a new benchmark
([details](#validation-with-the-token-limit-settings-2026-11-18)).

## Triton kernel cache

The **largest avoidable delay** was the recompilation of the same ~1,100 Triton kernel files
(55.7 MB) on every cold container. vLLM compiles them while it profiles and warms up the
Qwen3.5 model. The cache lived in the container's temporary home directory, so it was lost
at scale-down. The deployment now keeps the cache on the existing model Volume
(`/cache/triton-cache`, set only for the vLLM process).

Measured on 2026-11-14. Every trial ran on an L40S in `us-east` with the same Modal image
(`im-R4bcBoIDZM5UpqKybmE17n`), prompt (617 prompt tokens), and vLLM flags. The model
answered without a tool call in every trial.

| | Baseline (n=3) | After (n=3) | Change |
|---|---|---|---|
| **Time to first answer text** | **211.2 s** (206.8–230.8) | **171.1 s** (155.0–181.4) | **−40.2 s median (−19%)** |
| `engine_init` (the targeted stage) | 74.2 s (72.2–75.1) | 15.2 s (13.4–22.2) | −59.0 s |
| `first_token` (ready → first token) | 3.0 s (2.9–3.9) | 1.5 s (1.2–6.3) | −1.5 s |

The ranges do not overlap: the slowest run after the change (181.4 s) beat the fastest
baseline run (206.8 s).

The total improved by less than `engine_init` because two CPU-bound stages that come before
the kernel cache is read were slower in the second set. `vllm_import` + `engine_spawn` took a
median of 58.2 s after the change, against 47.6 s before. One run took 50.5 s for
`engine_spawn` alone. Baseline run 1 shows the same spread, so this is run-to-run variation
between Modal workers, not an effect of the change. Three trials cannot separate it further.

Per-trial stages, in seconds (definitions below). Container IDs are the last 8 characters of
each Modal task ID:

| Trial | Container | GPU | Weights | Kernel cache (before → after) | local_backend | provisioning | runtime_init | cache_scan | tokenizer_sync | vllm_import | engine_spawn | model_load | engine_init | api_startup | first_token | return_path | backend_stream | **TTFT** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-1 | `3YSXBJ8R` | L40S | Volume hit, 0.0 MB downloaded | 0 → 1103 files | 0.03 | 9.1 | 1.8 | 0.16 | 0.18 | 20.4 | 39.9 | 79.1 | 74.2 | 2.0 | 3.9 | 0.07 | 0.00 | **230.8** |
| baseline-2 | `7K1SXQHR` | L40S | Volume hit, 0.0 MB downloaded | 0 → 1103 files | 0.03 | 8.5 | 1.3 | 0.03 | 0.03 | 13.6 | 32.6 | 74.1 | 72.2 | 1.6 | 2.9 | -0.09 | 0.00 | **206.8** |
| baseline-3 | `5W2D6HPR` | L40S | Volume hit, 0.0 MB downloaded | 0 → 1103 files | 0.02 | 2.7 | 1.1 | 0.02 | 0.04 | 13.8 | 33.8 | 79.6 | 75.1 | 2.0 | 3.0 | -0.10 | 0.00 | **211.2** |
| cache-priming | `WBGV4T4R` | L40S | Volume hit, 0.0 MB downloaded | 0 → 1103 files | 0.03 | 2.3 | 1.1 | 0.03 | 0.04 | 15.3 | 33.4 | 77.7 | 74.4 | 1.6 | 3.3 | -0.29 | 0.00 | **209.0** |
| after-1 | `GX6TR6SR` | L40S | Volume hit, 0.0 MB downloaded | 1103 → 1103 files | 0.01 | 1.5 | 1.9 | 1.67 | 0.04 | 18.4 | 39.3 | 78.4 | 22.2 | 1.7 | 6.3 | -0.32 | 0.00 | **171.1** |
| after-2 | `5QSP9K5R` | L40S | Volume hit, 0.0 MB downloaded | 1103 → 1103 files | 0.02 | 8.4 | 1.4 | 0.43 | 0.06 | 20.7 | 50.5 | 80.7 | 15.2 | 2.9 | 1.5 | -0.37 | 0.00 | **181.4** |
| after-3 | `8Q4GKQ6R` | L40S | Volume hit, 0.0 MB downloaded | 1103 → 1103 files | 0.03 | 1.7 | 1.2 | 0.44 | 0.04 | 18.5 | 39.7 | 77.3 | 13.4 | 1.9 | 1.2 | -0.39 | 0.00 | **155.0** |

Notes on the runs (result files: `baseline-1..3.json` for baseline-1..3,
`optimized-prime-1.json` for cache-priming, and `optimized-1..3.json` for after-1..3):

- **The cache-priming run is excluded from the comparison.** It was the first cold start
  after deploying the change. Its kernel cache was empty, so it compiled and saved the
  kernels (209.0 s, like the baseline). The same thing happens once after any change that
  alters the cache keys, such as a vLLM or Triton upgrade.
- **after-1 was slower than after-2 and after-3** for `engine_init` (22.2 s against about
  14 s) and `first_token` (6.3 s against about 1.3 s). It was the first container to read
  the newly written cache files from the Volume. Later runs matched a diagnostic that kept
  the same cache on local disk (13.5 s).
- **`cache_scan` is benchmark overhead.** It stats every cache file before vLLM starts. With
  1,103 Volume files it costs 0.4–1.7 s, which is charged against the "after" runs.
- **One unrecorded pilot run is excluded.** It ran before the log-fetch fix, so only its
  container logs exist. It ran in `eu-west`, while all recorded runs used `us-east`.
- **No L4 was assigned.** The function still allows `["L4", "L40S"]`, but Modal assigned an
  L40S every time. L4 cold starts were not measured.

### How the delay was identified

A diagnostic container ran the same vLLM command twice. With an empty Triton cache,
`init engine (profile, create kv cache, warmup model)` took 73.9 s. When run again with the
cache it had just produced, it took 13.5 s, and the first request dropped from 3.3 s to
1.0 s. Every baseline cold start wrote the same 1,103 files (55.7 MB) to
`~/.triton/cache`, and that directory was discarded at scale-down.

## What was ruled out

- **Model and tokenizer downloads.** In every trial, the pinned GGUF (5.68 GB), the
  multimodal projector (0.92 GB), and the tokenizer snapshot were already on the Volume.
  The container received about 0 MB over its network before vLLM was ready. Tokenizer and
  config sync took 0.03–0.2 s.
- **Volume read speed.** On a fresh CPU container, the 5.68 GB GGUF was read in 4.4 s
  (about 1.3 GB/s). Running 8 or 16 parallel readers did not help, so prefetching would
  not help.
- **GGUF parsing (second-largest, not changed).** A diagnostic run timed every
  `gguf.GGUFReader` construction. During model loading, the vLLM GGUF plugin parsed the full
  file's metadata 6 times in the engine process (10.6–13.3 s each, CPU-bound, the same with
  a warm page cache): vocabulary size, lm_head check, tensor names twice, unquantized
  parameters, then the weight iterator. About 58 s of the ~79 s model-load stage is the five
  redundant parses. This is the next candidate. A per-process reader cache would need a hook
  inside vLLM's processes, so it was left out of this change. A memory snapshot (below)
  skips this parsing on restored starts without changing vLLM.

## Memory snapshot experiment

### What a snapshot captures

During the experiment, the snapshot server ran as a separate Modal app
(`socratic-tutor-qwen-snapshot`). It used the same image, Volume, GPU list
(`["L4", "L40S"]`), `cpu=2`, `memory=16384`, `min_containers=0`, `max_containers=1`,
`scaledown_window=60`, and the same `vllm serve` command as the plain deployment it was
compared against. It is now the only server in
[modal_qwen.py](../deployment/modal_qwen.py). It enables `enable_memory_snapshot=True` and
`experimental_options={"enable_gpu_snapshot": True}` on a class. The class has:

1. A `@modal.enter(snap=True)` hook. It runs the whole normal startup: tokenizer sync, `vllm
   serve` (with `--enable-sleep-mode`) as a subprocess, GGUF parsing, model load, CUDA and
   Triton kernel setup (from the Volume kernel cache), and API startup. It then sends two
   short warmup requests (text and tool calling), resets the prefix cache, and puts vLLM
   to sleep at level 1. Level 1 copies the weights to host memory and frees the KV cache, so
   the snapshot doesn't store about 30 GB of empty KV-cache pages. Modal then snapshots the
   container's CPU memory (both vLLM processes) and GPU state.
2. A `@modal.enter(snap=False)` hook. It runs after every restore and calls vLLM's
   `/wake_up`, which copies the weights back to the GPU and reallocates the KV cache.
3. A `@modal.web_server(port=8000, requires_proxy_auth=True)` method. It serves requests
   only after the wake-up hook returns.

**Work a restored container skips** (control medians, about 151 s in total): container
runtime import and tokenizer sync (1.3 s); vLLM CLI imports (18.1 s); config resolution,
engine process spawn, and CUDA initialization (37.7 s); model load, mostly the six GGUF
metadata parses (78.3 s); profiling, KV-cache allocation, and kernel loading (13.6 s); and
API-server startup, including the multimodal processor (1.9 s).

**Work that still runs after a restore:**

- Modal routing and GPU scheduling.
- Loading the snapshot. This covers the memory of both vLLM processes, including the
  weights held in host memory, and the CUDA state. Modal reads it from the worker's local
  cache or from its distributed storage.
- The wake-up: weights copied back to the GPU and the KV cache reallocated (0.6–1.1 s).
- Routing the held request to the container, then prefill and the first token. The engine
  reported 0.63–0.94 s from request arrival to first token.
- The return stream through the backend.

Nothing is downloaded, and nothing is read from the model files on the Volume.

### Compatibility checks

- **Modal 1.6.3** (installed client): `App.cls` accepts `enable_memory_snapshot` and
  `experimental_options`, `modal.enter` takes `snap=`, and `web_server` works on a class
  method. Modal labels GPU memory snapshots alpha. Snapshots exist only for deployed apps.
  Modal keeps them per worker type and invalidates them on redeploy. Modal's own vLLM
  example uses the same pattern: `vllm serve` started with `subprocess.Popen` inside the
  `snap=True` hook, followed by sleep and wake-up.
- **vLLM 0.32.0** (pinned): `--enable-sleep-mode` uses the cumem allocator on CUDA. Level 1
  sleep offloads the tensors tagged `weights` and discards the KV cache. `/sleep`,
  `/wake_up`, and `/reset_prefix_cache` are registered only when `VLLM_SERVER_DEV_MODE=1`.
  `--middleware` takes an import path; vLLM adds a class with `app.add_middleware()`,
  after its own middleware, so it sees each request first.
- **GGUF plugin**: the quantized GGUF weights are ordinary GPU tensors to the allocator.
  Sleep, snapshot, restore, and wake-up worked without changes. Answers after restore were
  correct and used the same 617 prompt tokens.
- **Subprocess server**: the vLLM API server and engine processes, their pipes, the
  listening socket, and the log-forwarding thread all survived restore. vLLM's log lines
  continued with offsets counted from the original spawn.

### Results

Trials ran in this order:

1. Snapshot trials 1–5.
2. Contemporaneous control trials 1–3 on the production deployment (with the persistent
   Triton cache).
3. Snapshot trials 6–9.

All runs used an L40S with the same image (`im-R4bcBoIDZM5UpqKybmE17n`), prompt, and vLLM
flags. No model called a tool before answering.

| | Control (n=3) | Snapshot restored (n=6) | Snapshot created (n=3) |
|---|---|---|---|
| **Time to first answer text** | **156.8 s** (150.3–162.8) | **12.3 s** (6.6–98.6) | **196.3 s** (189.9–197.1) |
| Change against control (median) | | **−144.5 s (−92%)** | +39.5 s |

Restored trials, in seconds. `provisioning_restore` ends when the `snap=False` hook starts.
The "listed" column is when the container first appeared in `modal container list`, polled
about every 3 s on the client clock:

| Trial | Container | Snapshot from | Region | Listed | provisioning_restore | wake_up | first_token | return_path | backend_stream | **TTFT** |
|---|---|---|---|---|---|---|---|---|---|---|
| snapshot-3 | `GE1SHSDR` | `0AX3NEDR` (AWS us-east) | us-east | 10.6 | 17.4 | 1.0 | 2.4 | -0.6 | 0.00 | **20.3** |
| snapshot-4 | `J4W2E32R` | `DD9REXMR` (OCI eu-west) | us-east | 1.5 | 8.2 | 1.1 | 1.2 | -0.4 | 0.00 | **10.1** |
| snapshot-6 | `WEDKENKR` | `DD9REXMR` (OCI eu-west) | us-east | 3.4 | 4.8 | 1.1 | 1.1 | -0.4 | 0.00 | **6.6** |
| snapshot-7 | `QWX6D51R` | `0AX3NEDR` (AWS us-east) | us-east | 1.5 | 10.4 | 0.6 | 1.1 | -0.4 | 0.00 | **11.9** |
| snapshot-8 | `KNXYZCDR` | `0AX3NEDR` (AWS us-east) | us-east | 8.1 | 11.2 | 0.7 | 1.1 | -0.4 | 0.00 | **12.7** |
| snapshot-9 | `4XGEMVJR` | `0AX3NEDR` (AWS us-east) | us-east | 84.7 | 97.1 | 0.7 | 1.1 | -0.4 | 0.00 | **98.6** |

Creation and control trials, in seconds (the stages are defined under [Method](#stages)):

| Trial | Container | vllm_import | engine_spawn | model_load | engine_init | api_startup | warmup | sleep | snapshot | wake_up | first_token | **TTFT** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control-1 | `N7S44XCR` | 19.4 | 37.7 | 75.9 | 14.4 | 1.9 | – | – | – | – | 1.9 | **156.8** |
| control-2 | `2ZMR230R` | 18.1 | 38.4 | 78.3 | 13.3 | 1.9 | – | – | – | – | 1.1 | **162.8** |
| control-3 | `G546PKSR` | 14.0 | 33.3 | 79.9 | 13.6 | 2.0 | – | – | – | – | 1.0 | **150.3** |
| snapshot-1 | `DD9REXMR` | 22.3 | 38.7 | 84.5 | 25.7 | 2.1 | 9.2 | 2.3 | 4.7 | 1.1 | 1.6 | **197.1** |
| snapshot-2 | `0AX3NEDR` | 29.2 | 48.4 | 81.0 | 14.3 | 2.0 | 1.6 | 1.7 | 3.7 | 0.7 | 1.7 | **189.9** |
| snapshot-5 | `1Q9DYN7R` | 34.7 | 49.0 | 82.0 | 15.3 | 2.0 | 1.8 | 1.7 | 3.3 | 1.4 | 1.5 | **196.3** |

The remaining stages (local_backend, provisioning, runtime_init, cache_scan,
tokenizer_sync, return_path, backend_stream) totalled 5–12 s in each row. They are in the
JSON files.

### How restores were verified

Every restored trial passed all of these checks:

- **Zero containers** were running before the request.
- **New container:** a new task ID that no earlier trial used.
- **Restore after the request:** the `restored` event came after the request was sent.
- **Modal's own message:** Modal logged `Restoring Function from memory snapshot.` for
  that container.
- **Different source container:** the snapshot ID and source task ID were recorded in the
  `snap=True` hook and read back after restore, and they name a different container. The
  restored process's `/proc` boot time also matched the source container's, while its
  environment held the new task ID.
- **No downloads:** the container received 0 MB over the network before serving.

Creation trials logged `Creating GPU memory snapshot for Function.` and then restored in
place: same container, `restored_from_other_container: false`.

### Interpreting the results

- **Restore time varies with scheduling.** In snapshot-9, Modal didn't list a container for
  84.7 s; the request waited for a GPU worker. Its restore and wake-up then took about 13 s,
  like the other trials. The control and the baseline runs could hit the same wait.
  Excluding snapshot-9, restored cold starts took 6.6–20.3 s (median 11.9 s). From the
  "listed" times, loading the snapshot itself took roughly 1–12 s, but the 3 s polling
  makes that split approximate.
- **Snapshot creation is slower than the control.** About 8 s of the extra time is warmup,
  sleep, capture, and wake-up (medians 1.8 + 1.8 + 3.7 + 1.1 s). snapshot-1's first warmup
  request took 9.2 s. The rest comes from `vllm_import` and `engine_spawn`: medians 29.2 and
  48.4 s, against 18.1 and 37.7 s for the control. The snapshot-capable container runtime,
  the cumem allocator, and worker variation could each explain this. Three trials can't
  separate them, and the earlier "after" set showed similar spread for these two stages.
- **How often snapshots are created.** Modal created three snapshots on L40S workers: one
  on OCI `eu-west` and two on AWS `us-east`, the second of those after a snapshot for the
  same provider and region already existed. Restores used both an OCI snapshot (on an OCI
  worker in `us-east`) and an AWS snapshot. Modal's guidance expects 2–3 snapshots per GPU
  type. Any redeploy, image change, or code change starts over. L4 workers were not
  assigned in this run; the [re-test](#re-test-with-the-route-guard-2026-11-18) later
  created and restored a separate L4 snapshot. With these medians, the average cold start
  beats the control as long as fewer than about 79% are creations:
  (156.8 − 12.3) / (196.3 − 12.3).

### Tradeoffs

- **Dev-mode routes need a guard.** `VLLM_SERVER_DEV_MODE=1` also registers
  `/collective_rpc`, `/server_info`, `/reset_*`, and vLLM's RL weight-update routes. Modal
  proxy auth only checks that a caller holds a workspace proxy token, and vLLM warns against
  exposing these routes. The snapshot server therefore loads
  [`vllm_route_guard.RouteGuard`](../deployment/vllm_route_guard.py) into vLLM with
  `--middleware`. Callers outside the container reach only `POST /v1/chat/completions`,
  `GET /v1/models`, and `GET /health`. Every other route answers 404 unless the request
  carries a random token that the container generates at startup and never sends out.
  The snapshot hooks use that token for sleep, wake-up, and metrics. The guard is pure ASGI
  and passes allowed requests through unchanged, so streaming and cancellation are
  unaffected. `/metrics` is internal-only.
- **GPU snapshots are alpha.** Modal says the driver-level checkpoint and restore are new.
  A restore failure was not observed, so the fallback behavior was not measured. There is
  no longer a separate plain app to switch to; the plain deployment is in the Git history.
- **The first cold starts after each deploy are slower** (about +40 s each) until Modal has
  created its snapshots.
- **Code complexity:** a class with three lifecycle hooks, warmup requests, three extra
  vLLM settings (sleep mode, dev mode, the guard), and a 90-line route guard with its own
  tests. The Triton cache is still used: it keeps snapshot-creation runs from recompiling
  kernels.
- **The endpoint URL changes.** The class endpoint has a different URL from the old plain
  function, so `.env` `LLM_BASE_URL` must point at it.

### Functional checks on the snapshot deployment

These checks went through the real backend (`/api/chat`) to the snapshot endpoint:

- **Streaming:** 62 text chunks.
- **PDF citations:** the answer linked `[biology-notes.pdf, p. 1](/api/attachments/…)`.
- **Images:** a red image was answered "Red".
- **Proxy auth:** HTTP 401 without a token or with an invalid one, and 200 with a valid one.
- **Cancellation:** the backend stream was closed after the first text, the way the
  browser's Stop button does. In 3 of 3 measured tries, the request had reached the engine
  and stopped after 3 generated tokens (the limit is 512), with nothing left running. In a
  fourth try, Modal's log shows the request ending after 262 ms.
- **Tool calling:** a hint prompt completed a `load_skill` round trip through the backend
  (tool call, tool result, then the answer) on a restored cold start, in 10.7 s. With
  `tool_choice="required"`, vLLM's `qwen3_xml` parser returned
  `load_skill(name="worked-example")` for the tutor's real system prompt.

Whether the model calls a tool varies from request to request on both deployments. With
`tool_choice="auto"`, the worked-solution prompt was answered without a tool in 3 of 3
tries on each deployment.

**Cancellation on the plain deployment.** The same backend check passed 9 times on the
plain deployment and failed 3 times. In the failures, the request ran to the full 512
tokens after the client disconnected. All 3 failures were consecutive, on one backend
process and one Modal container, and were not reproduced afterwards, including after
replaying the same request sequence and after a cold start that used Modal's `303`
redirect. Closing a stream sent straight to either endpoint stopped vLLM after 2 tokens.
The cause was not found. It occurred only on the plain deployment, so the snapshot change
did not cause it.

### Re-test with the route guard (2026-11-18)

After the route guard was added, the snapshot app was redeployed, which discarded the old
snapshots. Six more cold starts were measured with the same method, prompt, and settings
(`snapshot-guard-*.json`). This time Modal assigned an L4 twice:

| Trial | Mode | GPU | Container listed | TTFT | Notes |
|---|---|---|---|---|---|
| snapshot-guard-1 | created | L40S | – | **176.7 s** | Kernel cache hit (1,111 files) |
| snapshot-guard-2 | restored | L40S | – | **6.2 s** | From snapshot-guard-1's snapshot |
| snapshot-guard-3 | restored | L40S | 313.9 s | **334.2 s** | Waited for a GPU; restore and wake-up then took about 20 s |
| snapshot-guard-4 | created | L4 | – | **392.4 s** | 88.2 s provisioning wait; L4 stages are slower |
| snapshot-guard-5 | restored | L4 | 2.1 s | **27.7 s** | From snapshot-guard-4's L4 snapshot |
| snapshot-guard-6 | restored | L40S | 1.9 s | **11.3 s** | From snapshot-guard-1's snapshot |

- **The guard did not slow anything down.** The L40S restores that did not wait for a GPU
  took 6.2 s and 11.3 s, within the earlier range. Snapshot creation worked through the
  guard, which shows that sleep and wake-up reach vLLM with the internal token.
- **Snapshots are kept per GPU type.** Modal created a separate L4 snapshot and restored
  it only on an L4; the L40S restores used the L40S snapshot. Every restore passed the
  checks in [How restores were verified](#how-restores-were-verified).
- **L4 is slower but works.** The L4 restore took 27.7 s: provisioning and restore
  23.1 s, wake-up 1.2 s, first token 3.7 s. The engine's own time to first token was 2.8 s,
  against 0.6 s on the L40S. The L4 creation did not compile new kernels. The shared
  Triton cache stayed at 1,111 files, so both GPU types use the same cache.
- **Waiting for a GPU is now the largest variable.** snapshot-guard-3 waited 313.9 s
  before Modal listed a container, and snapshot-guard-4 waited 88.2 s. A snapshot cannot
  shorten that wait.

Checks on the guarded deployment, through the real backend where a feature involves it:

- **Route guard:** with a valid proxy token, 15 dev and non-tutor routes returned 404,
  including `/sleep`, `/wake_up`, `/is_sleeping`, `/collective_rpc`, `/server_info`,
  `/reset_prefix_cache`, `/update_weights`, `/metrics`, `/v1/completions`, `/tokenize`,
  `/docs`, and `/openapi.json`. `/health` and `/v1/models` returned 200, and requests
  without a proxy token got 401.
- **Internal access:** inside the container, `/is_sleeping` returned 404 without the
  internal token and 200 (`{"is_sleeping": false}`) with it.
- **Cancellation:** after the backend stream was closed at the first text, vLLM stopped
  after 3 generated tokens (limit 512), with nothing left running. Metrics were read
  inside the container because `/metrics` is no longer public.
- **Streaming, PDF citations, and images** passed as before (62 text chunks; the
  citation link; "Red").
- **Tool calling:** with `tool_choice="required"`, vLLM parsed
  `load_skill(name="worked-example")` through the guarded chat route. Through the
  backend, the model answered 5 tutoring prompts directly without calling the tool; as
  noted above, that varies from request to request and was also seen on the plain
  deployment.

### Validation with the token-limit settings (2026-11-18)

This run checked that the final configuration works: the consolidated snapshot server in
`modal_qwen.py`, deployed as `socratic-tutor-qwen`, with #48's settings
(`--max-model-len auto`, no `max_new_tokens` cap, tool-call flags). It was an integration
check, not a benchmark: it used 4 cold trials and the same prompt and tooling as above, but
too few trials to compare timings. All earlier performance figures were measured with
`--max-model-len 4096` and the 512-token cap. Results: `combined-1..4.json`, and the
filtered vLLM log lines in `combined-vllm-memory.log`.

| Trial | Mode | GPU | Container listed | TTFT |
|---|---|---|---|---|
| combined-1 | created | L40S | 7.5 s | 181.4 s |
| combined-2 | restored (from combined-1's snapshot) | L40S | 1.6 s | 15.3 s |
| combined-3 | created | L4 | 1.5 s | 271.0 s |
| combined-4 | restored (from combined-1's snapshot) | L40S | 11.8 s | 22.1 s |

Both restores passed every check in
[How restores were verified](#how-restores-were-verified). The L4 created a snapshot but
was not assigned again, so **an L4 restore with the final configuration was not
validated**.

**Context length and memory**, from vLLM's logs. No memory errors or tracebacks appeared:

| | L40S | L4 |
|---|---|---|
| Context chosen by `--max-model-len auto` | 262,144 tokens (the model's full length) | 262,144 tokens |
| Model weights on the GPU | 6.44 GiB | 6.44 GiB |
| KV cache | 29.29 GiB, 946,949 tokens (3.61 full-length requests at once) | 10.31 GiB, 333,021 tokens (1.27×) |
| Sleep | freed 35.87 GiB (0.58 GiB left), 1.8 s | freed 16.89 GiB (0.30 GiB left), 3.3 s |
| Wake-up | 0.65 s at creation; 0.61 s and 1.19 s on the restores | 1.08 s at creation |

The KV cache size still comes from `--gpu-memory-utilization 0.85`; `auto` only raised
the per-request context limit. While awake after a restore (L40S), the GPU used
37,402 of 46,068 MiB. The container's cgroup reported 17.1 GiB of host memory in use
(`memory.usage_in_bytes`), and the vLLM engine process had 15.4 GB resident. That is above
the function's `memory=16384` request, which Modal treats as a reservation rather than a
limit. Host memory was not recorded with the earlier settings, so this run cannot say
whether `auto` changed it. The resident figure may include CUDA host mappings.

**Checks through the real backend and on the endpoint:**

- **Long answer:** an essay request produced 2,897 output tokens in one model call and
  finished with `stop` (no `length` or `abort`), ending on a complete sentence.
- **Cancellation:** closing the backend stream at the first text stopped vLLM after
  3 generated tokens, with nothing left running.
- **Tool calling:** vLLM parsed `load_skill(name="worked-example")` with
  `tool_choice="required"`. Two backend requests completed a `load_skill` round trip (an
  explicit "load your worked-example skill" prompt, and the image request); other
  worked-solution prompts were answered directly, as before.
- **Images, PDF citations, streaming:** "Red"; the citation linked `[biology-notes.pdf,
  p. 1]`; 58 text chunks.
- **Authentication and route guard:** 401 without a proxy token or with an invalid one.
  `/health` and `/v1/models` returned 200, and all 15 dev and non-tutor routes returned
  404 with a valid proxy token. Inside the container, `/is_sleeping` returned 404 without
  the internal token and 200 with it.

Metrics for these checks were read inside the container. `modal container exec` cuts its
output off at 8 KiB, so the scripts return only the counters they need. A first attempt
that read the full `/metrics` page gave inconsistent counts and was discarded.

**GPU time versus waiting.** Modal listed each container 1.5–11.8 s after the request
(polled about every 3 s); before that, no GPU was allocated. From then until the container
stopped, it held a GPU, including the 60 s idle window before scale-down. Estimated from
trial start times, which have about ±10 s uncertainty each: combined-1 about 290 s,
combined-2 about 130 s, combined-3 (L4) about 380 s, and combined-4 about 820 s. The last
one ran all the backend checks until the app was stopped. In total, about 27 minutes of
GPU time (about 21 on L40S and 6 on L4) over a 28.6-minute session. These are estimates,
not Modal billing data.

### Decision

The snapshot server, with the route guard, replaces the plain deployment: the team decided
one deployment was enough. A restored cold start removes about 145 s on an L40S. The
persistent Triton cache stays, because snapshot creation still runs the full startup. If
GPU snapshots misbehave, the plain deployment can be restored from the Git history.

## Method

### Path and clocks

```text
benchmark client --POST /api/chat--> backend (Docker) --> timing relay (host) --> Modal --> vLLM
```

- The **client and relay** run in one local process and share one monotonic clock. The relay
  only forwards requests and timestamps stream events. It does not save headers, bodies,
  credentials, or the endpoint host.
- The **container** prints JSON `[cold-start]` events with wall and monotonic times, the
  Modal task ID, region, and image ID. It prefixes every vLLM log line with its offset since
  vLLM started (`[vllm +12.345s]`). These lines are read back with `modal app logs`.
- **Container times** are placed on the client's timeline by wall clock. The local clock's
  offset is measured with one SNTP query per trial. Modal's container clock is assumed to be
  synchronized (not verifiable).

### What counts as "first answer token"

The stop point is the first `0:` data-stream line whose decoded text is nonempty. These do
not count: response headers (sent after about 17 ms), keepalives, empty text, `b:`/`c:`/`a:`
tool events, `f:`/`e:`/`d:` step and finish events, and `3:` errors. Upstream, the relay
separates vLLM's role-only first chunk (`delta.role`, empty content) from the first
`delta.content`, `delta.tool_calls`, and `[DONE]`. It also flags any trial whose model
called a tool before answering. No trial did.

### Stages

The stages are consecutive and sum to the time to first token. "Basis" is how each was
measured.

| Stage | Ends at | Basis |
|---|---|---|
| `local_backend` | relay receives the backend's upstream request | same clock |
| `provisioning` | container process start (`/proc`, 10 ms ticks) | cross-host wall clock |
| `runtime_init` | `serve()` entered (Modal runtime, Volume mount, module import) | container |
| `cache_scan` | cache-state check done (instrumentation overhead) | container |
| `tokenizer_sync` | `snapshot_download` of tokenizer/config returns | container |
| `vllm_import` | vLLM prints its arguments (torch/vLLM imported) | container |
| `engine_spawn` | model load starts (end minus vLLM's reported duration) | container |
| `model_load` | vLLM reports "Model loading took …" | container + vLLM |
| `engine_init` | vLLM reports "init engine (profile, KV cache, warmup)" | container + vLLM |
| `api_startup` | `/health` returns 200 (polled every 0.1 s) | container |
| `first_token` | first-token count appears in `/metrics` (polled every 0.1 s) | container |
| `return_path` | relay receives the first `delta.content` | cross-host wall clock |
| `backend_stream` | client receives the first nonempty `0:` text | same clock |

Snapshot runs use the same start and end stages, with these in between:

| Snapshot stage | Ends at | Basis |
|---|---|---|
| `warmup` (creation only) | two warmup requests done, after `api_startup` | container |
| `sleep` (creation only) | prefix cache reset and vLLM asleep | container |
| `snapshot` (creation only) | `snap=False` hook starts: Modal captured the snapshot and restored in place | container |
| `provisioning_restore` (restore only, replaces everything from `provisioning` to `snapshot`) | `snap=False` hook starts: routing, GPU scheduling, snapshot load | cross-host wall clock |
| `wake_up` | vLLM `/wake_up` returns | container |

After a restore, container events are placed relative to the `restored` event's own wall
time; the monotonic clock is not assumed to continue across the restore. Each event also
records the container ID from Modal's log prefix. After a restore, the process environment
could still describe the snapshot's source container, so the prefix is the reliable ID.

### Overlaps and uncertainty

- **`provisioning`** combines Modal routing, GPU scheduling, and container creation. These
  can't be separated from outside Modal. Its end point depends on the cross-host clock offset
  (SNTP round trip 20–30 ms; the container clock error is unknown).
- **`return_path`** is about 0 s and sometimes slightly negative (−0.1 to −0.3 s). It also
  depends on the clock offset, plus up to 0.1 s of metrics-polling delay. Errors in the two
  cross-host stages cancel out in the total, which uses only the same-clock measurements.
- **`engine_spawn`** bundles config resolution (including Hugging Face metadata requests),
  the engine-core process start and its imports, and CUDA initialization. vLLM gives no finer
  markers.
- **`model_load`** covers GGUF parsing, Volume reads, and copying to the GPU together. The
  diagnostic above attributes most of it to GGUF parsing.
- **API-server work overlaps the engine.** vLLM's API-server process prepares the multimodal
  processor (about 17 s) while the engine process loads. This overlap does not add to the
  total.
- **The request waits during startup.** It is sent before the container exists. After 150 s,
  Modal answers with a `303` redirect. The OpenAI client follows it through the relay, and
  the wait continues. This is normal Modal web-endpoint behavior and happened in every trial.
- **Engine-reported time to first token** (`ttft_seconds` in the JSON) is vLLM's own
  measurement from when the request reaches the engine to its first token.

### Cold-start checks recorded per trial

- The app had zero running containers before the request.
- The trial's Modal task ID differs from every earlier trial.
- The container process started after the request was sent. For a snapshot restore, the
  restore began after the request, and a different container made the snapshot.
- The GPU model comes from `nvidia-smi` inside the container.
- Cache state is recorded separately from coldness. It covers GGUF, mmproj, and tokenizer
  presence and size; Triton cache file counts before startup and at the first token; and
  network bytes received until vLLM was ready.

## Reproduce

Prerequisites: Docker Desktop running, `.env` pointing `LLM_BASE_URL` at the Modal endpoint
with a proxy token in `LLM_API_KEY` (see [modal-serving.md](modal-serving.md)), and the
deployment virtual environment.

```powershell
.\deployment\.venv\Scripts\python.exe -X utf8 -m modal deploy deployment/modal_qwen.py
docker compose -f compose.yaml -f deployment/cold-start-benchmark.compose.yaml up -d --build backend
.\deployment\.venv\Scripts\python.exe -X utf8 deployment/cold_start_benchmark.py run --label snapshot --trials 5
.\deployment\.venv\Scripts\python.exe -X utf8 deployment/cold_start_benchmark.py report docs/cold-start-results/*.json
```

The compose override points the backend at the relay
(`http://host.docker.internal:8787/v1`). The model name and key still come from `.env`.
Before each trial, the runner waits until Modal reports zero containers for the app. That
takes about 2 minutes after the previous answer: `scaledown_window=60` plus shutdown. Each
trial takes about 6 minutes in total. The prompt is fixed:
"What does the derivative of a function tell us? Answer in two sentences."

To measure a run with an empty kernel cache again, delete the cache directory first. Model
weights are unaffected.

```powershell
.\deployment\.venv\Scripts\python.exe -X utf8 -m modal volume rm -r socratic-tutor-model-cache /triton-cache
```

Each trial records whether it created or restored a snapshot. The first few trials after a
deploy create snapshots. `--first-trial` continues the numbering across separate runs, and
`--upstream` measures a different endpoint than `.env` without saving its URL.

Afterwards, stop everything with `modal app stop --yes socratic-tutor-qwen` and
`docker compose stop`. To use the tutor normally again, start the backend without the
override file.
