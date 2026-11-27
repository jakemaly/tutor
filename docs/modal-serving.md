# Serve Qwen3.5-9B on Modal



This deployment serves Qwen3.5-9B through vLLM, using Unsloth’s Safetensors weights (`unsloth/Qwen3.5-9B`, pinned revision). The weights in that repository are unquantized (BF16, about 19.3 GB). vLLM quantizes them to 4-bit with BitsAndBytes while it loads them (in-flight quantization), through the `vllm-bnb-plugin` that vLLM 0.32.0 requires for BitsAndBytes. Only Safetensors files are downloaded and served; the server refuses to start if weights in any other format are present. The tokenizer and chat template are the official `Qwen/Qwen3.5-9B` ones. The tutor connects to it through an authenticated OpenAI-compatible endpoint.



## Start the deployment



From the repository root in PowerShell, set up Modal once:



```powershell

python -m venv deployment/.venv

.\deployment\.venv\Scripts\python.exe -m pip install modal==1.6.3

.\deployment\.venv\Scripts\python.exe -m modal setup

```



Set a $30 workspace usage limit in Modal before deploying. The first time, or after changing the model revision, download the weights onto the `socratic-tutor-model-cache` Volume with a CPU-only job, so the first GPU cold start does not spend paid GPU time on the download (about 2 minutes for 19.3 GB):



```powershell

.\deployment\.venv\Scripts\python.exe -m modal run deployment/modal_qwen.py::download_model

```



Then deploy the server (it restores from a GPU memory snapshot, see [Cold starts](#cold-starts)):



```powershell

.\deployment\.venv\Scripts\python.exe -m modal deploy deployment/modal_qwen.py

```



## Connect the tutor



Create a proxy token in the Modal dashboard. The endpoint URL is `https://<workspace>--socratic-tutor-qwen-serve.<region>.modal.run`, the same as before the snapshot server replaced the plain function, so an existing `LLM_BASE_URL` keeps working. To print it:



```powershell

.\deployment\.venv\Scripts\python.exe -c "import modal; print(modal.Cls.from_name('socratic-tutor-qwen', 'SnapshotServer')().serve.get_web_url())"

```



Set these values in your local `.env`:



```dotenv

LLM_BASE_URL=https://YOUR-MODAL-ENDPOINT/v1

LLM_MODEL=socratic-tutor

LLM_API_KEY=YOUR_PROXY_TOKEN_ID.YOUR_PROXY_TOKEN_SECRET

```



Start Docker Desktop, then run `docker compose up -d --build`. Open http://localhost:5173 and send a question. After the GPU scales to zero, the first request usually answers in about 6–20 seconds on an L40S, or about 30 seconds on an L4. After a deploy that changes the code or configuration, the first few cold starts take about 3 minutes on an L40S, or up to about 6.5 minutes on an L4, while vLLM loads the model and Modal creates new snapshots. Any cold start can wait longer when Modal has no free GPU. Send a follow-up while the GPU is warm to check that the answer appears in chunks.



## Context size and long answers

vLLM starts with `--max-model-len auto`. It uses the model's full context length if that fits in GPU memory, and otherwise the largest length that fits. The deployment can run on an L4 or an L40S, so the size can change whenever Modal starts a new container.

The server sets no fixed output cap, and the backend sends no `max_tokens`. An answer can use all of the context that remains after the prompt, so it stops early only when the context is full. Continuing such an answer would have no room left, so the backend does not try to resume it.

## Stop the deployment



When finished testing, run:



```powershell

.\deployment\.venv\Scripts\python.exe -m modal app stop socratic-tutor-qwen

```



Run `docker compose stop` to stop the local tutor.



## Cold starts



Weights and tokenizer files load from the `socratic-tutor-model-cache` Volume; nothing is downloaded on a cold start once `download_model` has run. If the weights are missing, the first GPU cold start downloads them itself. A full startup reads the four BF16 Safetensors shards and quantizes them to 4-bit (about 18 s on an L40S, about 7.7 GiB on the GPU afterwards). Existing GGUF files from the earlier deployment stay on the Volume but are no longer read. vLLM's compiled Triton kernels are kept on the same Volume in `/triton-cache`, so only the first cold start after a vLLM or Triton change compiles them (about 60 s extra). The vLLM server is started with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` because the tutor sends a `load_skill` tool with every request.



The server starts vLLM, warms it up, and puts it to sleep before Modal saves a GPU memory snapshot of the container. Later cold starts restore that snapshot and only copy the weights back to the GPU. Redeploying with new code or configuration makes the previous snapshots obsolete, and Modal may create several compatible snapshots per GPU type (L4 and L40S). The first few cold starts after such a deploy therefore take the full startup path plus about 8 s to create a snapshot. In our tests every redeploy changed the code and was followed by new snapshots; redeploying unchanged code was not tested. GPU memory snapshots are an alpha Modal feature. If a restore fails, Modal logs `Transient snapshot error: failed to restore container from snapshot ... Will retry with no snapshots.` and starts the container normally instead (about 2 minutes on an L40S). This happened once in four cold starts with the Safetensors model (exit code 137). Just before it, the restored vLLM process crashed in PyTorch's NCCL heartbeat monitor with `DistStoreError: Bad file descriptor` from a `FileStore` lock. Why the descriptor was invalid after that restore is not yet known. As a workaround, the vLLM subprocess runs with `TORCH_NCCL_DUMP_ON_TIMEOUT=0`, which stops that periodic store poll (it gives up flight-recorder debug dumps after a collective timeout). In 7 further cold starts with the workaround (2 snapshot creations on L40S and L4, 5 restores on L40S), no restore failed. That shows no recurrence in those trials, not that restores cannot fail; see [Modal cold-start benchmark](cold-start-benchmark.md). vLLM's sleep and wake routes need `VLLM_SERVER_DEV_MODE=1`, which also adds vLLM's other administrative routes. A route guard ([deployment/vllm_route_guard.py](../deployment/vllm_route_guard.py)) runs inside vLLM, so proxy-token holders can reach only `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`. Every other route, including `/metrics`, `/sleep`, and `/wake_up`, answers 404 unless the request carries a token that the container generates at startup and uses only for its own sleep and wake calls.



### Prepare snapshots after deploying



New snapshots after a deploy with new code or configuration are expected. The first request after such a deploy runs the full startup and creates a snapshot: about 3 minutes on an L40S, longer on an L4. To get this done before people start testing, send one authenticated request right after deploying. Use your `LLM_API_KEY` value and the endpoint URL from `LLM_BASE_URL`:



```powershell

$key = "YOUR_PROXY_TOKEN_ID.YOUR_PROXY_TOKEN_SECRET"

curl.exe -sS -L --max-time 900 -H "Authorization: Bearer $key" https://YOUR-MODAL-ENDPOINT/v1/models

```



It returns the model list (`socratic-tutor`) once the server is ready. Modal redirects requests that wait longer than 150 seconds, and `-L` follows the redirect. Sending one message in the tutor has the same effect.



To confirm that the snapshot was created, check the app logs:



```powershell

.\deployment\.venv\Scripts\python.exe -m modal app logs socratic-tutor-qwen --since 15m --tail 2000 | Select-String -SimpleMatch 'Creating GPU memory snapshot', '"event": "sleeping"', '"event": "awake"'

```



`Creating GPU memory snapshot for Function.` followed by the `sleeping` and `awake` events means the snapshot was saved and the server woke up. After the container scales down (60 seconds idle), later cold starts on the same kind of worker log `Restoring Function from memory snapshot.` instead.



This prepares only the worker that served the request: one GPU type (L40S or L4) from one cloud provider. Modal keeps separate snapshots per GPU type and may create more for other workers, so a later cold start on a different worker can still take the full startup. The request costs one full startup of GPU time plus the 60-second idle window. `min_containers` stays at 0 and the scale-down window is unchanged, so the container then shuts down as usual.



Each container logs `[cold-start]` timing events. See [Modal cold-start benchmark](cold-start-benchmark.md) for the stage breakdown, the measured results, and how to repeat the measurement. The snapshot-restored server was checked through the tutor backend for streaming, PDF citations, image input, cancellation, tool-call parsing, and proxy authentication.



## Verification



Verified the authenticated model endpoint, streamed a response directly from vLLM, and observed text streaming in the tutor browser. The deployment disables thinking by default and does not cap generated output.
