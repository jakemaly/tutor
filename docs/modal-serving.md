# Serve Qwen3.5-9B on Modal



This deployment serves Unsloth’s Qwen3.5-9B Q4_K_M checkpoint through vLLM. The tutor connects to it through an authenticated OpenAI-compatible endpoint.



## Start the deployment



From the repository root in PowerShell, set up Modal once:



```powershell

python -m venv deployment/.venv

.\deployment\.venv\Scripts\python.exe -m pip install modal==1.6.3

.\deployment\.venv\Scripts\python.exe -m modal setup

```



Set a $30 workspace usage limit in Modal before deploying. Then deploy the snapshot-restored server (recommended, see [Cold starts](#cold-starts)):



```powershell

.\deployment\.venv\Scripts\python.exe -m modal deploy deployment/modal_qwen_snapshot.py

```



To deploy the plain server instead, which is slower to start but has no GPU snapshots or vLLM dev-mode routes, run `.\deployment\.venv\Scripts\python.exe -m modal deploy deployment/modal_qwen.py`.



## Connect the tutor



Create a proxy token in the Modal dashboard. Use the endpoint URL of the server you deployed. The plain server's URL is printed by the deploy command. For the snapshot server, print it with:



```powershell

.\deployment\.venv\Scripts\python.exe -c "import modal; print(modal.Cls.from_name('socratic-tutor-qwen-snapshot', 'SnapshotServer')().serve.get_web_url())"

```



Set these values in your local `.env`:



```dotenv

LLM_BASE_URL=https://YOUR-MODAL-ENDPOINT/v1

LLM_MODEL=socratic-tutor

LLM_API_KEY=YOUR_PROXY_TOKEN_ID.YOUR_PROXY_TOKEN_SECRET

```



Start Docker Desktop, then run `docker compose up -d --build`. Open http://localhost:5173 and send a question. After the GPU scales to zero, the first request on the snapshot server usually answers in about 6–20 seconds on an L40S, or about 30 seconds on an L4. The first few cold starts after each deploy, and every cold start on the plain server, take about 2.5–3.5 minutes on an L40S, or up to about 6.5 minutes on an L4, while vLLM loads the model. Any cold start can wait longer when Modal has no free GPU. Send a follow-up while the GPU is warm to check that the answer appears in chunks.



## Cold starts



Weights and tokenizer files load from the `socratic-tutor-model-cache` Volume; nothing is downloaded on a cold start. vLLM's compiled Triton kernels are kept on the same Volume in `/triton-cache`, so only the first cold start after a vLLM or Triton change compiles them (about 60 s extra). The vLLM server is started with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` because the tutor sends a `load_skill` tool with every request.



The snapshot server starts vLLM, warms it up, and puts it to sleep before Modal saves a GPU memory snapshot of the container. Later cold starts restore that snapshot and only copy the weights back to the GPU. Modal creates a few snapshots per GPU worker type after each deploy, so the first few cold starts are about 40 s slower than the plain server. GPU memory snapshots are an alpha Modal feature. vLLM's sleep and wake routes need `VLLM_SERVER_DEV_MODE=1`, which also adds vLLM's other administrative routes. A route guard ([deployment/vllm_route_guard.py](../deployment/vllm_route_guard.py)) runs inside vLLM, so proxy-token holders can reach only `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`. Every other route, including `/metrics`, `/sleep`, and `/wake_up`, answers 404 unless the request carries a token that the container generates at startup and uses only for its own sleep and wake calls.



Each container logs `[cold-start]` timing events. See [Modal cold-start benchmark](cold-start-benchmark.md) for the stage breakdown, the measured results for both servers, and how to repeat the measurement.



## Stop the deployment



When finished testing, run:



```powershell

.\deployment\.venv\Scripts\python.exe -m modal app stop socratic-tutor-qwen-snapshot

.\deployment\.venv\Scripts\python.exe -m modal app stop socratic-tutor-qwen

```



Run `docker compose stop` to stop the local tutor.



## Verification



Verified the authenticated model endpoint, streamed a response directly from vLLM, and observed text streaming in the tutor browser. The deployment disables thinking by default and caps generated output at 512 tokens. The snapshot server was checked through the tutor backend for streaming, PDF citations, image input, cancellation, tool-call parsing, and proxy authentication.
