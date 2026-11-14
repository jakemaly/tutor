# Serve Qwen3.5-9B on Modal



This deployment serves Unsloth’s Qwen3.5-9B Q4_K_M checkpoint through vLLM. The tutor connects to it through an authenticated OpenAI-compatible endpoint.



## Start the deployment



From the repository root in PowerShell, set up Modal once:



```powershell

python -m venv deployment/.venv

.\deployment\.venv\Scripts\python.exe -m pip install modal==1.6.3

.\deployment\.venv\Scripts\python.exe -m modal setup

```



Set a $30 workspace usage limit in Modal before deploying. Then run:



```powershell

.\deployment\.venv\Scripts\python.exe -m modal deploy deployment/modal_qwen.py

```



## Connect the tutor



Create a proxy token in the Modal dashboard. Use the endpoint URL printed by the deploy command and set these values in your local `.env`:



```dotenv

LLM_BASE_URL=https://YOUR-MODAL-ENDPOINT/v1

LLM_MODEL=socratic-tutor

LLM_API_KEY=YOUR_PROXY_TOKEN_ID.YOUR_PROXY_TOKEN_SECRET

```



Start Docker Desktop, then run `docker compose up -d --build`. Open http://localhost:5173 and send a question. The first request can take several minutes while Modal assigns a GPU and loads the model. Send a follow-up while the GPU is warm to check that the answer appears in chunks.



## Context size and long answers

vLLM starts with `--max-model-len auto`. It uses the model's full context length if that fits in GPU memory, and otherwise the largest length that fits. The deployment can run on an L4 or an L40S, so the size can change whenever Modal starts a new container.

The server sets no fixed output cap. A request without `max_tokens` can use the context that remains after the prompt.

The backend reads the current size from the `max_model_len` field of the authenticated `/v1/models` response. It does this only after an answer stops with `finish_reason` set to `length`, using the same URL and key as chat. If context remains, it continues the partial answer in the same browser response. It stops on a natural finish or when the context is full.

Endpoints that do not report `max_model_len` for `LLM_MODEL` still work. Their length-limited answers end without continuation, and the backend logs a warning instead of guessing a size.

## Stop the deployment



When finished testing, run:



```powershell

.\deployment\.venv\Scripts\python.exe -m modal app stop socratic-tutor-qwen

```



Run `docker compose stop` to stop the local tutor.



## Verification



Verified the authenticated model endpoint, streamed a response directly from vLLM, and observed text streaming in the tutor browser. The deployment disables thinking by default and does not cap generated output.

