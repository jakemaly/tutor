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



## Stop the deployment



When finished testing, run:



```powershell

.\deployment\.venv\Scripts\python.exe -m modal app stop socratic-tutor-qwen

```



Run `docker compose stop` to stop the local tutor.



## Verification



Verified the authenticated model endpoint, streamed a response directly from vLLM, and observed text streaming in the tutor browser. The deployment disables thinking by default and caps generated output at 512 tokens.



## Serve the trained tutor (issue #46)



`deployment/modal_qwen_trained.py` serves the RL-trained checkpoint on Modal, for trying it in the app. This is optional: the week 2 comparison runs in molab without Modal. It is a separate Modal app (`socratic-tutor-qwen-trained`), so the deployment above and the app's `LLM_BASE_URL` stay as they are. One vLLM server on an L40S offers two model names:



- `socratic-tutor-base`: Qwen3.5-9B in bf16, the model training started from.
- `socratic-tutor-trained`: the same model with the trained LoRA adapter (`final/adapter` from `socratictutor/tutor-rl-qwen35-9b-bigmath` on Hugging Face).



Both names share the same weights, server and settings, so any difference between them comes from the adapter. The server renames the adapter's weights to the layout vLLM expects when it starts. The run repo is private, so create a Modal secret named `huggingface` with a Hugging Face read token once (Modal dashboard → Secrets → Hugging Face, key `HF_TOKEN`). Then deploy:



```powershell
.\deployment\.venv\Scripts\python.exe -m modal deploy deployment/modal_qwen_trained.py
```



Check that both names are listed, using the endpoint URL from the deploy command and a proxy token. The first request waits a few minutes for a GPU:



```powershell
curl.exe -H "Authorization: Bearer YOUR_PROXY_TOKEN_ID.YOUR_PROXY_TOKEN_SECRET" https://YOUR-TRAINED-ENDPOINT/v1/models
```



To try the trained tutor in the app locally, set `LLM_MODEL=socratic-tutor-trained` and this endpoint's `LLM_BASE_URL` in your own `.env` only. The committed defaults do not change.



### Compare base and trained



The comparison itself runs in the molab notebook (`training/tutor_rl_molab.py`, section 6), which serves the same two names with vLLM on molab's GPU, so it needs no Modal account. To compare against this Modal endpoint instead, run `training/evaluate.py --policy base=TUTOR_BASE --policy trained=TUTOR_TRAINED` with the student and judge running, and set `TUTOR_BASE_BASE_URL`, `TUTOR_BASE_MODEL=socratic-tutor-base`, `TUTOR_BASE_API_KEY` and the matching `TUTOR_TRAINED_*` variables.



Stop the app when done:



```powershell
.\deployment\.venv\Scripts\python.exe -m modal app stop socratic-tutor-qwen-trained
```
