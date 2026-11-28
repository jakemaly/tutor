# Model startup status

The chat shows service-level Modal startup events while waiting for a reply. The backend
polls recent logs and current service statistics for the deployed `socratic-tutor-qwen`
app. It returns only allowed milestones, GPU name and memory, provider, region, model-weight
source, quantization, and current service-level queue, running-input, and container counts.
Model shard progress is shown only when vLLM emits it. These events describe the service and are not a
per-request container assignment. If more than one container appears, the status falls back
to a neutral waiting message. If the backend cannot read the logs, chat continues normally.

Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the backend environment to enable the feed.
Set `MODAL_STATUS_APP` only if the deployment uses a different Modal app name. Never expose
these credentials to frontend code. Without both credentials the endpoint returns unavailable.
The endpoint does not create or deploy a Modal app and does not start a GPU container.

Queue and container counts describe the whole Modal function, not a specific user request.
Startup events come from the existing snapshot server logs and may be delayed or absent; the
frontend never fabricates allocation details or a progress percentage when Modal has not
reported them.
