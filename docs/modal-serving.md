\# Qwen3.5-9B serving on Modal



\## Status

Deployment planning for issue #17. Not yet deployed or tested.



\## Goal

Serve Unsloth's Q4 Qwen3.5-9B using vLLM on Modal and

verify streaming through the existing tutor backend.



\## Implementation checklist

\- \[ ] Confirm the exact quantized checkpoint and vLLM compatibility.

\- \[ ] Verify available credits and set a $30 usage limit.

\- \[ ] Add a reproducible Modal deployment script.

\- \[ ] Require authentication for the model endpoint.

\- \[ ] Configure the backend using its existing LLM settings.

\- \[ ] Test authentication and streaming directly and through the tutor.

\- \[ ] Document startup, shutdown, and troubleshooting.



\## References

\- https://modal.com/docs/examples/vllm\_inference

\- https://recipes.vllm.ai/Qwen/Qwen3.5-9B

