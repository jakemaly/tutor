"""`ChatModel` over any OpenAI-compatible chat endpoint (vLLM, Modal), standard library only."""

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tutor_rl.episode import Message


@dataclass
class OpenAIChatModel:
    base_url: str  # e.g. http://127.0.0.1:8101/v1
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 256
    seed: int | None = None  # base seed; request i of a batch uses seed + i
    extra_body: dict | None = None  # e.g. {"chat_template_kwargs": {"enable_thinking": False}}
    max_workers: int = 32
    timeout: float = 300.0
    retries: int = 3

    @classmethod
    def from_env(cls, prefix: str, **defaults) -> "OpenAIChatModel":
        """Read <PREFIX>_BASE_URL, <PREFIX>_MODEL and optional <PREFIX>_API_KEY."""
        try:
            base_url = os.environ[f"{prefix}_BASE_URL"]
            model = os.environ[f"{prefix}_MODEL"]
        except KeyError as e:
            raise RuntimeError(f"Set {prefix}_BASE_URL and {prefix}_MODEL") from e
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get(f"{prefix}_API_KEY", "EMPTY"),
            **defaults,
        )

    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]:
        return self._request(messages, n, self.seed)

    def generate_batch(
        self, conversations: Sequence[Sequence[Message]], *, n: int = 1
    ) -> list[list[str]]:
        seeds = [None if self.seed is None else self.seed + i for i in range(len(conversations))]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(conversations))) as pool:
            return list(pool.map(lambda c, s: self._request(c, n, s), conversations, seeds))

    def _request(self, messages: Sequence[Message], n: int, seed: int | None) -> list[str]:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": n,
            **(self.extra_body or {}),
        }
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                choices = sorted(payload["choices"], key=lambda c: c["index"])
                return [c["message"]["content"] or "" for c in choices]
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.retries:
                    detail = e.read().decode(errors="replace")[:500]
                    raise RuntimeError(f"{self.base_url} returned {e.code}: {detail}") from e
                time.sleep(2**attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def model_from_section(section: dict, *, max_tokens_key: str = "max_tokens_per_turn"):
    """Build the student or judge model from its config section and the environment."""
    return OpenAIChatModel.from_env(
        section["env_prefix"],
        temperature=section.get("temperature", 0.7),
        max_tokens=section.get(max_tokens_key, section.get("max_tokens", 256)),
    )
