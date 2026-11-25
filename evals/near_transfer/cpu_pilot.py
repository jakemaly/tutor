"""Optional CPU inference pilot. Load one GGUF once and reset sampling per completion."""

from pathlib import Path

from tutor_rl.episode import Message


class CPUModel:
    def __init__(self, engine, name, max_tokens, temperature):
        self.engine = engine
        self.model = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = 0
        self.truncated = set()

    def generate(self, messages: list[Message], *, n=1):
        history = [vars(m) for m in messages]
        replies = []
        for i in range(n):
            response = self.engine.create_chat_completion(
                messages=history,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed + i,
            )
            choice = response["choices"][0]
            reply = choice["message"]["content"]
            if choice["finish_reason"] == "length":
                self.truncated.add((tuple(messages), reply))
            replies.append(reply)
        return replies

    def was_truncated(self, messages, text):
        return (tuple(messages), text) in self.truncated


def pilot_models(path: Path):
    from llama_cpp import Llama

    engine = Llama(
        model_path=str(path),
        n_ctx=4096,
        n_threads=4,
        verbose=False,
        chat_format="chatml",
    )
    return tuple(
        CPUModel(engine, path.name, cap, temp)
        for cap, temp in ((96, 1.0), (96, 0.6), (192, 0.6), (192, 0.7))
    )
