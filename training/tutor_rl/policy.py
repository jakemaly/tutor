"""The trainable tutor: Qwen3.5 (text-only) with a bf16 LoRA adapter, as a batched `ChatModel`.

Needs torch, transformers>=5 and peft (GPU machine). Every generation is recorded by
(prompt token ids, text) so the trainer can score exactly the tokens that were sampled.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tutor_rl.episode import Message


def load_tutor(
    tutor_cfg: dict,
    *,
    adapter_path: str | Path | None = None,
    trainable: bool = True,
    device: str = "cuda",
):
    """Load the base model in bf16 and attach a new (or saved) LoRA adapter."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, revision = tutor_cfg["model"], tutor_cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    # Qwen3.5 checkpoints are vision-language; AutoModelForCausalLM loads the text model only.
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, device_map={"": device}
    )
    model.config.use_cache = True
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable)
    else:
        lora = LoraConfig(
            r=tutor_cfg.get("lora_rank", 32),
            lora_alpha=tutor_cfg.get("lora_alpha", 32),
            lora_dropout=tutor_cfg.get("lora_dropout", 0.0),
            target_modules=tutor_cfg["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    return HFTutorPolicy(model, tokenizer, tutor_cfg)


@dataclass
class HFTutorPolicy:
    model: object  # peft.PeftModel
    tokenizer: object
    cfg: dict
    use_adapter: bool = True  # False = the base model (for evaluation and the KL reference)
    seed: int | None = None  # fixed sampling seeds for evaluation
    records: dict = field(default_factory=dict)
    _calls: int = 0

    def __post_init__(self) -> None:
        tok = self.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        stops |= set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
        self.stop_ids = sorted(i for i in stops if isinstance(i, int) and i >= 0)

    # ----- prompts -----------------------------------------------------------------------
    def prompt_ids(self, messages: Sequence[Message]) -> list[int]:
        """Exactly the chat template the app's vLLM server applies (thinking disabled).

        Qwen3.5's template refuses a prompt with no user message ("No user query found"), which
        is the tutor's first turn when it opens the dialogue. Then a placeholder user message is
        rendered and its block cut out, leaving system + generation prompt.
        """
        chat = [{"role": m.role, "content": m.content} for m in messages]
        placeholder = "\x00tutor-opens\x00"
        no_user = not any(m["role"] == "user" for m in chat)
        if no_user:
            chat.append({"role": "user", "content": placeholder})
        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.get("enable_thinking", False),
        )
        if no_user:
            block = f"<|im_start|>user\n{placeholder}<|im_end|>\n"
            assert block in text, "unexpected chat template layout"
            text = text.replace(block, "")
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def was_truncated(self, messages: Sequence[Message], text: str) -> bool:
        """True if this generation hit max_tokens_per_turn without a stop token."""
        ids = self.records.get((tuple(self.prompt_ids(messages)), text))
        return ids is not None and (not ids or ids[-1] not in self.stop_ids)

    def completion_ids(self, prompt_ids: Sequence[int], text: str) -> list[int]:
        """The sampled tokens for `text`; re-tokenized (+ stop token) if it was not recorded."""
        recorded = self.records.get((tuple(prompt_ids), text))
        if recorded is not None:
            return recorded
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids + [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]

    def adapter_context(self):
        return nullcontext() if self.use_adapter else self.model.disable_adapter()

    # ----- ChatModel ---------------------------------------------------------------------
    def generate(self, messages: Sequence[Message], *, n: int = 1) -> list[str]:
        return self.generate_batch([messages], n=n)[0]

    @torch.no_grad()
    def generate_batch(
        self, conversations: Sequence[Sequence[Message]], *, n: int = 1
    ) -> list[list[str]]:
        from transformers import GenerationConfig

        prompts = [self.prompt_ids(c) for c in conversations for _ in range(n)]
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=self.cfg.get("temperature", 1.0),
            top_p=self.cfg.get("top_p", 1.0),
            top_k=0,  # sample from the full distribution so training log-probs match
            max_new_tokens=self.cfg.get("max_tokens_per_turn", 256),
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        texts: list[str] = []
        batch_size = self.cfg.get("generation_batch_size", 16)
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, len(prompts), batch_size):
                texts += self._generate_chunk(prompts[start : start + batch_size], gen_cfg)
        finally:
            if was_training:
                self.model.train()
        return [texts[i * n : (i + 1) * n] for i in range(len(conversations))]

    def _generate_chunk(self, prompts: list[list[int]], gen_cfg) -> list[str]:
        if self.seed is not None:
            torch.manual_seed(self.seed + self._calls)
        self._calls += 1
        width = max(len(p) for p in prompts)
        pad = self.tokenizer.pad_token_id
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
        attention = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device
        )
        with self.adapter_context():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attention, generation_config=gen_cfg
            )
        texts = []
        for prompt, row in zip(prompts, out[:, width:].tolist(), strict=True):
            ids = []
            for token in row:
                ids.append(token)
                if token in self.stop_ids:
                    break
            body = ids[:-1] if ids and ids[-1] in self.stop_ids else ids
            text = self.tokenizer.decode(body, skip_special_tokens=True)
            self.records[(tuple(prompt), text)] = ids
            texts.append(text)
        return texts


def load_adapter_weights(policy: HFTutorPolicy, adapter_dir: str | Path) -> None:
    """Copy a saved adapter's weights into the already-loaded policy (same LoRA config)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    set_peft_model_state_dict(policy.model, weights)
