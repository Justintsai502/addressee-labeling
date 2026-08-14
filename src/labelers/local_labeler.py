"""Candidate labeler backed by a LOCALLY DOWNLOADED model (transcript only).

Use this when you want to download open-weight models and swap them freely to
compare — change one string (`--model <hf_id>`) and you are testing a different
model. Same prompts / parsing as the Qwen endpoint labeler; the only difference is
that the weights are loaded in-process instead of reached over HTTP.

RUN THIS ON THE SERVER (it downloads weights and needs a GPU). Two engines:

  backend="vllm"  (default, fast, batched):   pip install vllm
  backend="hf"    (universal fallback):        pip install transformers torch accelerate

Model weights download to the HuggingFace cache (~/.cache/huggingface) on first use.
Nothing here loads a model until `label_conversation` actually runs, so the module
is safe to import on a laptop with no GPU / no vllm / no torch installed.

Swap models by id, e.g.:
    --model Qwen/Qwen3-32B
    --model Qwen/Qwen3-8B
    --model meta-llama/Llama-3.1-8B-Instruct
    --model mistralai/Mistral-7B-Instruct-v0.3
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..parsing import parse_label_response
from ..prompts import CANDIDATE_SYSTEM, build_user_prompt
from ..schema import AddresseeLabel, Conversation
from .base import AddresseeLabeler


class LocalLLMLabeler(AddresseeLabeler):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-32B",
        backend: str = "vllm",              # "vllm" | "hf"
        temperature: float = 0.0,
        max_new_tokens: int = 8192,  # thinking models (Qwen3) spend a chunk of
                                      # this on a <think> trace before the answer;
                                      # 4096 was empirically not always enough for
                                      # a 40-turn window's think trace + full JSON
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,      # #GPUs for vllm
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        enable_thinking: bool = True,  # some turns can send a thinking model
                                        # into an unbounded chain of thought
                                        # that never reaches an answer even at
                                        # a generous max_new_tokens — set False
                                        # as an escape hatch when that happens
        max_turns_per_window: int = 40,
        context_turns: int = 10,
    ) -> None:
        super().__init__(max_turns_per_window, context_turns)
        self.enable_thinking = enable_thinking
        self.name = model if enable_thinking else f"{model}(no-think)"
        self.model = model
        self.backend = backend.lower()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self._engine = None       # loaded once, lazily
        self._tokenizer = None    # hf backend only
        # timing/throughput stats, for speed benchmarking across models
        self.stats = {
            "load_seconds": 0.0,
            "generate_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "n_calls": 0,
        }

    # -- load the model once (not per conversation) ---------------------------
    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        import sys as _sys
        import time as _time

        _t_load = _time.time()
        if self.backend == "vllm":
            from vllm import LLM  # lazy: keeps vllm optional

            kw = dict(
                model=self.model,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
            )
            if self.max_model_len:
                kw["max_model_len"] = self.max_model_len
            self._engine = LLM(**kw)
        elif self.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
            # device_map="auto" has been observed placing the whole model on CPU
            # even with a free GPU present (~7 tok/s instead of ~30+, with no
            # warning). Pin to the GPU explicitly when there is one; only fall
            # back to "auto" for multi-GPU sharding or genuinely CPU-only hosts.
            if torch.cuda.is_available():
                device_map = "auto" if self.tensor_parallel_size > 1 else {"": 0}
            else:
                device_map = "auto"
            self._engine = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=("auto" if self.dtype == "auto" else self.dtype),
                device_map=device_map,
                trust_remote_code=self.trust_remote_code,
            )
            dev = next(self._engine.parameters()).device
            if dev.type == "cpu" and torch.cuda.is_available():
                print(f"  WARNING: {self.model} loaded on CPU despite an available "
                      f"GPU — generation will be ~5x slower. Check accelerate/"
                      f"device_map behaviour before trusting any timing.",
                      file=_sys.stderr, flush=True)
            else:
                print(f"  [hf] model on {dev}, dtype "
                      f"{next(self._engine.parameters()).dtype}",
                      file=_sys.stderr, flush=True)
        else:
            raise ValueError(f"unknown backend {self.backend!r} (use 'vllm' or 'hf')")
        self.stats["load_seconds"] = _time.time() - _t_load
        return self._engine

    def _chat(self, messages: List[dict]) -> str:
        """Run one chat completion and return raw text."""
        engine = self._get_engine()
        if self.backend == "vllm":
            from vllm import SamplingParams

            import time as _t

            sp = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
            _t0 = _t.time()
            out = engine.chat(
                messages, sp, use_tqdm=False,
                chat_template_kwargs={"enable_thinking": self.enable_thinking},
            )
            self.stats["generate_seconds"] += _t.time() - _t0
            self.stats["n_calls"] += 1
            self.stats["input_tokens"] += len(out[0].prompt_token_ids or [])
            self.stats["output_tokens"] += len(out[0].outputs[0].token_ids or [])
            return out[0].outputs[0].text
        # hf backend
        import sys
        import time

        import torch  # lazy

        tok = self._tokenizer
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = tok(prompt, return_tensors="pt").to(engine.device)
        n_in = inputs["input_ids"].shape[1]
        # transformers' plain generate() prints nothing while it runs (unlike
        # vLLM), so a thinking model chewing through max_new_tokens looks
        # indistinguishable from hung — print before/after so it isn't.
        print(f"  [hf] generating: {n_in} input tokens, up to {self.max_new_tokens} "
              f"new (no progress bar during generation, this can take minutes)...",
              file=sys.stderr, flush=True)
        t0 = time.time()
        with torch.no_grad():
            gen = engine.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=(self.temperature or None),
                pad_token_id=tok.eos_token_id,
            )
        n_out = gen.shape[1] - n_in
        dt = time.time() - t0
        self.stats["generate_seconds"] += dt
        self.stats["n_calls"] += 1
        self.stats["input_tokens"] += n_in
        self.stats["output_tokens"] += n_out
        print(f"  [hf] done: {n_out} tokens in {dt:.1f}s ({n_out / max(dt, 0.01):.1f} tok/s)",
              file=sys.stderr, flush=True)
        return tok.decode(gen[0][n_in:], skip_special_tokens=True)

    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        target_ids = [t.turn_id for t in target]
        messages = [
            {"role": "system", "content": CANDIDATE_SYSTEM},
            {"role": "user", "content": build_user_prompt(window_text)},
        ]
        raw = self._chat(messages)
        return parse_label_response(raw, conv, target_ids, self.name)
