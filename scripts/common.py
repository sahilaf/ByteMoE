from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924"
DEFAULT_PROMPTS = [
    "The capital city of Bangladesh is",
    "Write a Python function that reverses a list.",
    "The proof begins by assuming that",
    "In a mixture-of-experts model, routing selects",
    "Once upon a time, a small robot discovered",
    "Translate to French: good morning",
    "The following SQL query returns",
    "A healthy machine learning evaluation should",
]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the feasibility benchmarks.")
    return torch.device("cuda")


def load_model(model_id: str, dtype: str):
    device = require_cuda()
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.eval().to(device)
    return model, tokenizer, device


def prompt_batches(tokenizer, device: torch.device, prompts: list[str], batch_size: int, max_length: int):
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        yield {key: value.to(device) for key, value in encoded.items()}


def load_prompts(path: str | None, copies: int) -> list[str]:
    prompts = DEFAULT_PROMPTS if path is None else [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Prompt file contains no non-empty lines.")
    return (prompts * copies)[: len(prompts) * copies]


def write_json(path: str, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n")
