"""Load a small language model and turn prompts into predicted SQL.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


def load_slm(model_name: str, device: str = "cuda") -> tuple[Any, Any]:
    """Load a causal language model and tokenizer."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )

    model.eval()
    return model, tokenizer


def make_generator(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Build a ``prompt -> predicted SQL`` function bound to a loaded model.

    Args:
        model: A loaded causal-LM.
        tokenizer: Its tokenizer.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature; ``0.0`` means greedy (deterministic).

    Returns:
        A callable mapping a prompt string to a cleaned SQL string.
    """
    import torch

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Keep only the newly generated tokens, not the echoed prompt.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return extract_sql(raw)

    return generate


def extract_sql(raw: str) -> str:
    """Clean a model's raw output down to a single SQL statement.

    Strips markdown code fences and leading ``SQL:`` cues, then returns the
    first statement.

    Args:
        raw: The model's decoded text.

    Returns:
        The cleaned SQL string.
    """
    text = raw.strip()

    # Pull SQL out of a ```...``` code block if present.
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Drop a leading "SQL:" label if the model echoed the cue.
    text = re.sub(r"^\s*SQL:\s*", "", text, flags=re.IGNORECASE)

    # Take the first statement (up to the first semicolon), normalise whitespace.
    text = text.split(";")[0].strip()
    return " ".join(text.split())
