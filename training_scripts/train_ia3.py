"""This file trains an IA3 adapter for Qwen on the Spider training split.

The training examples use the same prompt construction as the LoRA experiment:
schema + question are the user message, and the gold SQL is the assistant reply.

"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import IA3Config, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from common.config import load_config
from common.data_loader import Example, load_examples, load_schema
from common.prompt_builder import build_prompt
from common.schema_serializer import serialize


def build_training_dataset(
    examples: list[Example],
    *,
    tables_json: str | Path,
    schema_style: str,
    include_keys: bool,
) -> Dataset:
    """Build a conversational prompt-completion dataset for TRL."""
    rows: list[dict[str, Any]] = []

    for example in examples:
        schema = load_schema(example.db_id, tables_json)
        schema_text = serialize(
            schema,
            style=schema_style,
            include_keys=include_keys,
        )
        prompt = build_prompt(schema_text, example.question)

        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [
                    {"role": "assistant", "content": example.gold_sql}
                ],
            }
        )

    return Dataset.from_list(rows)


def inspect_lengths(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> None:
    """Report sequence lengths so truncation is visible rather than silent."""
    lengths: list[int] = []

    for row in dataset:
        messages = row["prompt"] + row["completion"]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(len(token_ids))

    lengths.sort()
    n = len(lengths)
    p95 = lengths[min(n - 1, int(0.95 * n))]
    over_limit = sum(length > max_length for length in lengths)

    print(
        f"Token lengths: median={lengths[n // 2]}, "
        f"p95={p95}, max={lengths[-1]}"
    )
    print(
        f"Sequences over max_length={max_length}: "
        f"{over_limit}/{n} ({over_limit / n:.1%})"
    )

    if over_limit:
        print(
            "WARNING: over-length examples will be truncated. "
            "Consider rerunning with a larger --max_length."
        )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="IA3 SFT on Spider.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Persistent directory for checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N training examples (for smoke tests).",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest Trainer checkpoint in output_dir.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select a Colab GPU runtime.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero.")
    if args.gradient_accumulation <= 0:
        parser.error("--gradient_accumulation must be greater than zero.")
    if args.max_length <= 0:
        parser.error("--max_length must be greater than zero.")
    if args.save_steps <= 0:
        parser.error("--save_steps must be greater than zero.")

    cfg = load_config()
    seed = cfg.seed
    set_seed(seed)

    model_name = cfg["models"]["slm"]
    train_path = cfg["paths"]["train_json"]
    tables_json = cfg["paths"]["tables_json"]
    schema_style = cfg["prompt"]["schema_style"]
    include_keys = cfg["prompt"]["include_keys"]

    examples = load_examples(train_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    print(f"Training examples: {len(examples)}")
    print(f"Base model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = build_training_dataset(
        examples,
        tables_json=tables_json,
        schema_style=schema_style,
        include_keys=include_keys,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    inspect_lengths(dataset, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"],
        init_ia3_weights=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        max_length=args.max_length,
        completion_only_loss=True,
        eos_token="<|im_end|>",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=ia3_config,
    )

    trainer.model.print_trainable_parameters()

    trainable_params = sum(
        p.numel()
        for p in trainer.model.parameters()
        if p.requires_grad
    )
    total_params = sum(
        p.numel()
        for p in trainer.model.parameters()
    )
    trainable_percent = 100.0 * trainable_params / total_params

    checkpoint: str | None = None
    if args.resume:
        checkpoint = get_last_checkpoint(str(args.output_dir))
        if checkpoint is None:
            raise FileNotFoundError(
                f"--resume was requested, but no checkpoint exists in "
                f"{args.output_dir}"
            )
        print(f"Resuming from: {checkpoint}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train(
        resume_from_checkpoint=checkpoint
    )
    trainer.save_state()

    torch.cuda.synchronize()

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    train_runtime_s = float(
        train_result.metrics.get("train_runtime", 0.0)
    )
    num_tokens = float(
        train_result.metrics.get("num_tokens", 0.0)
    )

    if num_tokens == 0.0:
        for log in reversed(trainer.state.log_history):
            if "num_tokens" in log:
                num_tokens = float(log["num_tokens"])
                break

    if train_runtime_s > 0.0 and num_tokens > 0.0:
        tokens_per_second = num_tokens / train_runtime_s
    else:
        tokens_per_second = None

    adapter_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    metadata = {
        "base_model": model_name,
        "train_path": str(train_path),
        "training_examples": len(examples),
        "seed": seed,
        "schema_style": schema_style,
        "include_keys": include_keys,
        "ia3": {
            "target_modules": ["k_proj", "v_proj", "down_proj"],
            "feedforward_modules": ["down_proj"],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation
            ),
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "save_steps": args.save_steps,
            "train_loss": train_result.training_loss,
        },
        "efficiency": {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": trainable_percent,
            "peak_gpu_memory_allocated_gb": peak_allocated_gb,
            "peak_gpu_memory_reserved_gb": peak_reserved_gb,
            "training_runtime_seconds": train_runtime_s,
            "training_tokens": num_tokens,
            "training_tokens_per_second": tokens_per_second,
        },
        "versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "trl": package_version("trl"),
            "peft": package_version("peft"),
            "datasets": package_version("datasets"),
        },
    }

    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nFinal adapter saved to: {adapter_dir}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({trainable_percent:.4f}%)"
    )
    print(
        f"Peak GPU memory allocated: "
        f"{peak_allocated_gb:.2f} GB"
    )
    print(
        f"Peak GPU memory reserved:  "
        f"{peak_reserved_gb:.2f} GB"
    )

    if tokens_per_second is not None:
        print(
            f"Training throughput: "
            f"{tokens_per_second:.2f} tokens/s"
        )
    else:
        print("Training throughput: unavailable")


if __name__ == "__main__":
    main()
