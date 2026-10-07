"""This file evaluates a trained LoRA adapter through the shared Spider pipeline.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from peft import PeftModel

from common.config import load_config
from common.data_loader import Example, load_examples
from common.evaluation import execution_accuracy
from common.model import load_slm, make_generator
from common.runner import EvalRecord, run_evaluation


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint JSON: {path}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Checkpoint must contain a JSON list: {path}")

    return data


def validate_existing(
    existing: list[dict[str, Any]],
    examples: list[Example],
) -> None:
    if len(existing) > len(examples):
        raise ValueError(
            f"Checkpoint has {len(existing)} records, but this run requests "
            f"only {len(examples)} examples."
        )

    for index, (saved, example) in enumerate(zip(existing, examples)):
        expected = (example.db_id, example.question, example.gold_sql)
        actual = (
            saved.get("db_id"),
            saved.get("question"),
            saved.get("gold_sql"),
        )
        if actual != expected:
            raise ValueError(
                f"Checkpoint does not match the dev set at index {index}."
            )


def save_atomic(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
        file.flush()
        os.fsync(file.fileno())

    temporary.replace(path)


def print_score(records: list[dict[str, Any]]) -> None:
    correct = [bool(record["correct"]) for record in records]
    score = execution_accuracy(correct)
    print(
        f"\nExecution Accuracy (LoRA): {score:.3f} "
        f"({sum(correct)}/{len(correct)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Spider LoRA adapter.")
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Directory containing adapter_config.json and adapter weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Persistent JSON results path.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if args.save_every <= 0:
        parser.error("--save_every must be greater than zero.")

    cfg = load_config()
    examples = load_examples(cfg["paths"]["dev_json"])
    if args.limit is not None:
        examples = examples[: args.limit]

    existing = load_existing(args.output) if args.resume else []
    if args.resume:
        validate_existing(existing, examples)

    remaining = examples[len(existing) :]

    if not remaining:
        print("All requested examples are already complete.")
        print_score(existing)
        return

    print(f"Loading base model {cfg['models']['slm']} ...")
    base_model, tokenizer = load_slm(
        cfg["models"]["slm"],
        device=cfg["device"],
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(args.adapter),
    )
    model.eval()

    generate_fn = make_generator(
        model,
        tokenizer,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
    )

    new_records: list[EvalRecord] = []
    existing_correct = sum(bool(record["correct"]) for record in existing)
    start_time = time.perf_counter()

    print(
        f"Evaluating {len(remaining)} examples "
        f"(checkpointing every {args.save_every}) ..."
    )

    try:
        for chunk_start in range(0, len(remaining), args.save_every):
            chunk = remaining[chunk_start : chunk_start + args.save_every]
            chunk_records = run_evaluation(
                chunk,
                generate_fn,
                tables_json=cfg["paths"]["tables_json"],
                database_dir=cfg["paths"]["database_dir"],
                schema_style=cfg["prompt"]["schema_style"],
                include_keys=cfg["prompt"]["include_keys"],
            )
            new_records.extend(chunk_records)

            combined = existing + [asdict(record) for record in new_records]
            save_atomic(combined, args.output)

            done = len(combined)
            elapsed = time.perf_counter() - start_time
            rate = len(new_records) / elapsed if elapsed else 0.0
            remaining_n = len(examples) - done
            eta_min = remaining_n / rate / 60 if rate else float("nan")
            correct_so_far = (
                existing_correct
                + sum(record.correct for record in new_records)
            )

            print(
                f"[{done}/{len(examples)}] "
                f"running EX={correct_so_far / done:.3f} "
                f"({rate:.2f} ex/s, ~{eta_min:.1f} min remaining)"
            )

    except KeyboardInterrupt:
        print(
            f"\nInterrupted. The latest completed checkpoint is at "
            f"{args.output}."
        )
        return

    final_records = existing + [asdict(record) for record in new_records]
    print_score(final_records)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
