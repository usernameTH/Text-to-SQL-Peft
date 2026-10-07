"""Establish the baseline: vanilla small model, zero-shot, over the dev set.

Saves results incrementally  and can resume
from a previous partial run.

"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common.config import load_config
from common.data_loader import load_examples
from common.evaluation import execution_accuracy
from common.model import load_slm, make_generator
from common.runner import EvalRecord, run_evaluation

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "slm_vanilla.json"


def _load_existing(path: Path) -> list[dict]:
    """Load previously saved records, if any (for --resume)."""
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


def _save(records: list[dict], path: Path) -> None:
    """Write records to disk, overwriting the previous checkpoint."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def main() -> None:
    """Run the vanilla SLM over the dev set, saving progress incrementally."""
    parser = argparse.ArgumentParser(description="Vanilla SLM floor.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--save_every", type=int, default=25, help="Checkpoint every N examples.")
    parser.add_argument("--resume", action="store_true", help="Continue from a previous partial run.")
    args = parser.parse_args()

    cfg = load_config()
    examples = load_examples(cfg["paths"]["dev_json"])
    if args.limit:
        examples = examples[: args.limit]

    # --- Resume support: skip examples already completed ---
    existing = _load_existing(RESULTS_FILE) if args.resume else []
    start_idx = len(existing)
    if start_idx:
        print(f"Resuming: {start_idx} examples already completed, {len(examples) - start_idx} remaining.")
    remaining = examples[start_idx:]
    if not remaining:
        print("Nothing to do -- all requested examples are already completed.")
        return

    print(f"Loading model {cfg['models']['slm']} ...")
    model, tokenizer = load_slm(cfg["models"]["slm"], device=cfg["device"])
    generate_fn = make_generator(
        model,
        tokenizer,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
    )

    print(f"Evaluating {len(remaining)} examples (checkpointing every {args.save_every}) ...")
    all_records: list[EvalRecord] = []
    start_time = time.time()

    # Process in chunks so we can print progress and checkpoint to disk.
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
        all_records.extend(chunk_records)

        done = start_idx + len(all_records)
        elapsed = time.time() - start_time
        rate = len(all_records) / elapsed if elapsed > 0 else 0
        remaining_n = len(examples) - done
        eta_min = (remaining_n / rate / 60) if rate > 0 else float("nan")
        correct_so_far = sum(r.correct for r in all_records)

        print(
            f"  [{done}/{len(examples)}] "
            f"running EX={correct_so_far/len(all_records):.3f}  "
            f"({rate:.2f} ex/s, ~{eta_min:.1f} min remaining)"
        )

        # Checkpoint: existing (already-saved) + everything done so far this run.
        combined = existing + [r.__dict__ for r in all_records]
        _save(combined, RESULTS_FILE)

    final_records = existing + [r.__dict__ for r in all_records]
    ex_score = execution_accuracy([r["correct"] for r in final_records])
    print(f"\nExecution Accuracy (floor): {ex_score:.3f}  ({sum(r['correct'] for r in final_records)}/{len(final_records)})")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
