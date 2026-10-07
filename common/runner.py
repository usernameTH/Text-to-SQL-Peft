"""Shared evaluation loop: prompt -> generate -> execute -> score.

Every baseline and experiment runs its model through this single function, so
scoring is identical across configurations.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from common.data_loader import Example, load_schema
from common.db_executor import resolve_db_path, try_execute
from common.evaluation import execution_match
from common.profiler import timed
from common.schema_serializer import serialize

# A function that turns a prompt string into a predicted SQL string.
GenerateFn = Callable[[str], str]
# A function that selects few-shot exemplars for an example (or None for zero-shot).
ExemplarFn = Callable[[Example], Sequence[Example]]


@dataclass(frozen=True)
class EvalRecord:
    """The outcome of evaluating one example.

    Attributes:
        db_id: The example's database.
        question: The natural-language question.
        gold_sql: The reference SQL.
        predicted_sql: The model's SQL.
        correct: Whether predicted and gold result sets matched.
        latency_s: Wall-clock seconds for generation.
    """

    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    correct: bool
    latency_s: float


def run_evaluation(
    examples: Sequence[Example],
    generate_fn: GenerateFn,
    *,
    tables_json: str | Path,
    database_dir: str | Path,
    schema_style: str = "simple_ddl",
    include_keys: bool = True,
    exemplar_fn: ExemplarFn | None = None,
) -> list[EvalRecord]:
    """Evaluate ``generate_fn`` over ``examples`` and return per-example records.

    Args:
        examples: The examples to evaluate (e.g. the dev set).
        generate_fn: Maps a prompt to a predicted SQL string.
        tables_json: Path to Spider's ``tables.json``.
        database_dir: Path to Spider's ``database/`` folder.
        schema_style: Schema serialization style (``simple_ddl`` | ``verbose``).
        include_keys: Whether to include keys in the serialized schema.
        exemplar_fn: Optional selector of few-shot exemplars per example.

    Returns:
        One :class:`EvalRecord` per input example.
    """
    # Imported here to avoid a circular import at module load time.
    from common.prompt_builder import build_prompt

    records: list[EvalRecord] = []
    for ex in examples:
        schema = load_schema(ex.db_id, tables_json)
        schema_text = serialize(schema, style=schema_style, include_keys=include_keys)
        exemplars = exemplar_fn(ex) if exemplar_fn else None
        prompt = build_prompt(schema_text, ex.question, exemplars=exemplars)

        predicted_sql, latency_s = timed(lambda: generate_fn(prompt))

        db_path = resolve_db_path(ex.db_id, database_dir)
        pred_result = try_execute(predicted_sql, db_path)
        gold_result = try_execute(ex.gold_sql, db_path)
        correct = (
            pred_result is not None
            and gold_result is not None
            and execution_match(pred_result, gold_result)
        )

        records.append(
            EvalRecord(
                db_id=ex.db_id,
                question=ex.question,
                gold_sql=ex.gold_sql,
                predicted_sql=predicted_sql,
                correct=correct,
                latency_s=latency_s,
            )
        )
    return records
