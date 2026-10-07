"""Assemble the final prompt from instruction, schema and question.

The prompt template is fixed across experiments so that only the enhancement
varies
"""
from __future__ import annotations

from collections.abc import Sequence

from common.data_loader import Example

INSTRUCTION = (
    "You are an expert SQL writer. Given the database schema below, write a "
    "single SQLite query that answers the question. Output only the SQL query, "
    "with no explanation."
)


def build_prompt(
    schema_text: str,
    question: str,
    exemplars: Sequence[Example] | None = None,
) -> str:
    """Build the model prompt.

    Args:
        schema_text: The serialized schema (from ``schema_serializer.serialize``).
        question: The natural-language question to answer.
        exemplars: Optional solved examples to prepend as few-shot
            demonstrations (used by the few-shot and RAG experiments). ``None``
            or empty gives a zero-shot prompt.

    Returns:
        The complete prompt string, ending on an ``SQL:`` cue for the model to
        continue from.
    """
    blocks: list[str] = [INSTRUCTION]

    if exemplars:
        blocks.append(_format_exemplars(exemplars))

    blocks.append(f"Schema:\n{schema_text}")
    blocks.append(f"Question: {question}\nSQL:")

    return "\n\n".join(blocks)


def _format_exemplars(exemplars: Sequence[Example]) -> str:
    """Render solved examples as ``Question:`` / ``SQL:`` demonstration pairs."""
    demos = [f"Question: {ex.question}\nSQL: {ex.gold_sql}" for ex in exemplars]
    return "Here are some examples:\n\n" + "\n\n".join(demos)