"""Evaluation metrics.

Every baseline and experiment imports these. Defining them once guarantees
all configurations are scored identically, which is essential for the
gap-closed comparison to be valid.
"""
from __future__ import annotations

from collections.abc import Sequence

from common.db_executor import ResultSet


def execution_match(predicted: ResultSet, gold: ResultSet, order_matters: bool = False) -> bool:
    """Return True if two result sets are considered equal.

    By default result sets are compared as multisets (order-insensitive),
    since SQL result ordering is only meaningful under ORDER BY.

    Args:
        predicted: Result set from the predicted query.
        gold: Result set from the gold query.
        order_matters: If True, compare as ordered sequences instead.

    Returns:
        Whether the two result sets match.
    """
    if order_matters:
        return list(predicted) == list(gold)
    return sorted(map(repr, predicted)) == sorted(map(repr, gold))


def execution_accuracy(matches: Sequence[bool]) -> float:
    """Compute Execution Accuracy (EX): the fraction of matching examples.

    EX = (1/N) * sum_n 1[ result(pred_n) == result(gold_n) ]

    Args:
        matches: Per-example booleans from :func:`execution_match`.

    Returns:
        Execution accuracy in [0, 1].

    Raises:
        ValueError: If ``matches`` is empty.
    """
    if not matches:
        raise ValueError("Cannot compute accuracy over zero examples.")
    return sum(1 for m in matches if m) / len(matches)


def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    """Component-level Exact Match  between two SQL strings.

    Secondary metric. Guards against EX false positives (a wrong query that
    coincidentally returns the gold result set).

    Args:
        predicted_sql: The predicted SQL string.
        gold_sql: The gold SQL string.

    Returns:
        Whether the queries match at the component level.
    """
    raise NotImplementedError


#Only Execution metrics is used in final write up not exact match