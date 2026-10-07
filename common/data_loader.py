"""Load the Spider dataset and resolve questions to their database schema.

  """
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """A single example.

    Attributes:
        db_id: Identifier of the database this question targets.
        question: The natural-language question.
        gold_sql: The reference (ground-truth) SQL query.
    """

    db_id: str
    question: str
    gold_sql: str


@dataclass(frozen=True)
class Schema:
    """Structural description of one database (tables, columns, keys).

    Holds *structure only* -- never row data.

    Attributes:
        db_id: The database identifier.
        tables: Maps each table name to its ``[(column, type), ...]``.
        primary_keys: Maps each table name to its list of primary-key columns.
        foreign_keys: ``(table1, column1, table2, column2)`` relationships,
            where ``column1`` in ``table1`` references ``column2`` in ``table2``.
    """

    db_id: str
    tables: dict[str, list[tuple[str, str]]]
    primary_keys: dict[str, list[str]]
    foreign_keys: list[tuple[str, str, str, str]]


def load_examples(json_path: str | Path) -> list[Example]:
    """Load Spider examples from a split JSON file.

    Args:
        json_path: Path to ``train_spider.json`` or ``dev.json``.

    Returns:
        List of :class:`Example` objects, one per entry in the file.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
    """
    with Path(json_path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Keep only the three fields we need; drop token/parsed-tree extras.
    return [
        Example(db_id=item["db_id"], question=item["question"], gold_sql=item["query"])
        for item in raw
    ]


def load_schema(db_id: str, tables_json: str | Path) -> Schema:
    """Build a :class:`Schema` for ``db_id`` from ``tables.json``.

    Args:
        db_id: The database identifier.
        tables_json: Path to Spider's ``tables.json``.

    Returns:
        The :class:`Schema` for the requested database.

    Raises:
        KeyError: If ``db_id`` is not present in ``tables.json``.
    """
    return _load_all_schemas(str(tables_json))[db_id]


@lru_cache(maxsize=None)
def _load_all_schemas(tables_json: str) -> dict[str, Schema]:
    """Parse every database in ``tables.json`` into a ``{db_id: Schema}`` map.

    Cached so the file is read and parsed once, even when ``load_schema`` is
    called for each of the ~1,000 dev examples.
    """
    with Path(tables_json).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {entry["db_id"]: _parse_schema(entry) for entry in raw}


def _parse_schema(entry: dict) -> Schema:
    """Convert one raw ``tables.json`` entry into a :class:`Schema`.

    Spider stores columns as ``[table_index, column_name]`` pairs (with a
    leading ``[-1, "*"]`` wildcard), and keys as indices into that column list.
    We resolve those indices back to ``(table, column)`` names.
    """
    table_names: list[str] = entry["table_names_original"]
    columns: list[list] = entry["column_names_original"]  # [[table_idx, name], ...]
    column_types: list[str] = entry["column_types"]

    # Group real columns (skip the wildcard at index 0, whose table_idx is -1).
    tables: dict[str, list[tuple[str, str]]] = {name: [] for name in table_names}
    for (table_idx, col_name), col_type in zip(columns, column_types):
        if table_idx < 0:
            continue
        tables[table_names[table_idx]].append((col_name, col_type))

    # Helper: turn a column index (into `columns`) into (table, column).
    def resolve(col_index: int) -> tuple[str, str]:
        table_idx, col_name = columns[col_index]
        return table_names[table_idx], col_name

    primary_keys: dict[str, list[str]] = {name: [] for name in table_names}
    for col_index in entry["primary_keys"]:
        table, column = resolve(col_index)
        primary_keys[table].append(column)

    foreign_keys: list[tuple[str, str, str, str]] = []
    for from_idx, to_idx in entry["foreign_keys"]:
        t1, c1 = resolve(from_idx)
        t2, c2 = resolve(to_idx)
        foreign_keys.append((t1, c1, t2, c2))

    return Schema(
        db_id=entry["db_id"],
        tables=tables,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
    )