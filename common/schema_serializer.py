"""Turn a database :class:`~common.data_loader.Schema` into prompt text.

This is a tuned enhancement point (SimpleDDL vs verbose), so it lives in one
place and is selected via config, never re-implemented per experiment.
"""
from __future__ import annotations

from common.data_loader import Schema


def serialize(schema: Schema, style: str = "simple_ddl", include_keys: bool = True) -> str:
    """Serialize a schema to text in the requested style.

    Args:
        schema: The database schema (structure only).
        style: ``"simple_ddl"`` for CREATE TABLE statements, or ``"verbose"``
            for natural-language descriptions.
        include_keys: Whether to include primary/foreign key information.

    Returns:
        The schema rendered as a prompt-ready string.

    Raises:
        ValueError: If ``style`` is not recognised.
    """
    if style == "simple_ddl":
        return _to_simple_ddl(schema, include_keys)
    if style == "verbose":
        return _to_verbose(schema, include_keys)
    raise ValueError(f"Unknown schema style: {style!r}")


def _to_simple_ddl(schema: Schema, include_keys: bool) -> str:
    """Render the schema as ``CREATE TABLE`` statements."""
    statements: list[str] = []
    for table, columns in schema.tables.items():
        lines = [f"  {name} {col_type}" for name, col_type in columns]

        if include_keys:
            primary = schema.primary_keys.get(table, [])
            if primary:
                lines.append(f"  PRIMARY KEY ({', '.join(primary)})")
            for t1, c1, t2, c2 in schema.foreign_keys:
                if t1 == table:
                    lines.append(f"  FOREIGN KEY ({c1}) REFERENCES {t2}({c2})")

        body = ",\n".join(lines)
        statements.append(f"CREATE TABLE {table} (\n{body}\n);")

    return "\n\n".join(statements)


def _to_verbose(schema: Schema, include_keys: bool) -> str:
    """Render the schema as natural-language descriptions."""
    parts: list[str] = []
    for table, columns in schema.tables.items():
        col_phrases = [f"{name} ({col_type})" for name, col_type in columns]
        sentence = f"Table {table} has columns: {', '.join(col_phrases)}."

        if include_keys:
            primary = schema.primary_keys.get(table, [])
            if primary:
                sentence += f" Its primary key is {', '.join(primary)}."

        parts.append(sentence)

    if include_keys and schema.foreign_keys:
        relations = [
            f"{t1}.{c1} references {t2}.{c2}" for t1, c1, t2, c2 in schema.foreign_keys
        ]
        parts.append("Foreign keys: " + "; ".join(relations) + ".")

    return "\n".join(parts)