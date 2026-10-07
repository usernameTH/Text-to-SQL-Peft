"""Execute SQL against a Spider SQLite database and return result sets.

Used only at evaluation time.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

# A result set: rows, each a tuple of column values.
ResultSet = list[tuple]


def resolve_db_path(db_id: str, database_dir: str | Path) -> Path:
    """Return the ``.sqlite`` file path for a database id.

    Spider stores each database at ``<database_dir>/<db_id>/<db_id>.sqlite``.

    Args:
        db_id: The database identifier.
        database_dir: Path to Spider's ``database/`` folder.

    Returns:
        Path to the database's ``.sqlite`` file.
    """
    return Path(database_dir) / db_id / f"{db_id}.sqlite"


def execute(sql: str, db_path: str | Path, timeout: float = 30.0) -> ResultSet:
    """Run ``sql`` against the SQLite database at ``db_path`` (read-only).

    The database is opened read-only so a predicted query can never modify the
    evaluation data. A watchdog interrupts queries that exceed ``timeout``.

    Args:
        sql: The SQL query to execute.
        db_path: Path to the ``.db``/``.sqlite`` file.
        timeout: Maximum seconds to allow the query to run.

    Returns:
        The result set as a list of row tuples.

    Raises:
        sqlite3.Error: On invalid SQL, execution failure, or timeout. Callers
            decide whether a failed query counts as incorrect (see
            :func:`try_execute`).
    """
    uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout)

    # interrupt a runaway query after `timeout` seconds.
    watchdog = threading.Timer(timeout, conn.interrupt)
    try:
        watchdog.start()
        return conn.execute(sql).fetchall()
    finally:
        watchdog.cancel()
        conn.close()


def try_execute(sql: str, db_path: str | Path, timeout: float = 30.0) -> ResultSet | None:
    """Execute ``sql`` but return ``None`` instead of raising on failure.

    Convenience wrapper for scoring: a predicted query that fails to execute is
    simply wrong, so callers usually prefer ``None`` over an exception.

    Args:
        sql: The SQL query to execute.
        db_path: Path to the ``.db``/``.sqlite`` file.
        timeout: Maximum seconds to allow the query to run.

    Returns:
        The result set, or ``None`` if execution failed for any reason.
    """
    try:
        return execute(sql, db_path, timeout)
    except sqlite3.Error:
        return None