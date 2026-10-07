"""Load and validate the global project configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Immutable view over ``config.yaml``.

    Frozen so no experiment can mutate global settings at runtime.
    """

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read ``config.yaml`` into an immutable :class:`Config`.

    Args:
        path: Path to the YAML config file.

    Returns:
        A frozen ``Config`` wrapping the parsed dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return Config(raw=yaml.safe_load(fh))
