"""Load and read `config.yaml`.

Every phase reads its settings through this module so that a run's behaviour is
fully described by one versioned file. Scripts accept CLI overrides, but the
resolved values are always logged with the run (CLAUDE.md §8: reproducibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read a YAML config file into a plain dict.

    Args:
        path: Path to the YAML file, relative to the current working directory.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If the file does not exist. We fail loudly rather
            than falling back to defaults, because a silently-defaulted config
            produces metrics that cannot be reproduced.
        ValueError: If the file does not contain a YAML mapping.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Run from the repository root, or pass --config."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping, "
            f"got {type(loaded).__name__}."
        )
    return loaded


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value using dotted notation.

    `cfg_get(cfg, "ingest.extraction_mode")` returns `cfg["ingest"]["extraction_mode"]`.

    Args:
        config: The configuration mapping returned by `load_config`.
        dotted_key: Dot-separated path to the value, e.g. "paths.raw_dir".
        default: Returned if any segment of the path is missing.

    Returns:
        The value at `dotted_key`, or `default` if it is not present.
    """
    current: Any = config
    for segment in dotted_key.split("."):
        if not isinstance(current, dict) or segment not in current:
            return default
        current = current[segment]
    return current
