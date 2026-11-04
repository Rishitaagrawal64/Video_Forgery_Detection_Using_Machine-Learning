"""Application logging setup (level comes from FORGERY_LOG_LEVEL)."""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def resolve_level(name: str) -> int:
    level = logging.getLevelName(name.strip().upper())
    if not isinstance(level, int):
        raise ValueError(f"Invalid FORGERY_LOG_LEVEL {name!r}; use DEBUG, INFO, WARNING, ERROR or CRITICAL")
    return level


def setup_logging(level_name: str = "INFO") -> None:
    logging.basicConfig(level=resolve_level(level_name), format=_FORMAT)
