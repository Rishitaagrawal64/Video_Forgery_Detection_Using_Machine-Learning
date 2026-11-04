import logging

import pytest

from app.core.logging import resolve_level


def test_resolve_level_is_case_insensitive():
    assert resolve_level("debug") == logging.DEBUG
    assert resolve_level(" Warning ") == logging.WARNING


def test_resolve_level_rejects_garbage():
    with pytest.raises(ValueError, match="FORGERY_LOG_LEVEL"):
        resolve_level("loud")
