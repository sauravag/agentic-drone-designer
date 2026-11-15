"""Test isolation: no test sees the developer's real LLM keys or provider/model settings.

The snapshot is taken at import (before any fixture runs) so the opt-in live test
(tests/test_agent_e2e.py with ISR_FORGE_LIVE_LLM=1) can restore it explicitly.
"""

import os
from typing import Final

import pytest

LLM_ENV_VARS: Final[tuple[str, ...]] = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ISR_FORGE_LLM_PROVIDER",
                                        "ISR_FORGE_MODEL")
LIVE_LLM_ENV: Final[dict[str, str]] = {k: os.environ[k] for k in LLM_ENV_VARS if os.environ.get(k)}


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
