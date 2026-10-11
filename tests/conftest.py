"""Shared test setup: put the KM checkout on ``sys.path`` (guide §10).

Mirrors ``blockpoint/__init__.py`` (same ``.env``, same default) so KM modules import
even before ``blockpoint`` does. Never reads ``$KM_ROOT/.env``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)  # this repo's .env only

DEFAULT_KM_ROOT = Path(r"C:\Users\ebin\claude-ground\ai-incident-response\dosecurve")
KM_ROOT = Path(os.environ.get("KM_ROOT") or DEFAULT_KM_ROOT).resolve()
if str(KM_ROOT) not in sys.path:
    sys.dont_write_bytecode = True  # never write __pycache__ into the KM checkout
    sys.path.insert(0, str(KM_ROOT))


@pytest.fixture(scope="session")
def km_root() -> Path:
    """The KM checkout the package resolved (same rule as ``blockpoint.KM_ROOT``)."""
    import blockpoint

    assert blockpoint.KM_ROOT == KM_ROOT, (blockpoint.KM_ROOT, KM_ROOT)
    assert (KM_ROOT / "harness" / "records.py").is_file(), KM_ROOT
    return KM_ROOT
