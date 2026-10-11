"""rab (resample-at-block): the block-point swap resampling pilot (build guide section 10).

Importing this package does two things and nothing else:

1. loads this repo's own ``.env`` (``REPO_ROOT / ".env"``) into the process
   environment without overriding variables already set. ``$KM_ROOT/.env`` is never
   read: those keys are old.
2. resolves ``KM_ROOT`` (the Kobayashi Maru checkout at tag ``freeze-2026-09-14b``)
   from the ``KM_ROOT`` environment variable, defaulting to Ebin's checkout, and puts
   it on ``sys.path`` so ``harness.*`` and ``detectors.*`` import read-only. Nothing
   under ``KM_ROOT`` is ever written (bytecode writing is switched off for the process).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

#: This repository's root (``src/rab/__init__.py`` is two levels below it).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env", override=False)

DEFAULT_KM_ROOT = Path(r"C:\Users\ebin\claude-ground\ai-incident-response\dosecurve")

#: The Kobayashi Maru checkout; read-only.
KM_ROOT: Path = Path(os.environ.get("KM_ROOT") or DEFAULT_KM_ROOT).resolve()

if str(KM_ROOT) not in sys.path:
    # importing KM must not write __pycache__ into the KM checkout
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(KM_ROOT))

__all__ = ["KM_ROOT", "REPO_ROOT"]
