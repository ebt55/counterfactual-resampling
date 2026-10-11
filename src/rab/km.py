"""Read-only access to the Kobayashi Maru (KM) records and task tree (guide §2, §3, §10).

Vocabulary: a *run* here is one KM batch directory,
``KM_ROOT/results/runs/<run_id>/items.jsonl``, e.g. ``dsv41flash-sal-v2_baseline_I4_s0``
(KM calls this the ``batch_id``). KM's own per-record ``run_id`` field is
``f"{batch_id}/{position}"``; this module never uses it.

Task dirs follow ``harness.batch.build_batch``: a solvable item lives at
``KM_ROOT/tasks/<item_key>`` (``item_key == task_id``), an impossible variant at
``KM_ROOT/tasks/impossible/<item_key>`` (``item_key`` like
``t26_semver_compare__unreachable``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from rab import KM_ROOT
from rab.constants import BATCHES, RUN_LINE, SEEDS
from harness.batch import IMPOSSIBLE_DIR, make_batch_id

RUNS_DIR = KM_ROOT / "results" / "runs"
TASKS_DIR = KM_ROOT / "tasks"


def run_dir(run_id: str) -> Path:
    """The KM batch directory holding ``items.jsonl`` for ``run_id``."""
    return RUNS_DIR / run_id


def iter_records(run_id: str) -> Iterator[dict]:
    """Yield every record of one run, in file order (= batch position order)."""
    path = run_dir(run_id) / "items.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no KM records at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def line_run_ids(line: str = RUN_LINE) -> list[str]:
    """The 60 runs ``<line>_baseline_I{0,2,4,8,15}_s{0..11}``, I-major then seed.

    Built from the constants, so ``<line>-nonotes_...`` can never match.
    """
    return [make_batch_id(line, "baseline", I, s) for I in BATCHES for s in SEEDS]


def iter_line_records(line: str = RUN_LINE) -> Iterator[dict]:
    """Every record of every run in :func:`line_run_ids`, in that order."""
    for run_id in line_run_ids(line):
        yield from iter_records(run_id)


def load_record(run_id: str, position: int) -> dict:
    """The one record of ``run_id`` at batch ``position``."""
    for rec in iter_records(run_id):
        if rec["position"] == position:
            return rec
    raise KeyError(f"{run_id} has no record at position {position}")


def task_dir(item_key: str, kind: str) -> Path:
    """KM task dir for an item: ``tasks/<item_key>`` if solvable, else ``tasks/impossible/<item_key>``.

    ``kind`` is ``record["mutation"] or "solvable"``; any non-solvable kind is an
    impossible variant (``record["is_impossible"]`` is true). Raises if the dir is absent.
    """
    path = TASKS_DIR / item_key if kind == "solvable" else TASKS_DIR / IMPOSSIBLE_DIR / item_key
    if not path.is_dir():
        raise FileNotFoundError(f"no KM task dir for {item_key!r} ({kind}) at {path}")
    return path


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Hex sha256 of a string's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
