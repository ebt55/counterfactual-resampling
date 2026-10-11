"""Frozen interfaces between chunks (build plan "Interfaces"; guide §2, §4, §5).

``Event`` is one row of ``data/events.jsonl`` (``rab.select``), field for field.
``Sim`` is one simulator draw (``rab.simulate``), one line of ``results/sims.jsonl``.
``Placed`` is the state after placement and the blocked call's re-execution
(``rab.place``).

``CallDiff`` and ``Replay`` are defined in ``rab.prefix`` (chunk B owns that file) and
re-exported here lazily (module ``__getattr__``), so ``from rab.types import Replay``
works and importing ``rab.types`` never pulls in Docker code or risks an import cycle.
Moving the two definitions here is chunk B's edit to make; consumers do not change.

Fields after the frozen ones carry defaults, so rows written before they existed still
load.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from rab import REPO_ROOT

EVENTS_PATH = REPO_ROOT / "data" / "events.jsonl"


def __getattr__(name: str):
    if name in ("CallDiff", "Replay"):
        from rab import prefix

        return getattr(prefix, name)
    raise AttributeError(f"module 'rab.types' has no attribute {name!r}")


@dataclass
class Event:
    """One block point; see ``rab.select`` for the exact meaning of every field."""

    event_id: str
    run_id: str
    position: int
    task_id: str
    item_key: str
    kind: str
    draw_seed: int
    i: int
    k: int
    a: int
    call: dict
    recorded_obs_sha256: str
    read_path: str | None
    read_path_raw: list[str]
    combined: bool | None
    prefix_shows_size: bool
    work_mtime: str | None
    regrade_pass: bool
    declared: bool
    outcome: str | None
    note_names_file: bool
    later_reread: bool
    later_named: bool
    later_access: bool

    @classmethod
    def from_row(cls, row: dict) -> "Event":
        """Build from an ``events.jsonl`` row; the key set must match exactly."""
        want = {f.name for f in fields(cls)}
        have = set(row)
        if have != want:
            raise ValueError(f"event row keys differ: missing {sorted(want - have)}, "
                             f"extra {sorted(have - want)}")
        return cls(**row)

    def to_json(self) -> dict:
        return asdict(self)


def load_events(path: Path = EVENTS_PATH) -> list[Event]:
    """Every row of ``data/events.jsonl``, in file order (solvable, then unreachable)."""
    with open(path, "r", encoding="utf-8") as fh:
        return [Event.from_row(json.loads(line)) for line in fh if line.strip()]


@dataclass
class Sim:
    """One simulator draw for one (event, family): an attempt-set of 1 or 2 calls."""

    family: str                 # "SA" | "SB"
    model: str
    prompt_sha256: str          # sha256(system + "\x00" + user)
    raw_text: str               # the last attempt's reply, verbatim
    obj: dict | None            # parsed and validated, else None
    ok: bool
    failure: str | None         # "refusal" | "schema" | "leak" | None
    attempts: int
    usage: dict
    fenced: bool = False        # the accepted (or last) reply was wrapped in a ``` block
    event_id: str = ""
    detail: str | None = None   # schema failure reason, refusal category, or leak windows

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: dict) -> "Sim":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in known})


@dataclass
class Placed:
    """After placement (S: fake on both paths; R: nothing) and the blocked call's re-run."""

    arm: str                    # "R" | "SA" | "SB"
    observation: str            # the blocked call's live output, truncated as KM does
    truncated: bool
    obs_matches_recording: bool | None   # R only; rab.prefix.content_equal (decision 17b)
    snapshot_before: dict       # hashes, taken after placement
    canaries: dict              # view canaries (pilot answer canary for S)
    expected_values: list       # from the file on disk
    real_size: int
    fake_size: int | None
    size_matched: bool | None
    fake_correct_frac: float | None
    leak_hits: list[str]
    padded_by: int | None = None          # spaces added before the final newline (S)
    fake_agreement: dict | None = None    # rab.place.fake_agreement() in full (S)
    checks: dict = field(default_factory=dict)  # on-disk checks after placement (S)
    obs_matches_strict: bool | None = None      # R only; normalise() equality, reported beside

    def to_json(self) -> dict:
        return asdict(self)
