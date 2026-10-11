"""Keyless end-to-end dry run (guide §10; design v2): Docker, FakeProvider agent,
FakeSimulator, FakeDiscriminator, one event per kind, everything under ``results/dry/``.

Asserts every slot file is written, a second run skips them all (resumable), the tables
rebuild byte-for-byte from the files, and the placebo mode runs.
"""

from __future__ import annotations

import gzip
import json
import shutil

import pytest

from rab import analyze
from rab.constants import RUNS_PER_EVENT
from rab.discriminate import FakeDiscriminator, discriminate
from rab.run import DRY_ROOT, run, slot_path

ROOT = DRY_ROOT / "pytest"


@pytest.mark.docker
def test_dry_run_end_to_end():
    from harness.sandbox import docker_available, image_available

    if not docker_available() or not image_available():
        pytest.skip("Docker or dosecurve-sandbox:latest is not available")
    shutil.rmtree(ROOT, ignore_errors=True)
    try:
        first = run(dry_run=True, root=ROOT, per_kind=1, pilot=None, workers=4)
        n_slots = sum(sum(RUNS_PER_EVENT[k].values()) for k in RUNS_PER_EVENT)
        assert first["dropped"] == [] and first["slots_written"] == n_slots
        again = run(dry_run=True, root=ROOT, per_kind=1, pilot=None, workers=4)
        assert again["slots_written"] == 0 and again["slots_skipped"] == n_slots

        files = analyze.slot_files(ROOT)
        assert len(files) == n_slots
        with gzip.open(files[0], "rt", encoding="utf-8") as fh:
            slot = json.load(fh)
        assert {"cls", "cont", "placed", "replay", "timing"} <= set(slot)
        assert slot["cont"]["dry_run"] is True and slot["replay"]["ok"]

        discriminate(ROOT, FakeDiscriminator())
        a = analyze.run(ROOT, n_perm=300)["text"]
        b = analyze.run(ROOT, n_perm=300)["text"]
        assert a == b == (ROOT / "tables.md").read_text(encoding="utf-8")
        assert "## Verdict" in a and "## Discriminator" in a
        placebo = analyze.run(ROOT, placebo_mode=True, n_perm=300)
        assert (ROOT / "tables_placebo.md").is_file() and set(placebo["verdicts"]) == {"solvable", "unreachable"}
        assert (ROOT / "hand_read.md").is_file() and (ROOT / "hand_read_key.json").is_file()
        assert slot_path(ROOT, slot["event_id"], slot["arm"], slot["run_idx"]).is_file()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
