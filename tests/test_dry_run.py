"""Keyless end-to-end dry run (guide §10; design v2): Docker, FakeProvider agent,
FakeSimulator, FakeDiscriminator, everything under ``results/dry/``.

An exploratory pilot of one event first, then the headline run of one event per kind:
the headline run re-runs the pilot event in full (its ``-x`` files stay beside), a second
headline run skips every slot, every result carries the git state, the tables rebuild
byte-for-byte from the files with the pilot listed and excluded, and the placebo mode runs.
"""

from __future__ import annotations

import gzip
import json
import shutil

import pytest

from rab import analyze
from rab.constants import RUNS_PER_EVENT
from rab.discriminate import FakeDiscriminator, discriminate, read_rows
from rab.run import DRY_ROOT, run

ROOT = DRY_ROOT / "pytest"


@pytest.mark.docker
def test_dry_run_end_to_end():
    from harness.sandbox import docker_available, image_available

    if not docker_available() or not image_available():
        pytest.skip("Docker or dosecurve-sandbox:latest is not available")
    shutil.rmtree(ROOT, ignore_errors=True)
    try:
        pilot = run(dry_run=True, root=ROOT, per_kind=1, pilot=1, workers=4, exploratory=True)
        n_event = 18
        assert pilot["slots_written"] == n_event and pilot["exploratory"] is True
        x_files = sorted(ROOT.glob("continuations/*/*-x.json.gz"))
        assert len(x_files) == n_event

        first = run(dry_run=True, root=ROOT, per_kind=1, pilot=None, workers=4)
        n_slots = sum(sum(RUNS_PER_EVENT[k].values()) for k in RUNS_PER_EVENT)
        assert first["dropped"] == [] and first["slots_written"] == n_slots and first["slots_skipped"] == 0
        again = run(dry_run=True, root=ROOT, per_kind=1, pilot=None, workers=4)
        assert again["slots_written"] == 0 and again["slots_skipped"] == n_slots
        assert sorted(ROOT.glob("continuations/*/*-x.json.gz")) == x_files

        files = analyze.slot_files(ROOT)
        assert len(files) == n_slots + n_event
        for path in (files[0], x_files[0]):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                slot = json.load(fh)
            assert {"cls", "cont", "placed", "replay", "timing", "git"} <= set(slot)
            assert len(slot["git"]["git_hash"]) == 40 and isinstance(slot["git"]["dirty"], bool)
            assert slot["exploratory"] is path.name.endswith("-x.json.gz")
            assert slot["cont"]["dry_run"] is True and slot["replay"]["ok"]
        sims = [json.loads(l) for l in (ROOT / "sims.jsonl").read_text(encoding="utf-8").splitlines()]
        assert sims and all("git" in s for s in sims)

        discriminate(ROOT, FakeDiscriminator())
        assert all("git" in r and r["exploratory"] is False for r in read_rows(ROOT / "discriminator.jsonl"))
        a = analyze.run(ROOT, n_perm=300)["text"]
        b = analyze.run(ROOT, n_perm=300)["text"]
        assert a == b == (ROOT / "tables.md").read_text(encoding="utf-8")
        assert "## Verdict" in a and "## Discriminator" in a and "## Exploratory runs (excluded)" in a
        meta = json.loads((ROOT / "tables_meta.json").read_text(encoding="utf-8"))
        assert meta["n_exploratory"] == n_event and "git_hash" in meta["git"]
        placebo = analyze.run(ROOT, placebo_mode=True, n_perm=300)
        assert (ROOT / "tables_placebo.md").is_file() and set(placebo["verdicts"]) == {"solvable", "unreachable"}
        assert (ROOT / "hand_read.md").is_file() and (ROOT / "hand_read_key.json").is_file()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
