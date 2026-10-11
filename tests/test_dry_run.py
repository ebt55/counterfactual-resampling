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
import os
import shutil
import uuid

import pytest

from rab import analyze
from rab.constants import RUNS_PER_EVENT
from rab.discriminate import FakeDiscriminator, discriminate, read_rows
from rab.run import DRY_ROOT, run

# unique per test process: two agents running the suite at once must not share (and
# delete) one dry-run directory
ROOT = DRY_ROOT / f"pytest-{os.getpid()}-{uuid.uuid4().hex[:6]}"


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
        assert (ROOT / "tables_placebo.md").is_file() and set(placebo["placebo"]) == {"solvable", "unreachable"}
        assert (ROOT / "hand_read.md").is_file() and (ROOT / "hand_read_key.json").is_file()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)


class _Meter:
    def __init__(self):
        self.preflight_result, self.cash_usd, self.calls = None, 0.0, []

    @property
    def stop_reached(self):
        return self.cash_usd >= 15.0

    def add(self, provider, model, usage, seat="agent"):
        usd = 0.001 * int(usage.get("output_tokens") or 0)
        self.calls.append((provider, seat, usd))
        if provider == "openrouter":
            self.cash_usd += usd
        return usd


def test_stopper_and_stoppable_provider():
    from rab.run import StopRun, Stopper, StoppableProvider

    class P:
        model, name = "m", "fake"

        def chat(self, *a):
            return "turn"

    meter = _Meter()
    stopper = Stopper(meter)
    prov = StoppableProvider(P(), stopper)
    assert prov.chat() == "turn" and prov.model == "m"
    meter.preflight_result = {"abort": True, "reasons": ["zero cache"]}  # cached verdict of another worker
    with pytest.raises(StopRun):
        prov.chat()
    assert "pre-flight abort" in stopper.reason
    meter2 = _Meter()
    meter2.cash_usd = 15.0
    with pytest.raises(StopRun, match="ceiling"):
        Stopper(meter2).check()


def test_seed_meter_counts_spend_on_disk(tmp_path):
    from rab.run import seed_meter, slot_path, write_slot

    write_slot(slot_path(tmp_path, "e1", "R", 0), {"cont": {"cost_usd": 0.25}})
    write_slot(slot_path(tmp_path, "e1", "SA", 0, exploratory=True), {"cont": {"cost_usd": 0.5}})
    write_slot(slot_path(tmp_path, "e1", "B", 0, failed=True), {"cont": {"cost_usd": 0.125}})
    (tmp_path / "sims.jsonl").write_text(
        json.dumps({"family": "SB", "model": "glm", "usage": {"output_tokens": 1000}}) + "\n"
        + json.dumps({"family": "SA", "model": "sonnet", "usage": {"output_tokens": 1000}}) + "\n",
        encoding="utf-8")
    meter = _Meter()
    seeded = seed_meter(meter, tmp_path)
    assert seeded["agent_usd"] == pytest.approx(0.875)        # every slot file, failed ones too
    assert meter.cash_usd == pytest.approx(0.875 + 1.0)        # SB is cash, SA is credit
    assert [c[:2] for c in meter.calls] == [("openrouter", "simulator"), ("anthropic", "simulator")]


def test_wall_time_model_and_fallback(tmp_path):
    from rab.run import FALLBACK_EVENT_TIMING, event_timing, wall_estimate_s

    timing, src = event_timing(tmp_path / "missing.jsonl", events=[])
    assert "FALLBACK" in src and timing["solvable"]["work_s"] == FALLBACK_EVENT_TIMING["solvable"]["work_s"]
    t = {"solvable": {"work_s": 240.0, "max_slot_s": 40.0}, "unreachable": {"work_s": 1200.0, "max_slot_s": 250.0}}
    # 4 workers: 60 x 60 s + 60 x 300 s; 6 workers: 60 x 40 s + 60 x 250 s (the longest slot binds)
    assert wall_estimate_s({"solvable": 60, "unreachable": 60}, 4, t) == 60 * 60 + 60 * 300
    assert wall_estimate_s({"solvable": 60, "unreachable": 60}, 6, t) == 60 * 40 + 60 * 250
    log = tmp_path / "run_log.jsonl"
    rows = [{"status": "done", "event_id": "e1", "timing": {"total_s": s}} for s in (10.0, 30.0)]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    timing, src = event_timing(log, events=[{"event_id": "e1", "kind": "solvable"}])
    assert timing["solvable"] == {"work_s": 40.0, "max_slot_s": 30.0, "events": 1}
    assert timing["unreachable"]["events"] == 0


def test_consecutive_infrastructure_failures_stop_the_run():
    from rab.run import MAX_CONSECUTIVE_INFRA, StopRun, Stopper

    stopper = Stopper(_Meter())
    for _ in range(MAX_CONSECUTIVE_INFRA - 1):
        stopper.outcome(True)
    stopper.outcome(False)                      # a finished slot resets the count
    for _ in range(MAX_CONSECUTIVE_INFRA - 1):
        stopper.outcome(True)
    assert not stopper.event.is_set()
    stopper.outcome(True)
    assert "consecutive infrastructure failures" in stopper.reason
    with pytest.raises(StopRun):
        stopper.check()


def test_replay_infrastructure_failure_is_not_divergence():
    from types import SimpleNamespace as NS

    from rab.run import replay_infra_failure

    call = lambda i, ok: NS(index=i, content_matched=ok)
    genuine = NS(failures=[], calls=[call(0, True), call(1, False)], replayed=["a", "[exit code 0]\nother text"])
    assert replay_infra_failure(genuine) is None
    dead = NS(failures=[], calls=[call(0, False)], replayed=["[tool error] SandboxError: No such container: rab_x"])
    assert "No such container" in replay_infra_failure(dead)
    touch = NS(failures=["touch -d 'Sep 13 18:11' /work failed: ..."], calls=[], replayed=[])
    assert replay_infra_failure(touch).startswith("touch")


def test_simulator_error_in_the_slot_phase_is_not_a_crash(tmp_path, monkeypatch):
    import rab.simulate as S
    from rab.run import draws_for

    def boom(rec, ev, family, caller, cache):
        raise S.SimulatorError("SA claude-sonnet-5-5 e1: APIConnectionError")
    monkeypatch.setattr(S, "draw", boom)
    out = draws_for({"event_id": "e1", "kind": "solvable"}, {}, tmp_path, {"SA": None, "SB": None}, _Meter())
    assert out == {"SA": None, "SB": None}
    assert "simulator_error" in (tmp_path / "run_log.jsonl").read_text(encoding="utf-8")
