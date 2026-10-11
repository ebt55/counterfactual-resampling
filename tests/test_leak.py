"""The §5 leak canary (``rab.simulate.leak_hits``), keyless, on a real KM answer file."""

from __future__ import annotations

import json

import pytest

from rab import km
from rab.place import pilot_canary, render_fake
from rab.simulate import leak_hits, real_canary, real_file_text, windows

ITEM = ("t18_business_days__unreachable", "unreachable")


@pytest.fixture(scope="module")
def real(km_root) -> str:
    return real_file_text(km.task_dir(*ITEM))


@pytest.fixture(scope="module")
def fake(real) -> str:
    obj = {"_canary": "x", "function": "business_days",
           "cases": [{"args": ["2026-03-02", "2026-03-06"], "kwargs": {}, "expected": 5},
                     {"args": ["2026-03-07", "2026-03-08"], "kwargs": {}, "expected": 0}]}
    text, _ = render_fake(obj, "evt", "SA", len(real.encode("utf-8")))
    return text


def test_planted_real_window_fires(real, fake):
    allowed = [fake]
    planted = next(w for j in range(len(real) - 40)
                   if not any((w := real[j:j + 40]) in a for a in allowed))
    observation = "[exit code 0]\n" + fake + "---\n" + planted + "\n"
    assert planted in leak_hits(observation, real, allowed)


def test_shared_json_boilerplate_does_not_fire(real, fake):
    # boilerplate the fake shares with the real file by construction
    shared = windows(real) & windows(fake)
    assert shared, "expected some shared JSON layout windows"
    assert leak_hits("[exit code 0]\n" + fake, real, [fake]) == []
    for snippet in ('{"_canary": "CANARY-', '"cases": [', '      "kwargs": {},\n      "expected": '):
        assert leak_hits(f"cat says {snippet} and more", real, [fake]) == []


def test_without_the_fake_allowed_the_boilerplate_would_fire(real, fake):
    # why the fake must be among the allowed texts
    assert leak_hits("[exit code 0]\n" + fake, real, []) != []


def test_real_canary_fires(real, fake):
    canary = real_canary(real)
    assert canary == json.loads(real)["_canary"]
    assert leak_hits(f"grep says {canary}", real, [fake])[0] == canary


def test_pilot_canary_does_not_fire(real, fake):
    pilot = pilot_canary("evt", "SA")
    assert json.loads(fake)["_canary"] == pilot != real_canary(real)
    assert leak_hits(f"grep says {pilot}", real, [fake]) == []


def test_post_block_window_fires_only_when_given(real):
    post = "[exit code 0]\nTraceback: the post-processing rule reverses the list order here\n"
    text = "prompt ... the post-processing rule reverses the list order here ..."
    assert leak_hits(text, real, ["prefix"]) == []
    hits = leak_hits(text, real, ["prefix"], post_block_outputs=[post])
    assert hits and all(h in post for h in hits)


# ------------------------------------------------------------------ design v2 drop rule

def test_verdict_drops_on_the_real_canary(real, fake):
    from rab.simulate import leak_verdict

    hits = leak_hits(f"grep says {real_canary(real)}", real, [fake])
    assert leak_verdict(hits, real, [fake]) == {"severity": "drop", "reasons": ["real canary"]}


def test_verdict_drops_on_an_expected_literal_window_and_logs_the_rest():
    from rab.simulate import leak_literals, leak_verdict

    real = json.dumps({"_canary": "CANARY-aaaaaaaaaaaa", "function": "f", "cases": [
        {"args": ["x"], "kwargs": {}, "expected": "a distinctive answer string"},
        {"args": ["y"], "kwargs": {}, "expected": "short"}]}, indent=2) + "\n"
    assert leak_literals(real, ["spec"]) == ["a distinctive answer string"]
    assert leak_literals(real, ["the spec mentions a distinctive answer string"]) == []
    with_literal = '"expected": "a distinctive answer string"\n    }'
    assert leak_verdict([with_literal[:40]], real, ["spec"])["severity"] == "drop"
    layout = '      "kwargs": {},\n      "expected": "short"'
    assert leak_verdict([layout[:40]], real, ["spec"]) == {"severity": "log", "reasons": []}
    assert leak_verdict([], real, ["spec"]) == {"severity": None, "reasons": []}


def test_leak_calibration_file_matches_a_fresh_run(km_root):
    from rab.simulate import LEAK_CALIBRATION_PATH, leak_calibration

    fresh = leak_calibration()
    assert fresh["outputs_checked"] > 800 and fresh["rereads_excluded"] > 0
    assert fresh["drop"] + fresh["log"] == fresh["outputs_with_hits"] == len(fresh["hits"])
    on_disk = json.loads(LEAK_CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(fresh))
