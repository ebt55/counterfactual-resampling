"""Tests for ``rab.prefix`` (guide §3, §10): message rebuild, normalise, fixture replay.

Keyless. The round-trip test reads KM records under ``KM_ROOT/results/runs`` and
skips if they are absent; the replay test is marked ``docker`` and skips cleanly
without Docker or the ``dosecurve-sandbox:latest`` image.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rab import km
from rab.prefix import (
    LS_DATE_RE,
    PYTEST_TIME_RE,
    build_messages,
    continuation_budget,
    normalise,
    replay,
)
from harness.agent import NUDGE_TEXT, rules_text
from harness.providers.openai_compat import OpenAICompatProvider

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_ID = "dsv41flash-sal-v2_baseline_I8_s6_p13"

#: Three records across runs, chosen for the transcript shapes they carry:
#: a nudge; several tool calls in one assistant turn with text beside them;
#: an unparseable arguments string that KM stored as {"_raw": ...}.
ROUND_TRIP = [
    ("dsv41flash-sal-v2_baseline_I0_s0", 8),
    ("dsv41flash-sal-v2_baseline_I15_s4", 1),
    ("dsv41flash-sal-v2_baseline_I0_s4", 2),
]


def _fixture() -> tuple[dict, dict]:
    rec = json.loads((FIXTURES / f"record_{FIXTURE_ID}.json").read_text(encoding="utf-8"))
    event = json.loads((FIXTURES / f"event_{FIXTURE_ID}.json").read_text(encoding="utf-8"))
    return rec, event


def _record(run_id: str, position: int) -> dict:
    try:
        return km.load_record(run_id, position)
    except FileNotFoundError:
        pytest.skip(f"KM records not present under {km.RUNS_DIR}")


def wire_to_transcript(wire: list[dict]) -> list[dict]:
    """Invert ``_to_api`` modulo the fields the module docstring lists as dropped."""
    out = []
    for m in wire:
        calls = [{"id": c["id"], "name": c["function"]["name"],
                  "args": json.loads(c["function"]["arguments"])}
                 for c in m.get("tool_calls", [])]
        for c in m.get("tool_calls", []):
            assert c["type"] == "function"
        out.append({"role": m["role"], "content": m["content"] or "",
                    "tool_calls": calls, "tool_call_id": m.get("tool_call_id")})
    return out


def transcript_to_wire(entries: list[dict]) -> list[dict]:
    """What ``_to_api`` should emit, written out by hand from the OpenAI wire shape."""
    out = []
    for e in entries:
        if e["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": e["tool_call_id"], "content": e["content"]})
        elif e["role"] == "assistant" and e["tool_calls"]:
            out.append({"role": "assistant", "content": e["content"] or None, "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                for c in e["tool_calls"]]})
        else:
            out.append({"role": e["role"], "content": e["content"]})
    return out


@pytest.mark.parametrize("run_id,position", ROUND_TRIP)
def test_build_messages_round_trips(run_id: str, position: int) -> None:
    rec = _record(run_id, position)
    tr = rec["transcript"]
    assert tr[0]["content"] == rules_text("standard")  # the frozen rules.md text
    tool_ks = [k for k, e in enumerate(tr) if e["role"] == "tool"]
    assert tool_ks
    for k in tool_ks:
        system, messages = build_messages(rec, k)
        wire = OpenAICompatProvider._to_api(system, messages)
        assert wire == transcript_to_wire(tr[:k]), k
        assert wire_to_transcript(wire) == tr[:k], k
        # tool_name is recovered for every tool message (Ollama needs it; _to_api does not)
        names = {c["id"]: c["name"] for e in tr[:k] for c in e["tool_calls"]}
        assert all(m.tool_name == names[m.tool_call_id] for m in messages if m.role == "tool")


def test_round_trip_records_cover_the_edge_shapes() -> None:
    recs = [_record(r, p) for r, p in ROUND_TRIP]
    entries = [e for rec in recs for e in rec["transcript"]]
    assert any(e["role"] == "user" and e["content"] == NUDGE_TEXT for e in entries)
    assert any(len(e["tool_calls"]) > 1 for e in entries)
    assert any(e["role"] == "assistant" and e["content"] and e["tool_calls"] for e in entries)
    assert any("_raw" in c["args"] for e in entries for c in e["tool_calls"])


def test_continuation_budget_on_fixture() -> None:
    rec, event = _fixture()
    budget = continuation_budget(rec, event)
    turn = rec["tool_calls"][event["i"]]["turn"]
    assert budget == {"max_turns": 30 - turn, "max_nudges": 2, "wall_clock_s": 600,
                      "prefix_turns": turn, "prefix_nudges": 0}
    # the last message rebuilt for the event is the assistant turn holding the blocked call
    _, messages = build_messages(rec, event["k"])
    assert messages[-1].role == "assistant"
    assert messages[-1].tool_calls[-1].id == event["call"]["id"]


LS = "\n".join([
    "total 40",
    "drwxr-xr-x 4 dc dc 4096 Jan  1  1970 .",
    "drwxr-xr-x 1 dc dc 4096 Sep 13 18:11 ..",
    "-rw-r--r-- 1 dc dc 1474 Jan  1  1970 expected_outputs.json",
    "crw-rw-rw- 1 root root 1, 3 Oct 11 05:40 null",
])
PYTEST = "....F.                                                     [100%]\n" \
         "=================== 1 failed, 5 passed in 0.12s ===================\n"


def test_normalise_only_touches_dates_and_timings() -> None:
    # ls -l: a different date normalises equal, in either column form
    assert normalise(LS) == normalise(LS.replace("Sep 13 18:11", "Oct 11 10:42")
                                      .replace("Jan  1  1970", "Sep  1  2025"))
    assert normalise(LS).count("<ls-date>") == 4
    # ls -l: a different size does not
    assert normalise(LS) != normalise(LS.replace(" 1474 ", " 1475 "))
    # pytest: a different timing normalises equal
    assert normalise("3 passed in 0.12s") == normalise("3 passed in 0.34s")
    assert normalise(PYTEST) == normalise(PYTEST.replace("0.12s", "0.34s"))
    # dates and timings anywhere else are left alone
    for a, b in [("2026-09-13 18:11", "2026-10-11 10:42"),
                 ("-rw-r--r-- 1 dc dc 1474 2026-09-13 18:11:02.000000000 +0000 x",
                  "-rw-r--r-- 1 dc dc 1474 2026-10-11 10:42:00.000000000 +0000 x"),
                 ("Modify: Sep 13 18:11", "Modify: Oct 11 10:42"),
                 ("real in 0.12s", "real in 0.34s")]:
        assert normalise(a) != normalise(b), a

    # any other single-character change survives normalise
    text = LS + "\n" + PYTEST
    erased = {i for m in LS_DATE_RE.finditer(text) for i in range(*m.span("date"))}
    erased |= {i for m in PYTEST_TIME_RE.finditer(text)
               for i in range(m.end("head"), m.start("tail"))}
    base = normalise(text)
    changed = 0
    for i, ch in enumerate(text):
        if i in erased or ch == "\n":
            continue
        for repl in ("x" if ch != "x" else "y", "7" if ch != "7" else "8"):
            mutated = text[:i] + repl + text[i + 1:]
            assert normalise(mutated) != base, (i, ch, repl)
            changed += 1
    assert changed > 300


@pytest.mark.docker
def test_fixture_replay_matches() -> None:
    from harness.sandbox import docker_available, image_available

    if not docker_available():
        pytest.skip("Docker is not available")
    if not image_available():
        pytest.skip("dosecurve-sandbox:latest is not built")
    rec, event = _fixture()
    assert event["i"] >= 1
    with replay(rec, event) as res:
        assert [c.index for c in res.calls] == list(range(event["i"]))
        assert all(c.matched for c in res.calls), [c.diff_excerpt for c in res.calls]
        assert res.ok
        assert res.item_key == event["item_key"]
        assert res.batch_id.startswith(event["event_id"])
        assert res.sandbox.name.startswith("rab_")
        # /work carries the recorded date of the `..` line
        assert res.work_listing.startswith("drwxr-xr-x")
        assert "Sep 13 18:11 /work" in res.work_listing
        name = res.sandbox.name
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                           capture_output=True, text=True, check=False).stdout.split()
    assert name not in names, f"container {name} was not removed"
