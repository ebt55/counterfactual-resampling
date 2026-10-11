"""Keyless tests for ``rab.analyze`` (design v2 D) on synthetic continuation files."""

from __future__ import annotations

import gzip
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from rab import analyze as A
from rab.constants import CLASSES


def _row(eid, arm, idx, klass, kind="unreachable", task="t1", **kw):
    base = {"event_id": eid, "arm": arm, "run_idx": idx, "kind": kind, "task_id": task, "klass": klass,
            "flagged": klass == "flagged_problem", "fitted": klass == "fitted_and_submitted",
            "reread": False, "named": False, "regrade_pass": False, "size_visible": False,
            "error_class": None, "provider_mismatch": False, "cost_usd": 0.001, "wall_s": 1.0,
            "fake_vs_reference": None, "fake_encodes_T": None, "notes_reveal_T": False}
    return {**base, **kw}


def _cell(n_events=12, shift=0.0, seed=1, kind="unreachable"):
    """R and SA with class probabilities that differ by ``shift`` on two classes."""
    rng = random.Random(seed)
    rows = []
    for e in range(n_events):
        eid, task = f"e{e:02d}", f"t{e % 4}"
        for arm, n, sh in (("R", 8, 0.0), ("SA", 4, shift), ("B", 2, 0.0), ("P", 2, 0.0)):
            for i in range(n):
                p = 0.5 + sh
                klass = "flagged_problem" if rng.random() < p else "submitted_clean"
                rows.append(_row(eid, arm, i, klass, kind=kind, task=task,
                                 regrade_pass=(arm == "R")))
    return rows


def test_tvd_and_shares():
    a = np.array([2, 2, 0, 0, 0, 0], dtype=float)
    b = np.array([0, 0, 4, 0, 0, 0], dtype=float)
    assert A.tvd(a, b) == pytest.approx(1.0)
    assert A.tvd(a, a) == pytest.approx(0.0)
    assert len(A.CLASSES_NO_ERROR) == len(CLASSES) - 1


def test_permutation_detects_a_shift_and_not_noise():
    same = _cell(shift=0.0)
    moved = _cell(shift=0.45)
    p_same = A.compare(same, "SA", n_perm=2000)["p"]
    p_moved = A.compare(moved, "SA", n_perm=2000)["p"]
    assert p_moved < 0.01 < p_same
    # seeded: the same input gives the same p
    assert A.compare(moved, "SA", n_perm=500)["p"] == A.compare(moved, "SA", n_perm=500)["p"]


def test_errors_are_excluded_from_distributions():
    rows = [_row("e1", "R", 0, "error"), _row("e1", "R", 1, "submitted_clean"),
            _row("e1", "SA", 0, "submitted_clean")]
    d = A.class_dist(rows, "R")
    assert d["n"] == 2 and d["n_ok"] == 1 and d["shares"]["submitted_clean"] == 1.0
    assert A.compare(rows, "SA", n_perm=100)["tvd"] == 0.0


def test_holm_sign_flip_wilson():
    adj = A.holm({"a": 0.01, "b": 0.04})
    assert adj == {"a": 0.02, "b": 0.04}
    assert math.isnan(A.holm({"a": float("nan")})["a"])
    # all 7 tasks positive: the exact two-sided p is 2/128
    assert A.sign_flip({f"t{i}": 0.1 for i in range(7)}) == pytest.approx(2 / 128)
    lo, hi = A.wilson(5, 10)
    assert lo < 0.5 < hi and A.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12)


def test_placebo_relabels_r_4_7_and_sets_sa_aside():
    rows = _cell(n_events=2)
    out = A.placebo(rows)
    assert not any(r["arm"] == "SA" and r["run_idx"] >= 4 for r in out)
    sa = [r for r in out if r["arm"] == "SA"]
    assert len(sa) == 2 * 4 and {r["run_idx"] for r in sa} == {0, 1, 2, 3}
    assert all(r["arm"] != "R" or r["run_idx"] < 4 for r in out)


def test_verdict_rules():
    # control: R regrade pass 100%, P 0% (gap 100 points); B moved in one cell
    rows = _cell(shift=0.45)
    for r in rows:
        if r["arm"] == "B":
            r["klass"] = "kept_probing"
    an = A.analyse(rows, n_perm=500)
    ctl = A.control(an)
    assert not ctl["failed"]
    assert A.verdict(an) == {"unreachable": "positive"}
    # control failed when P regrades as well as R
    for r in rows:
        if r["arm"] == "P":
            r["regrade_pass"] = True
    an = A.analyse(rows, n_perm=500)
    assert A.verdict(an) == {"unreachable": "control failed"}


def _write_root(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    root = tmp_path / "res"
    events = {}
    for r in rows:
        events[r["event_id"]] = {"event_id": r["event_id"], "kind": r["kind"], "task_id": r["task_id"],
                                 "notes_reveal_T": False}
        d = root / "continuations" / r["event_id"]
        d.mkdir(parents=True, exist_ok=True)
        transcript = [{"role": "system", "content": "rules", "tool_calls": [], "tool_call_id": None},
                      {"role": "user", "content": "manifest", "tool_calls": [], "tool_call_id": None},
                      {"role": "assistant", "content": "",
                       "tool_calls": [{"id": "c1", "name": "bash", "args": {"command": "cat expected_outputs.json"}}],
                       "tool_call_id": None},
                      {"role": "tool", "content": f"obs {r['arm']}", "tool_calls": [], "tool_call_id": "c1"}]
        slot = {"event_id": r["event_id"], "arm": r["arm"], "run_idx": r["run_idx"], "kind": r["kind"],
                "task_id": r["task_id"],
                "cls": {"klass": r["klass"], "flagged_problem": r["flagged"], "fitted": r["fitted"],
                        "regrade_pass": r["regrade_pass"]},
                "cont": {"cost_usd": 0.001, "transcript": transcript, "block_k": 3, "observation": "x",
                         "submitted": True, "submit_summary": "done"},
                "placed": {}}
        with gzip.open(d / f"{r['arm']}-{r['run_idx']}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(slot, fh)
    ev_path = tmp_path / "events.jsonl"
    ev_path.write_text("".join(json.dumps(e) + "\n" for e in events.values()), encoding="utf-8")
    return root, ev_path


def test_tables_rebuild_byte_for_byte_and_hand_read(tmp_path):
    rows = _cell(n_events=6, shift=0.2) + _cell(n_events=6, shift=0.0, seed=3, kind="solvable")
    for r in rows:
        if r["kind"] == "solvable":
            r["event_id"] = "s" + r["event_id"]
    root, ev = _write_root(tmp_path, rows)
    recount = tmp_path / "recount.json"
    recount.write_text(json.dumps({"replay_drops": {"unreachable": ["x"], "solvable": []}}), encoding="utf-8")
    first = A.run(root, n_perm=300, events_path=ev, recount_path=recount)["text"]
    second = A.run(root, n_perm=300, events_path=ev, recount_path=recount)["text"]
    assert first == second == (root / "tables.md").read_text(encoding="utf-8")
    assert "## Verdict" in first and "| replay exclusions (before the draw) | 0 | 1 |" in first
    meta = json.loads((root / "tables_meta.json").read_text(encoding="utf-8"))
    assert "generated" in meta and "generated" not in first
    key = json.loads((root / "hand_read_key.json").read_text(encoding="utf-8"))
    assert len(key) == 5 and all({key[k]["A"][:2], key[k]["B"][:2]} == {"R-", "SA"} for k in key)
    md = (root / "hand_read.md").read_text(encoding="utf-8")
    assert "Continuation A" in md and "## Item 5" in md
    pl = A.run(root, placebo_mode=True, n_perm=300, events_path=ev, recount_path=recount)
    assert (root / "tables_placebo.md").is_file() and set(pl["verdicts"]) == {"solvable", "unreachable"}


def test_exploratory_files_are_listed_and_excluded_by_default(tmp_path):
    from rab.run import git_state, slot_path

    rows = _cell(n_events=3, shift=0.0)
    root, ev = _write_root(tmp_path, rows)
    # one exploratory pilot copy of every slot of event e00, beside the headline files
    for r in [r for r in rows if r["event_id"] == "e00"]:
        src = root / "continuations" / "e00" / f"{r['arm']}-{r['run_idx']}.json.gz"
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            slot = json.load(fh)
        slot.update(exploratory=True, git={"git_hash": "abc123", "dirty": False})
        slot["cls"]["klass"] = "no_submit"
        dst = slot_path(root, "e00", r["arm"], r["run_idx"], exploratory=True)
        assert dst.name.endswith("-x.json.gz")
        with gzip.open(dst, "wt", encoding="utf-8") as fh:
            json.dump(slot, fh)
    recount = tmp_path / "recount.json"
    recount.write_text("{}", encoding="utf-8")
    out = A.run(root, n_perm=200, events_path=ev, recount_path=recount)["text"]
    assert "## Exploratory runs (excluded)" in out and "`abc123`" in out
    assert "| no_submit | 0 (0.0)" in out
    meta = json.loads((root / "tables_meta.json").read_text(encoding="utf-8"))
    assert meta["n_exploratory"] == 16 and set(meta["git"]) == {"git_hash", "dirty"}
    inc = A.run(root, n_perm=200, events_path=ev, recount_path=recount, include_exploratory=True)["text"]
    assert "## Exploratory runs (included above)" in inc and "| no_submit | 0 (0.0)" not in inc
    assert len(git_state()["git_hash"]) == 40
