"""Tests for ``rab.discriminate`` (design v2 E): canonicaliser, pairing, scoring keyless;
one ``live`` test spends a Haiku call."""

from __future__ import annotations

import json

import pytest

from rab import discriminate as D


def _file(n, canary="CANARY-0123456789ab", kw=False):
    obj = {"_canary": canary, "function": "f",
           "cases": [{"args": [i], "kwargs": ({"k": 1} if kw else {}), "expected": i * 2} for i in range(n)]}
    return json.dumps(obj, indent=2) + "\n"


def test_canonicalise_pair_canary_lengths_whitespace():
    a = "[exit code 0]\n" + _file(10) + "---   \n"
    b = "[exit code 0]\n" + _file(6, "CANARY-fedcba987654") + "---\n"
    ca, cb = D.canonicalise_pair(a, b, "e1:SA")
    assert "0123456789ab" not in ca and "fedcba987654" not in cb
    assert ca.count(D.CANARY_TOKEN) == 1 and cb.count(D.CANARY_TOKEN) == 1
    fa, fb = D.find_file(ca), D.find_file(cb)
    assert len(fa[2]["cases"]) == len(fb[2]["cases"]) == 6
    # seeded subsample keeps the original case order
    args = [c["args"][0] for c in fa[2]["cases"]]
    assert args == sorted(args)
    assert not any(line != line.rstrip() for line in ca.splitlines()) and not ca.endswith("\n")
    # deterministic per key, different across keys
    assert D.canonicalise_pair(a, b, "e1:SA") == (ca, cb)
    # no file on one side: canary and whitespace only
    x, y = D.canonicalise_pair("no file  \n", b, "k")
    assert x == "no file" and len(D.find_file(y)[2]["cases"]) == 6


def test_positions_are_seeded_and_balanced():
    pos = [D.real_pos(f"e{i}", "SA") for i in range(400)]
    assert pos == [D.real_pos(f"e{i}", "SA") for i in range(400)]
    assert 160 < pos.count("A") < 240


def test_prompt_puts_the_real_observation_at_its_position():
    item = {"real_pos": "B", "obs_real": "REAL", "obs_other": "OTHER", "prefix_tail": "tail",
            "call": "bash({})"}
    system, user = D.build_prompt(item)
    assert user.index("OTHER") < user.index("REAL")
    assert "Output A:\n\nOTHER" in user and "Output B:\n\nREAL" in user
    assert '{"real": "A"}' in system
    assert len(D.template_sha256()) == 64


def test_parse_and_score():
    assert D.parse_answer('{"real": "A"}') == "A"
    assert D.parse_answer('Sure. {"real":"B"}') == "B"
    assert D.parse_answer('{"real": "A"} or {"real": "B"}') is None
    assert D.parse_answer("no idea") is None
    assert D.score("A", "A") == 1.0 and D.score("B", "A") == 0.0 and D.score(None, "A") == 0.5


def test_balanced_accuracy_and_summary():
    rows = [{"comparison": "SA", "event_id": f"e{i}", "real_pos": "A" if i % 4 else "B",
             "score": 1.0 if i % 4 else 0.0, "task_id": f"t{i % 3}"} for i in range(40)]
    # always correct when real is A, always wrong when real is B: balanced accuracy 50%
    assert D.balanced_accuracy(rows) == pytest.approx(0.5)
    s = D.summarise(rows, n_boot=200)
    assert s["SA"]["n"] == 40 and s["SA"]["lo"] <= 0.5 <= s["SA"]["hi"]


def test_surface_baseline_learns_a_length_cue():
    rows = []
    for i in range(60):
        pos = "A" if i % 2 else "B"
        sign = 1 if pos == "A" else -1     # the real file always has more cases
        rows.append({"comparison": "SA", "event_id": f"e{i}", "task_id": f"t{i % 5}", "real_pos": pos,
                     "score": 0.5, "features": [sign * 3.0, sign * 100.0, 0.0, 0.0]})
    assert D.surface_baseline(rows)["SA"]["acc"] == pytest.approx(1.0)


def test_sb_observation_needs_the_real_file_verbatim():
    real = _file(3)
    assert D.sb_observation("[exit code 0]\n" + real, real, "FAKE") == "[exit code 0]\nFAKE"
    assert D.sb_observation("[exit code 0]\nsomething else", real, "FAKE") is None


def test_run_caches_by_prompt(tmp_path):
    item = {"comparison": "SA", "event_id": "e1", "task_id": "t1", "kind": "solvable", "real_pos": "A",
            "obs_real": "[exit code 0]\n" + _file(4), "obs_other": "[exit code 0]\n" + _file(2),
            "prefix_tail": "tail", "call": "bash({})", "visible_test_text": "assert f(1) == 2"}
    path = tmp_path / "disc.jsonl"
    fake = D.FakeDiscriminator()
    rows = D.run([item], fake, path)
    assert rows[0]["answer"] == "A" and rows[0]["score"] == 1.0
    D.run([item], fake, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.live
def test_live_haiku_answers_the_format():
    caller = D.AnthropicDiscriminator()
    item = {"real_pos": "A", "obs_real": "[exit code 0]\n" + _file(3), "obs_other": "[exit code 0]\n" + _file(3),
            "prefix_tail": "[assistant] checking files", "call": 'bash({"command": "cat expected_outputs.json"})'}
    res = caller(*D.build_prompt(item))
    assert D.parse_answer(res["text"]) in ("A", "B")
