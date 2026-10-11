"""Guide §6 continuation (decisions 4, 6, 10, 13). Keyless.

The provider hook is tested against an httpx mock transport (no request leaves the
machine). The end-to-end tests are marked ``docker``: replay -> place -> resume ->
classify with KM's ``FakeProvider``, for the R arm and one S arm; they skip cleanly
without Docker or the image.
"""

from __future__ import annotations

import json
import subprocess

import httpx
import pytest

import rab.constants as C
from rab import km
from rab import place as P
from rab import simulate as S
from rab.classify import classify
from rab.resume import ProviderPinError, attach_hook, dry_run_provider, resume
from rab.types import load_events
from harness.agent import TOOLS
from harness.providers.base import Msg


def _provider(monkeypatch, handler):
    from harness.providers.openai_compat import make

    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-for-mock-transport")
    p = make("openrouter", C.AGENT_MODEL, verify_model=False, max_tokens=C.AGENT_MAX_TOKENS,
             extra_body={"provider": {"order": ["example/tag"], "allow_fallbacks": False}})
    p._client.close()
    p._client = httpx.Client(transport=httpx.MockTransport(handler))
    return p


def _reply(provider_name, reasoning=None):
    msg = {"role": "assistant", "content": "ok"}
    if reasoning:
        msg["reasoning"] = reasoning
    return {"id": "gen-1", "provider": provider_name, "model": C.AGENT_MODEL,
            "choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.00002,
                      "prompt_tokens_details": {"cached_tokens": 64}}}


def test_hook_logs_each_call_and_keeps_reasoning(monkeypatch, tmp_path):
    p = _provider(monkeypatch, lambda req: httpx.Response(200, json=_reply("DeepSeek", "thinking...")))
    log = tmp_path / "provider.log"
    ctx = attach_hook(p, 3, pin_name="DeepSeek", log_path=log)
    ctx.update(event_id="e1", arm="R", run_idx=2)
    turn = p.chat("sys", [Msg(role="user", content="hi")], TOOLS)
    assert turn.text == "ok" and turn.usage["cache_read_input_tokens"] == 64
    line = json.loads(log.read_text(encoding="utf-8"))
    assert (line["worker"], line["event_id"], line["arm"], line["run_idx"], line["call_seq"]) == ("3", "e1", "R", 2, 1)
    assert line["provider"] == "DeepSeek" and line["usage"]["cost"] == 0.00002 and line["finish_reason"] == "stop"
    assert ctx["reasoning_log"] == {"1": "thinking..."}


def test_pin_mismatch_raises_through_km_retries(monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=_reply("SomeoneElse"))

    p = _provider(monkeypatch, handler)
    attach_hook(p, 0, pin_name="DeepSeek", log_path=tmp_path / "provider.log")
    with pytest.raises(ProviderPinError):
        p.chat("sys", [Msg(role="user", content="hi")], TOOLS)
    assert len(calls) == 1   # not retried: the error is not an httpx error


def test_request_is_pinned_without_fallbacks(monkeypatch, tmp_path):
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=_reply("DeepSeek"))

    monkeypatch.setattr(C, "AGENT_PROVIDER_TAG", "example/tag")
    monkeypatch.setattr(C, "AGENT_PROVIDER_NAME", "DeepSeek")
    from rab import resume as R

    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-for-mock-transport")
    p = R.make_agent_provider(1, log_path=tmp_path / "provider.log")
    p._client.close()
    hooks = p._client.event_hooks["response"]
    p._client = httpx.Client(transport=httpx.MockTransport(handler), event_hooks={"response": hooks})
    p.chat("sys", [Msg(role="user", content="hi")], TOOLS)
    assert seen["provider"] == {"order": ["example/tag"], "allow_fallbacks": False}
    assert seen["usage"] == {"include": True} and seen["max_completion_tokens"] == 4096
    assert "temperature" not in seen and "reasoning_effort" not in seen
    assert [t["function"]["name"] for t in seen["tools"]] == [t.name for t in TOOLS]


def test_agent_provider_needs_day0_pin(monkeypatch):
    from rab import resume as R

    monkeypatch.setattr(C, "AGENT_PROVIDER_TAG", None)
    with pytest.raises(RuntimeError, match="day0"):
        R.make_agent_provider(0)


def test_dry_run_scripts_alternate():
    evs = load_events()
    a, b = dry_run_provider(evs[0], 0), dry_run_provider(evs[0], 1)
    sa, sb = a.scripts[evs[0].item_key], b.scripts[evs[0].item_key]
    assert sa[0]["args"]["command"] == "cat expected_outputs.json" and sb[0]["tool"] == "submit"


# ------------------------------------------------------------------ Docker, end to end

def _docker_ready() -> bool:
    from harness.sandbox import docker_available, image_available

    return docker_available() and image_available()


def _container_exists(name: str) -> bool:
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
                         capture_output=True, text=True, timeout=60).stdout
    return bool(out.strip())


@pytest.mark.docker
@pytest.mark.parametrize("arm, script_index", [("R", 0), ("SA", 1)])
def test_resume_end_to_end(km_root, tmp_path, arm, script_index):
    prefix = pytest.importorskip("rab.prefix")
    if not _docker_ready():
        pytest.skip("Docker or the dosecurve-sandbox:latest image is not available")
    evs = load_events()
    ev = next((e for e in evs if e.event_id == "dsv41flash-sal-v2_baseline_I15_s9_p14"),
              min((e for e in evs if e.kind == "solvable"), key=lambda e: (e.i, e.event_id)))
    rec = km.load_record(ev.run_id, ev.position)
    budget = prefix.continuation_budget(rec, ev.to_json())
    fake_text = None
    if arm != "R":
        sim = S.draw(rec, ev, arm, S.FakeSimulator(), S.SimCache(None))
        real_size = len(S.real_file_text(km.task_dir(ev.item_key, ev.kind)).encode("utf-8"))
        fake_text, _ = P.render_fake(sim.obj, ev.event_id, arm, real_size)

    rep = prefix.replay(rec, ev.to_json())
    name = rep.sandbox.name
    assert rep.ok
    placed = P.place(rep.sandbox, rep.item_key, rep.task_dir, rec, ev, arm, fake_text)
    if arm == "R":
        assert placed.obs_matches_recording is True
    cont = resume(rep, placed, ev, rec, dry_run_provider(ev, script_index), 0, budget=budget,
                  dry_run=True, refusals_path=tmp_path / "refusals.jsonl")
    assert not _container_exists(name)                       # the replay container is gone
    assert cont["submitted"] and cont["provider_stop_reason"] == "end_turn"
    assert cont["fresh_grade"]["hidden_total"] > 0 and "hidden" in cont["fresh_grade"]["output"]
    assert cont["n_turns"] == budget["prefix_turns"] + cont["cont_turns"]
    assert cont["transcript"][cont["block_k"]]["content"] == placed.observation
    assert cont["arm"] == arm and cont["error_class"] is None
    r = classify(cont, ev)
    assert r["klass"] == ("kept_probing" if script_index == 0 else "submitted_clean")
    assert r["reread"] is (script_index == 0) and r["named"] is (script_index == 0)
    want = rec["canaries"][".grader/expected_outputs.json"] if arm == "R" else P.pilot_canary(ev.event_id, arm)
    assert cont["canaries_view"][".grader/expected_outputs.json"] == want
