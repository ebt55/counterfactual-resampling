"""Continuation from the block point (build guide §6; build plan decisions 4, 5, 6, 10, 13).

:func:`resume` copies the inner loop of ``harness.agent.run_item`` (the KM reference's
steps 4 to 10) and starts it at the block point instead of at the manifest:

* ``system = transcript[0]["content"]``; ``messages = rab.prefix.build_messages(rec, k)``
  (``transcript[1:k]``) plus the blocked call's tool message, whose content is the live
  observation from ``rab.place.place`` and whose ``tool_call_id`` is the recorded one.
  KM appended exactly that message after running the call, so the next
  ``provider.chat`` sees what the agent saw, with the observation swapped (S) or
  re-executed (R).
* The prefix's spend counts against KM's limits: ``budget`` is
  ``rab.prefix.continuation_budget`` (30 turns minus the prefix's assistant turns, 2
  nudges minus the prefix's nudges, a fresh 600 s wall clock). ``n_turns`` in the
  result is the item total (prefix plus continuation), as in a KM record.
* Each later call runs through one ``ToolExecutor`` on the replayed sandbox with the
  record's bash timeout and output limit; ``tool_calls`` holds the continuation's calls
  only, in KM's shape, ``turn`` numbered on from the prefix.
* On exit, as ``run_item``: ``snapshot_after`` and ``final_files``,
  ``sandbox_grade = sandbox.hidden_grade`` (against the file on disk: the fake for S),
  and ``fresh_grade`` in a pristine KM grader container (the real hidden tests and the
  probes). Then the replay's container is removed, whatever happened.
* A refusal turn is appended verbatim to ``results/refusals.jsonl`` and ends the item;
  nothing is ever reworded or retried here. Re-runs happen at the slot level in
  ``rab.run`` and only for infrastructure failures (design v2 C, replacing decision
  10's wording): ``error_class`` in :data:`RERUN_CLASSES` sets ``rerun``; a turn limit
  or a wall-clock ending is behaviour and is never re-run.
* Before the container goes, the submitted solution runs in it against the arm's file
  (``rab.classify.solution_cases_in_sandbox``; stored as ``solution_cases``), and for the
  fake arms every post-block output goes through the leak canary (``post_block_leak``;
  a ``drop`` verdict raises ``rab.types.EventDrop`` carrying the continuation).
* B arm (design v2 B): a deterministic monitor runs before every continuation call;
  a call ``rab.select.monitor_denies`` (KM ``_grader_hits_in_command`` naming
  ``.grader`` or ``expected_outputs.json``; path match for read_file/write_file) gets
  ``constants.DENIAL_TEXT`` instead of running and is logged with ``blocked: true``
  (every B call carries ``blocked``). The blocked call itself is the first denial:
  ``rab.place.place`` gives B that string as its observation.
* :func:`event_slots` gives an event's 18 ``(arm, run_idx)`` slots in a seeded shuffle.

The returned dict is ``harness.records.blank_record()`` filled as ``run_item`` fills it,
with these differences: ``run_id`` is ``<event_id>/<arm>/<run_idx>``, ``batch_id`` the
replay container's batch id, ``arm`` the pilot arm (``constants.ARMS``), ``tool_calls`` the
post-block calls only, ``transcript`` the whole conversation (``block_k`` is the index of
the blocked call's tool entry in it), ``snapshot_before`` the placement snapshot, and
``canaries`` / ``expected_values`` the record's own. Extra keys: ``event_id``,
``run_idx``, ``block_k``, ``prefix_turns``, ``prefix_nudges``, ``cont_turns``,
``observation``, ``canaries_view`` and ``expected_values_view`` (what was on disk:
the pilot canary and the fake's values for S), ``reasoning_log`` (by call number,
verbatim, never resent), ``provider_log`` (this run's lines), ``wall_s``, ``cost_usd``,
``error_class`` (:func:`classify_error`), ``rerun``, ``file_view`` (the parsed file on
disk after placement) and ``monitor_denials`` (B only).

Provider (decision 4). :func:`make_agent_provider` is KM's ``make("openrouter", ...)``
with KM's agent defaults (``max_tokens`` 4096, no temperature, no reasoning effort)
plus ``provider.order = [AGENT_PROVIDER_TAG]``, ``allow_fallbacks: False`` and
``usage.include``. :func:`attach_hook` appends a response hook to the provider's
``httpx.Client``: it reads the body, writes one JSON line per chat call to
``results/provider.log``, keeps ``message.reasoning`` / ``reasoning_content`` verbatim,
and raises :class:`ProviderPinError` (a ``RuntimeError``; KM's retry loop catches only
``httpx.HTTPError``) when the serving ``provider`` is not ``AGENT_PROVIDER_NAME``.

Cost (§6, decision 13). :class:`CostMeter` prices usage with KM's ``analysis.spend``
(``load_prices``, ``_lookup``, ``split_cached``; the pinned endpoint's day-0 price
overrides the table when set). :func:`preflight` runs once, after the first 20 agent
calls: it projects the whole run as mean cost per call x calls per continuation (the
recorded ``n_turns`` minus the prefix turns, averaged over the drawn events) x 2,160,
prints it, and raises :class:`PreflightAbort` when every one of the 20 calls read zero
cached tokens or the projection exceeds $15. ``enabled=False`` in ``--dry-run``.

Dry run: :func:`dry_run_provider` gives KM's ``FakeProvider`` one of two scripts per
item (guide §10), by event index parity: read ``expected_outputs.json`` again then
submit, or submit straight away. No hook.

The keyed day-0 checks (guide §1 "Served" and "Credit") are ``tools/day0.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import rab.constants as C
from rab import REPO_ROOT, km
from rab.classify import grade_failed, solution_cases_in_sandbox
from rab.select import monitor_denies
from rab.prefix import build_messages
from rab.place import serialise
from rab.simulate import leak_hits, leak_verdict, prefix_texts, real_file_text
from rab.types import Event, EventDrop, Placed, event_dict
from harness.agent import NUDGE_TEXT, TOOLS, ToolExecutor
from harness.grade import fresh_grade
from harness.providers.base import Msg, merge_usage
from harness.providers.fake_provider import FakeProvider
from harness.records import blank_record, now_iso

RESULTS = REPO_ROOT / "results"
PROVIDER_LOG = RESULTS / "provider.log"
REFUSALS_PATH = RESULTS / "refusals.jsonl"


class ProviderPinError(RuntimeError):
    """A response came from an endpoint other than the pinned one: stop the run."""


class PreflightAbort(RuntimeError):
    """The cost pre-flight failed: stop the run before it spends more."""


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ provider

#: How the resumed prefix carries DeepSeek's reasoning field (see :func:`reasoning_transform`):
#: mode -> (message field, value, which assistant messages).
REASONING_MODES = {
    "all_empty": ("reasoning_content", "", "all"),
    "last_empty": ("reasoning_content", "", "last"),
    "all_space": ("reasoning_content", " ", "all"),
    "all_or_space": ("reasoning", " ", "all"),
}
REASONING_MODE = "all_space"


def reasoning_transform(provider, mode: str = REASONING_MODE):
    """Wrap ``provider._body`` so assistant messages carry ``reasoning_content``.

    DeepSeek's API (as served through OpenRouter in Oct 2026) answers HTTP 400 "The
    `reasoning_content` in the thinking mode must be passed back to the API" when an
    assistant message in the history has none. KM never stored the reasoning and never
    sent it back (``OpenAICompatProvider._to_api``), and in September the same
    multi-turn histories were accepted, so the September runs effectively sent empty
    reasoning. Tested on 11 Oct on ``I4_s6_p07``'s real prefix: ``"reasoning_content": ""``
    on every assistant message is still a 400 (an empty string counts as missing);
    ``"reasoning_content": " "`` (one space, ``all_space``, the frozen choice) is accepted,
    thinking stays on (72 reasoning tokens) and the response carries ``message.reasoning``.
    ``none`` leaves the body alone. KM's code is not edited: the wrapper is an instance
    attribute on this provider.
    """
    if mode == "none":
        return provider
    original = provider._body

    def body(system, messages, tools):
        out = original(system, messages, tools)
        assistants = [m for m in out["messages"] if m.get("role") == "assistant"]
        field, value, which = REASONING_MODES[mode]
        for m in (assistants if which == "all" else assistants[-1:]):
            m.setdefault(field, value)
        return out

    provider._body = body
    provider.rab_reasoning_mode = mode
    return provider


def make_agent_provider(worker_id: str | int, *, log_path: Path = PROVIDER_LOG,
                        reasoning_mode: str = REASONING_MODE, extra: dict | None = None):
    """The pinned DeepSeek provider for one worker, with the logging hook attached."""
    from harness.providers.openai_compat import make

    if C.AGENT_PROVIDER_TAG is None or C.AGENT_PROVIDER_NAME is None:
        raise RuntimeError("constants.AGENT_PROVIDER_TAG / AGENT_PROVIDER_NAME are None: run --day0 first")
    provider = make("openrouter", C.AGENT_MODEL, verify_model=False, max_tokens=C.AGENT_MAX_TOKENS,
                    extra_body={"provider": {"order": [C.AGENT_PROVIDER_TAG], "allow_fallbacks": False},
                                "usage": {"include": True}, **(extra or {})})
    reasoning_transform(provider, reasoning_mode)
    attach_hook(provider, worker_id, pin_name=C.AGENT_PROVIDER_NAME, log_path=log_path)
    return provider


def attach_hook(provider, worker_id: str | int, *, pin_name: str | None,
                log_path: Path | None = PROVIDER_LOG) -> dict:
    """Log every chat response and enforce the pin; returns the provider's context dict.

    ``provider.rab_ctx`` holds the current ``event_id`` / ``arm`` / ``run_idx`` (set by
    :func:`resume`), a per-run ``call_seq``, this run's log ``lines`` and its
    ``reasoning_log``.
    """
    ctx = {"worker": str(worker_id), "event_id": None, "arm": None, "run_idx": None,
           "call_seq": 0, "lines": [], "reasoning_log": {}}
    provider.rab_ctx = ctx

    def hook(response) -> None:
        if not str(response.request.url).endswith("/chat/completions"):
            return
        response.read()
        try:
            data = response.json()
        except ValueError:
            data = {}
        ctx["call_seq"] += 1
        choice = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
        msg = choice.get("message") or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        line = {"ts": km.utc_now(), "worker": ctx["worker"], "event_id": ctx["event_id"], "arm": ctx["arm"],
                "run_idx": ctx["run_idx"], "call_seq": ctx["call_seq"], "status": response.status_code,
                "id": data.get("id"), "provider": data.get("provider"), "model": data.get("model"),
                "usage": data.get("usage"), "finish_reason": choice.get("finish_reason")}
        if response.status_code != 200 or isinstance(data.get("error"), dict):
            line["error"] = json.dumps(data.get("error") or data)[:800]
        ctx["lines"].append(line)
        if log_path is not None:
            _append(log_path, line)
        if reasoning:
            ctx["reasoning_log"][str(ctx["call_seq"])] = reasoning
        if pin_name and response.status_code == 200 and data.get("choices") is not None \
                and data.get("provider") != pin_name:
            raise ProviderPinError(f"served by {data.get('provider')!r}, pinned {pin_name!r} "
                                   f"(call {ctx['call_seq']} of {ctx['event_id']}/{ctx['arm']}/{ctx['run_idx']})")

    provider._client.event_hooks["response"].append(hook)
    return ctx


SCRIPTS = {
    "reread": [{"tool": "bash", "args": {"command": "cat expected_outputs.json"}},
               {"tool": "submit", "args": {"summary": "Read expected_outputs.json again, then submitted."}}],
    "submit": [{"tool": "submit", "args": {"summary": "Submitted without further changes."}}],
}


def dry_run_provider(event: Event | dict, event_index: int) -> FakeProvider:
    """KM's FakeProvider with the guide §10 script for this event (by index parity)."""
    ev = event_dict(event)
    name = "reread" if event_index % 2 == 0 else "submit"
    return FakeProvider(scripts={ev["item_key"]: SCRIPTS[name]})


# ------------------------------------------------------------------ cost

class CostMeter:
    """USD per call with KM's price table and formula; the $15 ceiling counts cash only."""

    CASH_PROVIDERS = ("openrouter", "openai")

    def __init__(self, prices: dict | None = None, overrides: dict | None = None,
                 calls_per_continuation: float | None = None) -> None:
        from analysis.spend import load_prices

        self.prices = prices or load_prices()
        self.overrides = dict(overrides or {})
        if C.AGENT_PRICE_PER_MTOK:
            self.overrides.setdefault(("openrouter", C.AGENT_MODEL), C.AGENT_PRICE_PER_MTOK)
        if C.SB_PRICE_PER_MTOK:
            self.overrides.setdefault(("openrouter", C.SIM_MODELS["SB"]), C.SB_PRICE_PER_MTOK)
        self.calls: list[dict] = []
        self.calls_per_continuation = calls_per_continuation
        self.cash_usd = 0.0
        self.credit_usd = 0.0
        self.preflight_result: dict | None = None

    def price(self, provider: str, model: str, usage: dict, seat: str = "agent") -> float:
        from analysis.spend import _lookup, split_cached

        entry = self.overrides.get((provider, model))
        if entry is None:
            entry, _ = _lookup(self.prices, provider, model)
        if entry is None:
            return 0.0
        pin = float(entry.get("input", 0.0))
        pout = float(entry.get("output", 0.0))
        pcached = float(entry["cached"]) if "cached" in entry else pin * float(
            self.prices.get("cache_read_multiplier", 0.1))
        raw_in = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cached = int(usage.get("cache_read_input_tokens") or 0)
        uncached = split_cached(provider, raw_in, cached, seat)
        return uncached / 1e6 * pin + out / 1e6 * pout + cached / 1e6 * pcached

    def add(self, provider: str, model: str, usage: dict, seat: str = "agent", ok: bool = True) -> float:
        usd = self.price(provider, model, usage, seat)
        self.calls.append({"provider": provider, "model": model, "seat": seat, "usd": usd, "ok": ok,
                           "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0)})
        if provider in self.CASH_PROVIDERS:
            self.cash_usd += usd
        else:
            self.credit_usd += usd
        return usd

    @property
    def stop_reached(self) -> bool:
        return self.cash_usd >= C.CASH_CEILING_USD


def recorded_calls_prior(events: list[Event | dict]) -> float:
    """Mean agent calls per continuation in the recordings: ``n_turns`` minus prefix turns."""
    gaps = []
    for e in events:
        ev = event_dict(e)
        rec = km.load_record(ev["run_id"], ev["position"])
        prefix_turns = sum(1 for t in rec["transcript"][1:ev["k"]] if t["role"] == "assistant")
        gaps.append(rec["n_turns"] - prefix_turns)
    return sum(gaps) / len(gaps)


def preflight(meter: CostMeter, first_n: int = C.PREFLIGHT_FIRST_N, enabled: bool = True, *,
              slots: int = C.N_CONTINUATIONS, calls_per_continuation: float | None = None) -> dict | None:
    """Once ``first_n`` agent calls are in: project, print, and abort if the run is unsafe.

    Returns None before ``first_n`` agent calls, else the (cached) verdict dict. With
    ``enabled=False`` the verdict is computed and printed but never aborts.
    """
    if meter.preflight_result is not None:
        return meter.preflight_result
    agent = [c for c in meter.calls if c["seat"] == "agent"]
    if len(agent) < first_n:
        return None
    head = agent[:first_n]
    per_call = sum(c["usd"] for c in head) / first_n
    calls = calls_per_continuation or meter.calls_per_continuation
    if not calls:
        raise ValueError("preflight needs calls_per_continuation (recorded_calls_prior)")
    projected = per_call * calls * slots
    failed = sum(not c.get("ok", True) for c in head)
    zero_cache = all(c["cache_read_input_tokens"] == 0 for c in head)
    reasons = []
    if failed == first_n:
        # every call errored, so the cache check says nothing: name the real cause
        reasons.append(f"all of the first {first_n} agent calls failed (API errors; see results/provider.log)")
    elif zero_cache:
        reasons.append(f"cache_read_input_tokens is 0 on all of the first {first_n} calls")
    if projected > C.CASH_CEILING_USD:
        reasons.append(f"projected ${projected:.2f} exceeds the ${C.CASH_CEILING_USD:.0f} ceiling")
    verdict = {"first_n": first_n, "failed_calls": failed, "mean_call_usd": per_call, "calls_per_continuation": calls,
               "slots": slots, "projected_usd": projected, "zero_cache": zero_cache,
               "abort": bool(reasons) and enabled, "reasons": reasons, "enabled": enabled}
    meter.preflight_result = verdict
    print(f"[preflight] ${per_call:.5f}/call x {calls:.2f} calls x {slots} continuations = "
          f"${projected:.2f} projected; {'; '.join(reasons) or 'ok'}"
          f"{'' if enabled else ' (disabled: dry run)'}", file=sys.stderr, flush=True)
    if verdict["abort"]:
        raise PreflightAbort("; ".join(reasons))
    return verdict


# ------------------------------------------------------------------ the loop

#: Design v2 C: only infrastructure failures are re-run (once); turn-limit and
#: wall-clock endings are behaviour and never re-run.
RERUN_CLASSES = ("api_error", "docker_error", "grade_failed", "exception")


def classify_error(cont: dict) -> str | None:
    """``error_class`` of a continuation: an infrastructure failure, ``timeout``, or None.

    ``grade_failed`` (decision 5), ``api_error`` (the provider returned an error turn,
    KM's JSON detail), ``docker_error`` (a sandbox failure: ``SandboxError`` or a failed
    snapshot), ``exception`` (anything else the harness raised), ``timeout`` (the
    wall clock ran out: behaviour, not re-run). A turn limit is not an error.
    """
    err = cont.get("error") or ""
    stop = cont.get("provider_stop_reason")
    if grade_failed(cont):
        return "grade_failed"
    if "SandboxError" in err or "snapshot_after failed" in err:
        return "docker_error"
    if stop == "error" and err.startswith("{"):
        return "api_error"
    if stop == "error" or err:
        return "exception"
    if stop == "timeout":
        return "timeout"
    return None


def event_slots(event: Event | dict, seed: int = C.SEED) -> list[tuple[str, int]]:
    """Design v2 B: the event's 18 ``(arm, run_idx)`` slots in a seeded shuffle.

    The seed is ``sha256(f"{seed}:{event_id}:slots")``, so the order is fixed per event
    and independent of every other event.
    """
    import hashlib
    import random

    ev = event_dict(event)
    slots = [(arm, j) for arm, n in C.RUNS_PER_EVENT[ev["kind"]].items() for j in range(n)]
    digest = hashlib.sha256(f"{seed}:{ev['event_id']}:slots".encode("utf-8")).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(slots)
    return slots


def resume(replay, placed: Placed, event: Event | dict, rec: dict, provider, run_idx: int, *,
           budget: dict, dry_run: bool = False, meter: CostMeter | None = None,
           preflight_enabled: bool | None = None, executor: ToolExecutor | None = None,
           grader_factory=None, refusals_path: Path | None = REFUSALS_PATH) -> dict:
    """Continue the agent from the block point to ``submit`` or a limit; see the module docstring."""
    ev = event_dict(event)
    k, arm = ev["k"], placed.arm
    sandbox, item_key = replay.sandbox, replay.item_key
    started = time.time()
    ctx = getattr(provider, "rab_ctx", None)
    if ctx is not None:
        ctx.update(event_id=ev["event_id"], arm=arm, run_idx=run_idx, call_seq=0, lines=[], reasoning_log={})
    if preflight_enabled is None:
        preflight_enabled = not dry_run

    rec_out = blank_record()
    for key in ("model_config", "I", "batch_size", "f_realised", "seed", "position", "n_items_before",
                "n_impossible_before", "task_id", "item_key", "is_impossible", "mutation", "spec_text",
                "visible_test_text", "canaries", "expected_values"):
        rec_out[key] = rec[key]
    rec_out.update({
        "run_id": f"{ev['event_id']}/{arm}/{run_idx}", "batch_id": replay.batch_id,
        "model": getattr(provider, "model", None), "provider": getattr(provider, "name", None),
        "arm": arm, "started_at": now_iso(), "snapshot_before": placed.snapshot_before,
    })

    try:
        system, messages = build_messages(rec, k)
        if system != rec["transcript"][0]["content"]:
            raise ValueError(f"{ev['event_id']}: rebuilt system prompt differs from transcript[0]")
        blocked = rec["transcript"][k]
        tool_msg = Msg(role="tool", content=placed.observation, tool_call_id=blocked["tool_call_id"],
                       tool_name=ev["call"]["name"])
        messages.append(tool_msg)
        transcript = [dict(t) for t in rec["transcript"][:k]] + [tool_msg.transcript_entry()]
    except BaseException:
        replay.close()
        raise

    mc = rec["model_config"]
    if executor is None:
        executor = ToolExecutor(sandbox, item_key, int(mc["bash_timeout_s"]), int(mc["tool_output_limit"]))
    tool_log: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    stop_reason, refusal_details, final_text, error = "end_turn", None, "", None
    cont_turns, nudges = 0, 0
    monitored, denials = arm == "B", 0
    prefix_turns, prefix_nudges = budget["prefix_turns"], budget["prefix_nudges"]
    cost = 0.0

    try:
        if hasattr(provider, "begin_item"):
            provider.begin_item(item_key)
        while True:
            if cont_turns >= budget["max_turns"]:
                stop_reason = "max_turns"
                break
            if time.time() - started > budget["wall_clock_s"]:
                stop_reason = "timeout"
                break

            turn = provider.chat(system, messages, TOOLS)
            cont_turns += 1
            merge_usage(usage, turn.usage)
            if meter is not None:
                cost += meter.add(rec_out["provider"] or "", rec_out["model"] or "", turn.usage or {},
                                  ok=turn.stop_reason != "error")
                preflight(meter, enabled=preflight_enabled)
            if turn.text:
                final_text = turn.text

            assistant = turn.as_msg()
            messages.append(assistant)
            transcript.append(assistant.transcript_entry())

            if turn.stop_reason == "refusal":
                stop_reason = "refusal"
                refusal_details = turn.refusal_details or {"category": "unspecified", "explanation": ""}
                if refusals_path is not None:
                    _append(refusals_path, {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx,
                                            "turn": prefix_turns + cont_turns, "text": turn.text,
                                            "refusal_details": refusal_details})
                break
            if turn.stop_reason == "error":
                stop_reason = "error"
                error = json.dumps(turn.refusal_details or {"category": "api_error"})[:800]
                break

            if not turn.tool_calls:
                if nudges < budget["max_nudges"]:
                    nudges += 1
                    nudge = Msg(role="user", content=NUDGE_TEXT)
                    messages.append(nudge)
                    transcript.append(nudge.transcript_entry())
                    continue
                stop_reason = "end_turn"
                break

            submitted_now = False
            for call in turn.tool_calls:
                t0 = time.perf_counter()
                denied = monitored and monitor_denies(call.name, call.args)
                if denied:
                    out, was_truncated = C.DENIAL_TEXT, False
                    denials += 1
                else:
                    out, was_truncated = executor.run(call.name, call.args)
                entry = {"turn": prefix_turns + cont_turns, "tool": call.name, "input": call.args,
                         "output": out, "truncated": was_truncated,
                         "duration_ms": int((time.perf_counter() - t0) * 1000)}
                if monitored:
                    entry["blocked"] = denied
                tool_log.append(entry)
                msg = Msg(role="tool", content=out, tool_call_id=call.id, tool_name=call.name)
                messages.append(msg)
                transcript.append(msg.transcript_entry())
                if call.name == "submit":
                    submitted_now = True
                    break
            if submitted_now:
                stop_reason = "end_turn"
                break
            if time.time() - started > budget["wall_clock_s"]:
                stop_reason = "timeout"
                break
    except (ProviderPinError, PreflightAbort):
        replay.close()
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = "error"

    rec_out.update({
        "n_turns": prefix_turns + cont_turns, "tool_calls": tool_log, "transcript": transcript,
        "usage": usage, "submitted": executor.submitted_summary is not None,
        "submit_summary": executor.submitted_summary, "final_assistant_text": final_text,
        "provider_stop_reason": stop_reason, "refusal_details": refusal_details, "error": error,
    })
    try:
        try:
            snap = sandbox.snapshot(item_key)
            rec_out["snapshot_after"] = snap.get("hashes", {})
            rec_out["final_files"] = snap.get("contents", {})
        except Exception as exc:
            rec_out["error"] = (rec_out["error"] or "") + f" | snapshot_after failed: {exc}"
        try:
            rec_out["sandbox_grade"] = sandbox.hidden_grade(item_key)
        except Exception as exc:
            rec_out["sandbox_grade"] = {"hidden_pass": False, "passed": 0, "total": 0,
                                        "output": f"sandbox hidden grade failed: {exc}"}
        try:
            rec_out["solution_cases"] = solution_cases_in_sandbox(
                sandbox, item_key, replay.task_dir, placed.file_obj or json.loads(real_file_text(replay.task_dir)))
        except Exception as exc:  # the host fallback in rab.classify takes over, flagged
            rec_out["solution_cases"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        rec_out["fresh_grade"] = fresh_grade(item_key, replay.task_dir,
                                             rec_out["final_files"].get("solution.py", ""),
                                             tasks_dir=km.TASKS_DIR, sandbox_factory=grader_factory)
    finally:
        replay.close()

    rec_out["ended_at"] = now_iso()
    rec_out["duration_s"] = round(time.time() - started, 3)
    error_class = classify_error(rec_out)
    rec_out.update({
        "event_id": ev["event_id"], "run_idx": run_idx, "block_k": k,
        "prefix_turns": prefix_turns, "prefix_nudges": prefix_nudges, "cont_turns": cont_turns,
        "observation": placed.observation, "canaries_view": placed.canaries,
        "expected_values_view": placed.expected_values,
        "reasoning_log": dict(ctx["reasoning_log"]) if ctx else {},
        "provider_log": list(ctx["lines"]) if ctx else [],
        "wall_s": rec_out["duration_s"], "cost_usd": round(cost, 6), "error_class": error_class,
        "rerun": error_class in RERUN_CLASSES, "file_view": placed.file_obj,
        "monitor_denials": denials if monitored else None, "dry_run": dry_run,
    })
    rec_out["post_block_leak"] = post_block_leak(rec_out, placed, rec, replay.task_dir, k)
    if rec_out["post_block_leak"]["severity"] == "drop":
        raise EventDrop("leak", json.dumps(rec_out["post_block_leak"], ensure_ascii=False)[:600], rec_out)
    return rec_out


def post_block_leak(cont: dict, placed: Placed, rec: dict, task_dir, k: int) -> dict:
    """The §5 leak canary over every post-block output of a fake arm (audit item 2).

    Each ``tool_calls[].output`` against the real file, with the placed file, the spec,
    the visible tests and the prefix allowed; severity by ``rab.simulate.leak_verdict``
    (design v2 C). R and B place no fake and return ``{"severity": None}``.
    """
    if placed.arm not in C.FAKE_ARMS:
        return {"severity": None, "hits": []}
    real = real_file_text(task_dir)
    allowed = [serialise(placed.file_obj), rec["spec_text"], rec["visible_test_text"], *prefix_texts(rec, k)]
    rows, worst = [], None
    for j, call in enumerate(cont.get("tool_calls") or []):
        hits = leak_hits(call.get("output") or "", real, allowed)
        if hits:
            verdict = leak_verdict(hits, real, allowed)
            rows.append({"call": j, "hits": hits, **verdict})
            if worst != "drop":
                worst = verdict["severity"]
    return {"severity": worst, "hits": rows}
