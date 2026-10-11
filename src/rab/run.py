"""The run: every slot of every event, resumable (design v2 B, C, D; guide §6, §10).

``python -m rab.run [--dry-run] [--events N] [--workers 4] [--pilot 5] [--root DIR]``

Order. Events are taken in a seeded order (:func:`event_order`), event-major: an event's
slots all finish before the next event starts, so a stop (cost ceiling, pre-flight
abort, Ctrl-C) leaves whole events. ``--events N`` keeps the first N events of each
kind in that order; ``--pilot N`` keeps the first N events overall (design v2 F: the
5-event pilot). Within an event the slots are ``rab.resume.event_slots`` (18, from
``constants.RUNS_PER_EVENT``, in a seeded shuffle) and run on ``--workers`` threads.

Per event, before its slots: the simulator draws (``rab.simulate.draw``) for SA and SB,
cached in ``<root>/sims.jsonl``. A prompt leak of severity ``drop`` drops the event from
every arm (``<root>/leaks.jsonl``). An SA draw that failed (schema twice, refusal)
drops SA and SSTAR for that event only; the other arms run.

Per slot (:func:`run_slot`): replay (``rab.prefix.replay``, a fresh container with a
unique batch id) -> the arm's file (``rab.place.arm_file``) -> ``rab.place.place`` ->
``rab.resume.resume`` (closes the container, runs both grades) ->
``rab.classify.classify``. One gzip JSON file per slot,
``<root>/continuations/<event_id>/<arm>-<run>.json.gz``, written atomically; an
existing file is skipped (resumable). One ``<root>/run_log.jsonl`` line per slot
attempt and per event drop.

Re-runs (design v2 C, decision 6): a prefix that does not replay ``ok``, or an R
blocked call whose live output is not ``content_equal`` to the recording, is retried
once in a fresh container; a second divergence drops the event from every arm. A
continuation whose ``rerun`` is set (API, Docker, grade or harness failure) is re-run
once; turn-limit and wall-clock endings are behaviour and kept. A placement leak of
severity ``drop`` drops the event (``leaks.jsonl``). ``ProviderPinError`` and
``PreflightAbort`` stop the whole run.

Cost: one ``rab.resume.CostMeter``; pre-flight after the first 20 agent calls (off in
the dry run); at the $15 cash ceiling no new event starts.

Dry run: ``FakeProvider`` agent (``rab.resume.dry_run_provider``), ``FakeSimulator``,
``FakeDiscriminator``, Docker, 2 events per kind unless ``--events`` says otherwise,
everything under ``results/dry/``, then ``rab.discriminate`` and ``rab.analyze`` to
``results/dry/tables.md``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import rab.constants as C
from rab import REPO_ROOT, km
from rab.types import Event, load_events

RESULTS = REPO_ROOT / "results"
DRY_ROOT = RESULTS / "dry"
DRY_EVENTS_PER_KIND = 2

_log_lock = threading.Lock()


class EventDrop(Exception):
    """The event leaves every arm (replay divergence twice, or a leak drop)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ev(event) -> dict:
    return event.to_json() if isinstance(event, Event) else dict(event)


def log_line(root: Path, row: dict) -> None:
    with _log_lock:
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "run_log.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"t": _utc(), **row}, sort_keys=True, ensure_ascii=False) + "\n")


def slot_path(root: Path, event_id: str, arm: str, run_idx: int) -> Path:
    return Path(root) / "continuations" / event_id / f"{arm}-{run_idx}.json.gz"


def write_slot(path: Path, slot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}{threading.get_ident()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(slot, fh, sort_keys=True, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def event_order(events: list[dict], seed: int = C.SEED) -> list[dict]:
    """Events in a seeded order: sha256 of ``seed:event_id``."""
    key = lambda e: hashlib.sha256(f"{seed}:{e['event_id']}:order".encode("utf-8")).hexdigest()
    return sorted(events, key=key)


def select_events(events: list[dict], per_kind: int | None = None, pilot: int | None = None) -> list[dict]:
    ordered = event_order(events)
    if per_kind is not None:
        taken = {k: 0 for k in C.KINDS}
        keep = []
        for e in ordered:
            if taken[e["kind"]] < per_kind:
                taken[e["kind"]] += 1
                keep.append(e)
        ordered = keep
    return ordered[:pilot] if pilot else ordered


# ------------------------------------------------------------------ one slot

def _replay_ok(rec: dict, ev: dict, batch_id: str):
    """Replay with one retry in a fresh container (decision 6); raises EventDrop."""
    from rab.prefix import replay

    first = None
    for attempt in (1, 2):
        rep = replay(rec, ev, batch_id=f"{batch_id}-r{attempt}")
        if rep.ok:
            return rep, attempt, first
        first = [c.diff_excerpt[:300] for c in rep.calls if not c.content_matched] + rep.failures
        rep.close()
    raise EventDrop("replay_divergence", json.dumps(first)[:600])


def run_slot(ev: dict, rec: dict, arm: str, run_idx: int, *, root: Path, sim_obj: dict | None,
             provider_for, meter, dry_run: bool, event_index: int) -> dict:
    """Replay, place, resume, classify one slot; returns the slot dict (not yet written)."""
    from rab.classify import classify
    from rab.place import arm_file, place
    from rab.prefix import continuation_budget
    from rab.resume import resume

    t0 = time.time()
    task_dir = km.task_dir(ev["item_key"], ev["kind"])
    batch_id = f"{ev['event_id']}-{arm}-{run_idx}-{uuid.uuid4().hex[:6]}"
    for divergence_try in (1, 2):
        rep, replay_attempts, _ = _replay_ok(rec, ev, batch_id)
        t_replay = time.time()
        try:
            text, build = arm_file(arm, ev, task_dir, sim_obj)
            placed = place(rep.sandbox, rep.item_key, task_dir, rec, ev, arm, file_text=text, build=build)
        except BaseException:
            rep.close()
            raise
        if arm == "R" and not placed.obs_matches_recording:
            rep.close()
            if divergence_try == 2:
                raise EventDrop("blocked_call_divergence", placed.observation[:300])
            continue
        if placed.leak_severity == "drop":
            rep.close()
            raise EventDrop("leak", json.dumps(placed.leak_hits)[:600])
        break
    t_place = time.time()
    provider = provider_for(ev, event_index)
    cont = resume(rep, placed, ev, rec, provider, run_idx, budget=continuation_budget(rec, ev),
                  dry_run=dry_run, meter=meter, refusals_path=root / "refusals.jsonl")
    t_resume = time.time()
    cls = classify(cont, ev)
    t_end = time.time()
    return {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "kind": ev["kind"],
            "task_id": ev["task_id"], "dry_run": dry_run, "cls": cls, "cont": cont,
            "placed": placed.to_json(),
            "replay": {"ok": rep.ok, "strict_ok": rep.strict_ok, "batch_id": rep.batch_id,
                       "attempts": replay_attempts, "notes": rep.notes,
                       "calls": [vars(c) for c in rep.calls]},
            "timing": {"replay_s": round(t_replay - t0, 2), "place_s": round(t_place - t_replay, 2),
                       "resume_s": round(t_resume - t_place, 2), "classify_s": round(t_end - t_resume, 2),
                       "total_s": round(t_end - t0, 2)}}


def run_slot_with_rerun(ev, rec, arm, run_idx, **kw) -> dict:
    """Design v2 C: one re-run for an infrastructure failure, never for behaviour."""
    root = kw["root"]
    slot = None
    for attempt in (1, 2):
        try:
            slot = run_slot(ev, rec, arm, run_idx, **kw)
        except EventDrop:
            raise
        except Exception as exc:
            from rab.resume import PreflightAbort, ProviderPinError

            if isinstance(exc, (PreflightAbort, ProviderPinError)):
                raise
            log_line(root, {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "attempt": attempt,
                            "status": "exception", "error": f"{type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc()[-1200:]})
            if attempt == 2:
                raise
            continue
        slot["attempt"] = attempt
        cont = slot["cont"]
        log_line(root, {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "attempt": attempt,
                        "status": "done", "klass": slot["cls"]["klass"], "error_class": cont.get("error_class"),
                        "rerun": cont.get("rerun"), "timing": slot["timing"], "cost_usd": cont.get("cost_usd")})
        if not cont.get("rerun") or attempt == 2:
            return slot
    return slot


# ------------------------------------------------------------------ simulator draws

def draws_for(ev: dict, rec: dict, root: Path, dry_run: bool, callers: dict) -> dict:
    """SA and SB draws for one event (cached); writes leak drops to ``leaks.jsonl``."""
    from rab.simulate import SimCache, draw

    cache = SimCache(root / "sims.jsonl")
    out = {}
    for family in ("SA", "SB"):
        sim = draw(rec, ev, family, callers[family], cache)
        out[family] = sim
        if sim.failure == "leak":
            with _log_lock, open(root / "leaks.jsonl", "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"event_id": ev["event_id"], "kind": ev["kind"], "where": f"prompt {family}",
                                     "severity": "drop", "detail": sim.detail}, sort_keys=True) + "\n")
            raise EventDrop("leak", f"simulator prompt {family}")
    return out


def make_callers(dry_run: bool) -> dict:
    from rab.simulate import FakeSimulator, make_caller

    if dry_run:
        callers = {}
        for fam in ("SA", "SB"):
            c = FakeSimulator()
            c.model = f"fake-simulator-{fam}"
            callers[fam] = c
        return callers
    return {"SA": make_caller("SA"), "SB": make_caller("SB")}


# ------------------------------------------------------------------ the run

def run(*, dry_run: bool, root: Path, per_kind: int | None, pilot: int | None, workers: int) -> dict:
    from rab.resume import CostMeter, dry_run_provider, event_slots, make_agent_provider, recorded_calls_prior

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    events = select_events([_ev(e) for e in load_events()], per_kind, pilot)
    meter = CostMeter(calls_per_continuation=recorded_calls_prior(events))
    callers = make_callers(dry_run)
    local = threading.local()

    def provider_for(ev: dict, event_index: int):
        if dry_run:
            return dry_run_provider(ev, event_index)
        if not hasattr(local, "provider"):
            local.provider = make_agent_provider(threading.current_thread().name)
        return local.provider

    started = time.time()
    done, skipped, dropped, stopped = 0, 0, [], None
    print(f"[run] {len(events)} events, {workers} workers, root {root}, dry_run={dry_run}",
          file=sys.stderr, flush=True)
    for idx, ev in enumerate(events):
        if meter.stop_reached:
            stopped = f"cash ceiling ${C.CASH_CEILING_USD:.0f} reached before {ev['event_id']}"
            break
        slots = event_slots(ev)
        todo = [(a, j) for a, j in slots if not slot_path(root, ev["event_id"], a, j).is_file()]
        skipped += len(slots) - len(todo)
        if not todo:
            continue
        rec = km.load_record(ev["run_id"], ev["position"])
        try:
            sims = draws_for(ev, rec, root, dry_run, callers)
        except EventDrop as drop:
            dropped.append(ev["event_id"])
            log_line(root, {"event_id": ev["event_id"], "drop": drop.reason, "detail": drop.detail})
            continue
        sa = sims["SA"]
        if not sa.ok:
            log_line(root, {"event_id": ev["event_id"], "arms_dropped": ["SA", "SSTAR"],
                            "reason": f"SA draw failed: {sa.failure}"})
            todo = [(a, j) for a, j in todo if a not in ("SA", "SSTAR")]
        kw = dict(root=root, sim_obj=sa.obj if sa.ok else None, provider_for=provider_for, meter=meter,
                  dry_run=dry_run, event_index=idx)
        t_ev = time.time()
        drop_reason = None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rabw") as pool:
            futs = {pool.submit(run_slot_with_rerun, ev, rec, a, j, **kw): (a, j) for a, j in todo}
            for fut, (a, j) in futs.items():
                try:
                    slot = fut.result()
                except EventDrop as drop:
                    drop_reason = drop_reason or drop
                    continue
                if drop_reason is None:
                    write_slot(slot_path(root, ev["event_id"], a, j), slot)
                    done += 1
        if drop_reason is not None:
            dropped.append(ev["event_id"])
            log_line(root, {"event_id": ev["event_id"], "drop": drop_reason.reason, "detail": drop_reason.detail})
            if drop_reason.reason == "leak":
                with _log_lock, open(root / "leaks.jsonl", "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps({"event_id": ev["event_id"], "kind": ev["kind"], "where": "placement",
                                         "severity": "drop", "detail": drop_reason.detail}, sort_keys=True) + "\n")
            for a, j in slots:  # a dropped event leaves every arm: no partial event stays behind
                slot_path(root, ev["event_id"], a, j).unlink(missing_ok=True)
        print(f"[run] event {idx + 1}/{len(events)} {ev['event_id']} ({ev['kind']}): {len(todo)} slots in "
              f"{time.time() - t_ev:.1f}s{' DROPPED ' + drop_reason.reason if drop_reason else ''}; "
              f"cash ${meter.cash_usd:.4f}", file=sys.stderr, flush=True)
    summary = {"events": len(events), "slots_written": done, "slots_skipped": skipped, "dropped": dropped,
               "stopped": stopped, "seconds": round(time.time() - started, 1), "cash_usd": meter.cash_usd}
    log_line(root, {"summary": summary})
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rab.run", description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--events", type=int, default=None, help="events per kind")
    ap.add_argument("--pilot", type=int, default=None, help="first N events overall")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=None, help="analysis permutations (dry run default 2,000)")
    args = ap.parse_args(argv)
    root = args.root or (DRY_ROOT if args.dry_run else RESULTS)
    if args.dry_run and Path(root).resolve() != DRY_ROOT.resolve() and DRY_ROOT.resolve() not in Path(root).resolve().parents:
        raise SystemExit("dry-run outputs go under results/dry/ only")
    per_kind = args.events if args.events is not None else (DRY_EVENTS_PER_KIND if args.dry_run else None)
    summary = run(dry_run=args.dry_run, root=root, per_kind=per_kind, pilot=args.pilot, workers=args.workers)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    if args.dry_run:
        from rab import analyze
        from rab.discriminate import FakeDiscriminator, discriminate

        discriminate(root, FakeDiscriminator())
        n_perm = args.n_perm or 2000
        analyze.run(root, n_perm=n_perm)
        analyze.run(root, placebo_mode=True, n_perm=n_perm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
