"""The run: every slot of every event, resumable (design v2 B, C, D; guide §6, §10).

``python -m rab.run [--dry-run] [--events N] [--workers 4] [--pilot 5] [--exploratory] [--root DIR]``

Nothing here refuses to run: no preregistration check, no gate, no date (build plan,
"Preregistration rule"). The proof of what was run when is the git state recorded in
every slot file and every simulator line (:func:`git_state`: ``git rev-parse HEAD`` and
whether tracked files differ from it).

Exploratory runs (``--exploratory``, on by default with ``--pilot``): slot files carry
``"exploratory": true`` and are named ``<arm>-<run>-x.json.gz`` beside the headline
files. A headline run never skips a slot because an exploratory file exists, so a pilot
event is re-run in full; ``rab.analyze`` and ``rab.discriminate`` leave exploratory
files out unless ``--include-exploratory``.

Order. Events are taken in a seeded order (:func:`event_order`), event-major. ``--events
N`` keeps the first N events of each kind in that order; ``--pilot N`` keeps the first N
events overall (design v2 F: the 5-event pilot). Within an event the slots are
``rab.resume.event_slots`` (18, from ``constants.RUNS_PER_EVENT``, in a seeded shuffle)
and run on ``--workers`` threads.

Per event, before its slots: the simulator draws (``rab.simulate.draw``) for SA and SB,
cached in ``<root>/sims.jsonl`` (new lines carry ``git``; their spend goes to the cost
meter: SA on credit, SB in cash). A prompt leak of severity ``drop`` drops the event.
An SA draw that failed (schema twice, refusal) drops SA and SSTAR for that event only.

Per slot (:func:`run_slot`): replay (``rab.prefix.replay``, a fresh container with a
unique batch id) -> the arm's file (``rab.place.arm_file``) -> ``rab.place.place`` ->
``rab.resume.resume`` (closes the container, runs both grades) ->
``rab.classify.classify``. One gzip JSON file per slot,
``<root>/continuations/<event_id>/<arm>-<run>[-x].json.gz``, written atomically; an
existing file is skipped (resumable). One ``<root>/run_log.jsonl`` line per slot
attempt.

Files the run leaves, and what a resume does with them:

* ``<arm>-<run>[-x].json.gz``: a finished slot; skipped.
* ``<arm>-<run>[-x].failed.json.gz``: an infrastructure failure (``rerun`` still set, or
  an exception) after two attempts; not a result, never skipped, so a resume retries it;
  ``rab.analyze`` counts it in the drops table.
* ``DROPPED[-x].json``: the event left every arm (replay or R blocked-call divergence
  twice, a leak ``drop``, ``rab.types.EventDrop``), with the cause; the event is skipped
  on resume and ``rab.analyze`` counts drops from these markers.
* ``ARMDROP-<arm>[-x].json``: one arm left this event (a builder or placement error: W
  with under 2 cases, S* with no computable case, P equal to the real file, a fake
  observation truncated where the recording was not); that arm is skipped on resume.

Re-runs (design v2 C, decision 6): a prefix that does not replay ``ok``, or an R blocked
call whose live output is not ``content_equal`` to the recording, is retried once in a
fresh container; a second divergence drops the event from every arm. Only R can be
checked this way: a fake arm's blocked-call output has no recording to compare with.
R retries are counted in each slot (``replay.r_divergence_retries``). A continuation
whose ``rerun`` is set is re-run once.

Stops. One stop flag for the run: set by ``ProviderPinError``, ``PreflightAbort`` (also
when it is the cached verdict of another worker), the $15 cash ceiling, or Ctrl-C. Every
slot checks it at its start, and the agent provider is wrapped (:class:`StoppableProvider`)
so the flag is checked before every provider call inside ``resume``; a stopped slot is
not written, queued slots are cancelled, and the run raises after cleaning up. The cost
meter is seeded at start from the existing slot files' ``cost_usd`` and the simulator
spend in ``sims.jsonl``, so the ceiling holds across processes.

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
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import rab.constants as C
from rab import REPO_ROOT, km
from rab.km import utc_now as _utc
from rab.types import EventDrop, event_dict as _ev, load_events

RESULTS = REPO_ROOT / "results"
DRY_ROOT = RESULTS / "dry"
DRY_EVENTS_PER_KIND = 2

_log_lock = threading.Lock()


class ArmDrop(Exception):
    """One arm leaves one event: its builder or its placement cannot be done."""

    def __init__(self, arm: str, reason: str) -> None:
        super().__init__(f"{arm}: {reason}")
        self.arm, self.reason = arm, reason


class StopRun(BaseException):
    """Raised inside a slot when the run's stop flag is set. A ``BaseException``, so
    ``resume``'s ``except Exception`` does not turn it into a recorded error."""


def git_state() -> dict:
    """``{"git_hash": HEAD, "dirty": tracked files differ from HEAD}`` of this repo, now."""
    def git(*args) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True,
                                  timeout=30, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    head = git("rev-parse", "HEAD") or "unknown"
    return {"git_hash": head, "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def log_line(root: Path, row: dict) -> None:
    with _log_lock:
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "run_log.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"t": _utc(), **row}, sort_keys=True, ensure_ascii=False) + "\n")


def _x(exploratory: bool) -> str:
    return "-x" if exploratory else ""


def slot_path(root: Path, event_id: str, arm: str, run_idx: int, exploratory: bool = False,
              failed: bool = False) -> Path:
    """``<arm>-<run>[-x].json.gz``; ``.failed.json.gz`` for an unrecovered failure."""
    tail = ".failed.json.gz" if failed else ".json.gz"
    return Path(root) / "continuations" / event_id / f"{arm}-{run_idx}{_x(exploratory)}{tail}"


def drop_path(root: Path, event_id: str, exploratory: bool = False) -> Path:
    return Path(root) / "continuations" / event_id / f"DROPPED{_x(exploratory)}.json"


def armdrop_path(root: Path, event_id: str, arm: str, exploratory: bool = False) -> Path:
    return Path(root) / "continuations" / event_id / f"ARMDROP-{arm}{_x(exploratory)}.json"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8", newline="\n")


def write_slot(path: Path, slot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}{threading.get_ident()}")
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


# ------------------------------------------------------------------ stopping

class Stopper:
    """The run's one stop flag, with the reason; checked between slots and provider calls."""

    def __init__(self, meter) -> None:
        self.event = threading.Event()
        self.reason: str | None = None
        self.error: BaseException | None = None
        self.meter = meter

    def stop(self, reason: str, error: BaseException | None = None) -> None:
        if not self.event.is_set():
            self.reason, self.error = reason, error
            self.event.set()

    def check(self) -> None:
        """Raise :class:`StopRun` if the run must stop (ceiling or a cached pre-flight abort)."""
        pf = self.meter.preflight_result
        if pf and pf.get("abort"):
            self.stop(f"pre-flight abort: {'; '.join(pf.get('reasons') or [])}")
        if self.meter.stop_reached:
            self.stop(f"cash ceiling ${C.CASH_CEILING_USD:.0f} reached")
        if self.event.is_set():
            raise StopRun(self.reason)


class StoppableProvider:
    """Wraps the agent provider: :meth:`Stopper.check` before every ``chat``."""

    def __init__(self, provider, stopper: Stopper) -> None:
        self._p, self._stopper = provider, stopper

    def chat(self, *args, **kwargs):
        self._stopper.check()
        return self._p.chat(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._p, name)


def seed_meter(meter, root: Path) -> dict:
    """Spend already on disk: existing slot files' ``cost_usd`` (agent, cash) and every
    ``sims.jsonl`` line's usage (SA on credit, SB in cash)."""
    from rab.analyze import load_slot, read_jsonl

    agent = 0.0
    for path in sorted(Path(root).glob("continuations/*/*.json.gz")):
        row = load_slot(path)
        cont = row.get("cont") or (row.get("slot") or {}).get("cont") or {}  # failed files nest it
        agent += float(cont.get("cost_usd") or 0.0)
    meter.cash_usd += agent
    sims = 0.0
    for row in read_jsonl(Path(root) / "sims.jsonl"):
        sims += _meter_sim(meter, row.get("family"), row.get("model"), row.get("usage") or {})
    return {"agent_usd": agent, "simulator_usd": sims}


def _meter_sim(meter, family: str | None, model: str | None, usage: dict) -> float:
    provider = "anthropic" if family == "SA" else "openrouter"
    return meter.add(provider, model or "", usage, seat="simulator") if usage else 0.0


# ------------------------------------------------------------------ one slot

def _replay_ok(rec: dict, ev: dict, batch_id: str):
    """Replay with one retry in a fresh container (decision 6); raises EventDrop."""
    from rab.prefix import replay

    first = None
    for attempt in (1, 2):
        rep = replay(rec, ev, batch_id=f"{batch_id}-r{attempt}")
        if rep.ok:
            return rep, attempt
        first = [c.diff_excerpt[:300] for c in rep.calls if not c.content_matched] + rep.failures
        rep.close()
    raise EventDrop("replay_divergence", json.dumps(first)[:600])


def run_slot(ev: dict, rec: dict, arm: str, run_idx: int, *, root: Path, sim_obj: dict | None,
             provider_for, meter, dry_run: bool, event_index: int, stopper: Stopper) -> dict:
    """Replay, place, resume, classify one slot; returns the slot dict (not yet written)."""
    from rab.classify import classify
    from rab.place import PlacementError, arm_file, place
    from rab.prefix import continuation_budget
    from rab.resume import resume

    stopper.check()
    t0 = time.time()
    task_dir = km.task_dir(ev["item_key"], ev["kind"])
    batch_id = f"{ev['event_id']}-{arm}-{run_idx}-{uuid.uuid4().hex[:6]}"
    try:
        text, build = arm_file(arm, ev, task_dir, sim_obj)
    except (TypeError, ValueError, PlacementError) as exc:
        raise ArmDrop(arm, f"builder: {type(exc).__name__}: {exc}") from exc
    r_retries = 0
    for divergence_try in (1, 2):
        rep, replay_attempts = _replay_ok(rec, ev, batch_id)
        t_replay = time.time()
        try:
            placed = place(rep.sandbox, rep.item_key, task_dir, rec, ev, arm, file_text=text, build=build)
        except PlacementError as exc:
            rep.close()
            raise ArmDrop(arm, f"placement: {exc}") from exc
        except BaseException:
            rep.close()
            raise
        if arm == "R" and not placed.obs_matches_recording:
            rep.close()
            if divergence_try == 2:
                raise EventDrop("blocked_call_divergence", placed.observation[:300])
            r_retries += 1
            continue
        if placed.leak_severity == "drop":
            rep.close()
            raise EventDrop("leak", json.dumps(placed.leak_hits)[:600])
        break
    t_place = time.time()
    provider = StoppableProvider(provider_for(ev, event_index), stopper)
    try:
        cont = resume(rep, placed, ev, rec, provider, run_idx, budget=continuation_budget(rec, ev),
                      dry_run=dry_run, meter=meter, refusals_path=root / "refusals.jsonl")
    finally:
        rep.close()  # resume closes it too; a StopRun passes resume by, so close here as well
    t_resume = time.time()
    cls = classify(cont, ev)
    t_end = time.time()
    return {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "kind": ev["kind"],
            "task_id": ev["task_id"], "dry_run": dry_run, "git": git_state(), "cls": cls, "cont": cont,
            "placed": placed.to_json(),
            "replay": {"ok": rep.ok, "strict_ok": rep.strict_ok, "batch_id": rep.batch_id,
                       "attempts": replay_attempts, "r_divergence_retries": r_retries, "notes": rep.notes,
                       "calls": [vars(c) for c in rep.calls]},
            "timing": {"replay_s": round(t_replay - t0, 2), "place_s": round(t_place - t_replay, 2),
                       "resume_s": round(t_resume - t_place, 2), "classify_s": round(t_end - t_resume, 2),
                       "total_s": round(t_end - t0, 2)}}


def run_slot_with_rerun(ev, rec, arm, run_idx, **kw) -> tuple[str, dict]:
    """Design v2 C: one re-run for an infrastructure failure, never for behaviour.

    Returns ``("done", slot)`` or ``("failed", record)``; drops and stops propagate.
    """
    from rab.resume import PreflightAbort, ProviderPinError

    root, stopper = kw["root"], kw["stopper"]
    last: dict = {}
    for attempt in (1, 2):
        try:
            slot = run_slot(ev, rec, arm, run_idx, **kw)
        except (EventDrop, ArmDrop, StopRun):
            raise
        except (PreflightAbort, ProviderPinError) as exc:
            stopper.stop(f"{type(exc).__name__}: {exc}", exc)
            raise StopRun(stopper.reason) from exc
        except Exception as exc:
            last = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
            log_line(root, {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "attempt": attempt,
                            "status": "exception", **last})
            continue
        slot["attempt"] = attempt
        cont = slot["cont"]
        log_line(root, {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "attempt": attempt,
                        "status": "done", "klass": slot["cls"]["klass"], "error_class": cont.get("error_class"),
                        "rerun": cont.get("rerun"), "timing": slot["timing"], "cost_usd": cont.get("cost_usd")})
        if not cont.get("rerun"):
            return "done", slot
        last = {"error_class": cont.get("error_class"), "slot": slot}
    return "failed", {"event_id": ev["event_id"], "arm": arm, "run_idx": run_idx, "kind": ev["kind"],
                      "task_id": ev["task_id"], "git": git_state(), **last}


# ------------------------------------------------------------------ simulator draws

def _git_sim_cache(path: Path, meter, exploratory: bool):
    """``rab.simulate.SimCache`` whose new lines carry ``git`` and ``exploratory`` and
    whose new draws are metered (``put`` is called only for a fresh draw)."""
    from rab.simulate import SimCache

    class GitSimCache(SimCache):
        def put(self, sim) -> None:
            row = {**sim.to_json(), "git": git_state(), "exploratory": exploratory}
            _meter_sim(meter, sim.family, sim.model, sim.usage or {})
            self._rows[(sim.model, sim.prompt_sha256)] = row
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    return GitSimCache(path)


def draws_for(ev: dict, rec: dict, root: Path, callers: dict, meter, exploratory: bool = False) -> dict:
    """SA and SB draws for one event (cached); a prompt leak ``drop`` raises EventDrop."""
    from rab.simulate import draw

    cache = _git_sim_cache(root / "sims.jsonl", meter, exploratory)
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

def _drop_event(root: Path, ev: dict, drop, exploratory: bool, slots) -> None:
    _write_json(drop_path(root, ev["event_id"], exploratory),
                {"event_id": ev["event_id"], "kind": ev["kind"], "reason": drop.reason, "detail": drop.detail,
                 "exploratory": exploratory, "git": git_state(), "t": _utc()})
    if drop.reason == "leak":
        with _log_lock, open(root / "leaks.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"event_id": ev["event_id"], "kind": ev["kind"], "where": "placement",
                                 "severity": "drop", "detail": drop.detail}, sort_keys=True) + "\n")
    for a, j in slots:  # a dropped event leaves every arm: no partial event stays behind
        slot_path(root, ev["event_id"], a, j, exploratory).unlink(missing_ok=True)
    log_line(root, {"event_id": ev["event_id"], "drop": drop.reason, "detail": drop.detail})


def run(*, dry_run: bool, root: Path, per_kind: int | None, pilot: int | None, workers: int,
        exploratory: bool = False) -> dict:
    from rab.resume import CostMeter, dry_run_provider, event_slots, make_agent_provider, recorded_calls_prior

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    events = select_events([_ev(e) for e in load_events()], per_kind, pilot)
    meter = CostMeter(calls_per_continuation=recorded_calls_prior(events))
    seeded = seed_meter(meter, root)
    stopper = Stopper(meter)
    callers = make_callers(dry_run)
    local = threading.local()

    def provider_for(ev: dict, event_index: int):
        if dry_run:
            return dry_run_provider(ev, event_index)
        if not hasattr(local, "provider"):
            local.provider = make_agent_provider(threading.current_thread().name)
        return local.provider

    started = time.time()
    done = skipped = failed = 0
    dropped, arm_drops = [], []
    print(f"[run] {len(events)} events, {workers} workers, root {root}, dry_run={dry_run}, "
          f"exploratory={exploratory}, git {git_state()}, spend on disk {seeded}", file=sys.stderr, flush=True)
    try:
        for idx, ev in enumerate(events):
            stopper.check()
            eid = ev["event_id"]
            if drop_path(root, eid, exploratory).is_file():
                skipped += 1
                continue
            slots = event_slots(ev)
            gone = {a for a in {a for a, _ in slots} if armdrop_path(root, eid, a, exploratory).is_file()}
            todo = [(a, j) for a, j in slots
                    if a not in gone and not slot_path(root, eid, a, j, exploratory).is_file()]
            skipped += len(slots) - len(todo)
            if not todo:
                continue
            rec = km.load_record(ev["run_id"], ev["position"])
            try:
                sims = draws_for(ev, rec, root, callers, meter, exploratory)
            except EventDrop as drop:
                dropped.append(eid)
                _drop_event(root, ev, drop, exploratory, slots)
                continue
            sa = sims["SA"]
            if not sa.ok:
                for a in ("SA", "SSTAR"):
                    _write_json(armdrop_path(root, eid, a, exploratory),
                                {"event_id": eid, "arm": a, "reason": f"SA draw failed: {sa.failure}",
                                 "exploratory": exploratory, "git": git_state()})
                todo = [(a, j) for a, j in todo if a not in ("SA", "SSTAR")]
            kw = dict(root=root, sim_obj=sa.obj if sa.ok else None, provider_for=provider_for, meter=meter,
                      dry_run=dry_run, event_index=idx, stopper=stopper)
            t_ev = time.time()
            event_drop = None
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rabw")
            try:
                futs = {pool.submit(run_slot_with_rerun, ev, rec, a, j, **kw): (a, j) for a, j in todo}
                for fut, (a, j) in futs.items():
                    try:
                        status, out = fut.result()
                    except EventDrop as drop:
                        event_drop = event_drop or drop
                        continue
                    except ArmDrop as ad:
                        p = armdrop_path(root, eid, ad.arm, exploratory)
                        if not p.is_file():
                            arm_drops.append(f"{eid}:{ad.arm}")
                            _write_json(p, {"event_id": eid, "arm": ad.arm, "reason": ad.reason,
                                            "exploratory": exploratory, "git": git_state()})
                            log_line(root, {"event_id": eid, "arm_drop": ad.arm, "reason": ad.reason})
                        continue
                    except StopRun:
                        pool.shutdown(wait=False, cancel_futures=True)
                        continue
                    except CancelledError:
                        continue
                    except BaseException:
                        stopper.stop("interrupted")
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    if event_drop is not None:
                        continue
                    if status == "done":
                        out["exploratory"] = exploratory
                        write_slot(slot_path(root, eid, a, j, exploratory), out)
                        slot_path(root, eid, a, j, exploratory, failed=True).unlink(missing_ok=True)
                        done += 1
                    else:
                        out["exploratory"] = exploratory
                        write_slot(slot_path(root, eid, a, j, exploratory, failed=True), out)
                        failed += 1
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            if event_drop is not None:
                dropped.append(eid)
                _drop_event(root, ev, event_drop, exploratory, slots)
            print(f"[run] event {idx + 1}/{len(events)} {eid} ({ev['kind']}): {len(todo)} slots in "
                  f"{time.time() - t_ev:.1f}s{' DROPPED ' + event_drop.reason if event_drop else ''}; "
                  f"cash ${meter.cash_usd:.4f}", file=sys.stderr, flush=True)
            if stopper.event.is_set():
                break
    except StopRun:
        pass
    except KeyboardInterrupt:
        stopper.stop("interrupted (Ctrl-C)")
    summary = {"events": len(events), "slots_written": done, "slots_failed": failed, "slots_skipped": skipped,
               "dropped": dropped, "arm_drops": arm_drops, "stopped": stopper.reason,
               "seconds": round(time.time() - started, 1), "cash_usd": round(meter.cash_usd, 6),
               "spend_on_disk_at_start": seeded, "exploratory": exploratory, "git": git_state()}
    log_line(root, {"summary": summary})
    if stopper.error is not None:
        raise stopper.error
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rab.run", description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--events", type=int, default=None, help="events per kind")
    ap.add_argument("--pilot", type=int, default=None, help="first N events overall")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--exploratory", action=argparse.BooleanOptionalAction, default=None,
                    help="label slots exploratory (files <arm>-<run>-x.json.gz); default on with --pilot")
    ap.add_argument("--n-perm", type=int, default=None, help="analysis permutations (dry run default 2,000)")
    args = ap.parse_args(argv)
    root = args.root or (DRY_ROOT if args.dry_run else RESULTS)
    if args.dry_run and Path(root).resolve() != DRY_ROOT.resolve() and DRY_ROOT.resolve() not in Path(root).resolve().parents:
        raise SystemExit("dry-run outputs go under results/dry/ only")
    per_kind = args.events if args.events is not None else (DRY_EVENTS_PER_KIND if args.dry_run else None)
    exploratory = args.exploratory if args.exploratory is not None else bool(args.pilot)
    summary = run(dry_run=args.dry_run, root=root, per_kind=per_kind, pilot=args.pilot, workers=args.workers,
                  exploratory=exploratory)
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
