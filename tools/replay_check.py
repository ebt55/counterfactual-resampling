"""The day-0 replay check and the eligibility sweep (guide §1 "Replay", §3; build plan
decisions 17b, 17c), moved out of ``rab.prefix`` so the core module only rebuilds and
replays. Checks, not gates: nothing here stops anything else from running.

    uv run python tools/replay_check.py --check 20   # results/replay_check.md
    uv run python tools/replay_check.py --sweep      # data/replay_sweep.json

``--check N``: ``N`` seeded events from ``data/events.jsonl``, one fresh container each,
a per-event table with the content and strict counts (``rab.prefix.content_equal`` /
``rab.prefix.normalise``). ``--sweep``: every event ``rab.select`` keeps before the draw;
``rab.select`` drops events whose prefix is not ``ok`` before the seeded draw.
:func:`classify` labels every call that fails the strict comparison; its labels are the
sweep file's ``classes`` field and must stay byte-stable (``tests/test_prefix.py``).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rab import REPO_ROOT, km
from rab.constants import KINDS, REPLAY_PASS_SHARE, SEED
from rab.prefix import (EVENTS_PATH, EXCERPT_CHARS, _MONTH, content_equal, load_events, normalise,
                        record_for, replay)
from harness.config import IMAGE
from harness.sandbox import docker_available, image_available

SWEEP_PATH = REPO_ROOT / "data" / "replay_sweep.json"
REPLAY_REPORT = REPO_ROOT / "results" / "replay_check.md"

# ------------------------------------------------------------------ classification

_LS_NLINK = re.compile(r"^([-bcdlps][rwxsStT-]{9}[.+@]?)[ ]+(\d+)[ ]+(.*)$")


def classify(recorded: str, replayed: str) -> str:
    """A first-pass label for a call that fails the strict comparison.

    Both texts already normalised. Mechanical, so a reader can check it; first match
    wins:

    * ``ordering (stdout/stderr interleaving)``: what :func:`content_equal` accepts;
      plain ``ordering`` for any other reordering of the same lines.
    * ``timeout``: one side carries KM's ``[command timed out ...]`` head.
    * ``truncation boundary``: exactly one side was truncated.
    * ``lines only in the recording`` / ``lines only in the replay``.
    * ``ls -l link count (/proc nlink)`` / ``ls -l link count``: paired ``ls -l`` lines
      equal once the link count and its padding are dropped (the ``/proc`` form cannot
      occur after :func:`normalise`; kept for texts normalised the old way).
    * ``/proc/<pid> entries``: every differing line names ``/proc/<digits>``.
    * ``size-only``: paired ``ls -l`` / ``total`` lines differ only in digits.
    * ``container id/hostname``, ``version string``,
      ``date/time outside an ls -l date column``, else ``other``.
    """
    a, b = recorded.splitlines(), replayed.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    gone = [ln for op, i1, i2, _, _ in sm.get_opcodes() if op in ("delete", "replace") for ln in a[i1:i2]]
    new = [ln for op, _, _, j1, j2 in sm.get_opcodes() if op in ("insert", "replace") for ln in b[j1:j2]]
    if sorted(a) == sorted(b):
        return "ordering (stdout/stderr interleaving)" if content_equal(recorded, replayed) \
            else "ordering"
    if recorded.startswith("[command timed out") or replayed.startswith("[command timed out"):
        return "timeout"
    if ("[truncated:" in recorded) != ("[truncated:" in replayed):
        return "truncation boundary"
    if gone and not new:
        return "lines only in the recording"
    if new and not gone:
        return "lines only in the replay"
    pairs = list(zip(gone, new))
    if len(gone) == len(new):
        ma = [_LS_NLINK.match(x) for x in gone]
        mb = [_LS_NLINK.match(y) for y in new]
        if all(ma) and all(mb) and all((x[1], x[3]) == (y[1], y[3]) for x, y in zip(ma, mb)):
            if any(x[2] != y[2] and x[3].endswith(" proc") for x, y in zip(ma, mb)):
                return "ls -l link count (/proc nlink)"
            return "ls -l link count"
    if all(re.search(r"/proc/\d+", x) for x in gone + new):
        return "/proc/<pid> entries"
    ls_size = re.compile(r"^(?:total \d+|[-bcdlps][rwxsStT-]{9}.*)$")
    if all(ls_size.match(x) for x in gone + new) and len(gone) == len(new) and all(
            re.sub(r"\d+", "#", x) == re.sub(r"\d+", "#", y) for x, y in pairs):
        return "size-only"
    joined = "\n".join(gone + new)
    if re.search(r"\b[0-9a-f]{12}\b|HOSTNAME=", joined):
        return "container id/hostname"
    if re.search(r"\d+\.\d+\.\d+", joined) and len(gone) == len(new) and all(
            re.sub(r"\d+", "#", x) == re.sub(r"\d+", "#", y) for x, y in pairs):
        return "version string"
    if re.search(rf"\d{{4}}-\d\d-\d\d|\d\d:\d\d|{_MONTH} ", joined):
        return "date/time outside an ls -l date column"
    return "other"


def draw(events: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """``random.Random(seed).sample`` of ``n`` events, from event ids in sorted order."""
    by_id = {e["event_id"]: e for e in events}
    return [by_id[eid] for eid in random.Random(seed).sample(sorted(by_id), n)]


def eligible_events() -> list[dict]:
    """Every §2-eligible event before the draw (contradict and same-turn drops only).

    Built from ``rab.select``'s own ``scan`` and ``eligible_ids``, so the sweep covers
    exactly the set the draw samples from; sorted by event id.
    """
    from rab import select as sel

    found, _ = sel.scan(km.iter_line_records())
    keep = {eid for kind in KINDS for eid in sel.eligible_ids(found, kind)}
    return sorted((f["event"] for f in found if f["event"]["event_id"] in keep),
                  key=lambda e: e["event_id"])


def replay_one(ev: dict) -> dict:
    """Replay one event in its own container, always removed; a row for the reports."""
    t0 = time.time()
    row = {"event_id": ev["event_id"], "kind": ev["kind"], "prefix_calls": int(ev["i"]),
           "matched": 0, "content_matched": 0, "ok": False, "strict_ok": False,
           "first_fail": "", "excerpt": "", "class": "", "classes": [], "failed_calls": [],
           "work": "", "error": ""}
    res = None
    try:
        rec = record_for(ev)
        res = replay(rec, ev)
        row.update(matched=sum(c.matched for c in res.calls),
                   content_matched=sum(c.content_matched for c in res.calls),
                   ok=res.ok, strict_ok=res.strict_ok, work=res.work_listing)
        for c in res.calls:
            if c.matched:
                continue
            label = classify(normalise(rec["tool_calls"][c.index]["output"]),
                             normalise(res.replayed[c.index]))
            row["failed_calls"].append({"index": c.index, "name": c.tool, "class": label,
                                        "content_matched": c.content_matched,
                                        "excerpt": c.diff_excerpt})
        if res.failures:
            row["failed_calls"].append({"index": "touch", "name": "", "class": "host-side step (/tmp trace or touch)",
                                        "content_matched": False, "excerpt": "; ".join(res.failures)})
    except Exception as exc:  # recorded, never fatal to the check or the sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["failed_calls"].append({"index": "-", "name": "", "class": "replay error",
                                    "content_matched": False, "excerpt": row["error"]})
    finally:
        if res is not None:
            res.close()
    if row["failed_calls"]:
        first = row["failed_calls"][0]
        row["first_fail"] = f"{first['index']} ({first['name']})" if first["name"] else str(first["index"])
        row["excerpt"] = first["excerpt"]
        row["class"] = first["class"]
    row["classes"] = sorted({c["class"] for c in row["failed_calls"]})
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _preflight() -> None:
    if not docker_available():
        raise RuntimeError("Docker is not available")
    if not image_available(IMAGE):
        raise RuntimeError(f"image {IMAGE} missing: docker build -t {IMAGE} harness/docker in KM_ROOT")


def _progress(j: int, n: int, row: dict) -> None:
    print(f"[{j + 1}/{n}] {row['event_id']} i={row['prefix_calls']} ok={row['ok']} "
          f"strict_ok={row['strict_ok']} matched={row['matched']}/{row['content_matched']}"
          f"/{row['prefix_calls']} {','.join(row['classes'])} {row['seconds']}s",
          file=sys.stderr, flush=True)


def replay_check(n: int = 20, seed: int = SEED, events_path: Path = EVENTS_PATH,
                 out_path: Path = REPLAY_REPORT) -> dict:
    """Day-0 gate (guide §1): replay ``n`` seeded events, one container each, in turn.

    The gate counts ``Replay.ok`` (content); the strict count is reported beside it.
    Writes ``out_path`` and returns a summary. Every container is removed before the
    next event starts, whatever happens.
    """
    _preflight()
    picked = draw(load_events(events_path), n, seed)
    t_all = time.time()
    rows = []
    for j, ev in enumerate(picked):
        rows.append(replay_one(ev))
        _progress(j, n, rows[-1])
    need = math.ceil(REPLAY_PASS_SHARE * n)
    passed = sum(r["ok"] for r in rows)
    summary = {"n": n, "seed": seed, "passed": passed,
               "strict_passed": sum(r["strict_ok"] for r in rows), "need": need,
               "pass": passed >= need, "seconds": round(time.time() - t_all, 1),
               "image_id": _image_id(IMAGE), "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_report(summary, events_path), encoding="utf-8", newline="\n")
    return summary


def sweep(out_path: Path = SWEEP_PATH, events: list[dict] | None = None) -> dict:
    """Replay every eligible event once and write ``data/replay_sweep.json``.

    Eligibility rule (build plan decision): an event is kept only if its prefix replays
    clean in a fresh env-v2 container under :func:`content_equal` (``ok``). The file
    holds, per event id, ``ok``, ``strict_ok``, ``n_calls`` (prefix calls), ``kind`` and
    ``classes`` (the :func:`classify` labels of every call failing the strict
    comparison, so a content-only pass still shows its class), plus ``totals`` per kind.
    Keys are sorted; ``generated`` is the only field that changes between identical runs.
    """
    _preflight()
    events = eligible_events() if events is None else events
    entries: dict[str, dict] = {}
    t_all = time.time()
    for j, ev in enumerate(events):
        row = replay_one(ev)
        _progress(j, len(events), row)
        entries[ev["event_id"]] = {"ok": row["ok"], "strict_ok": row["strict_ok"],
                                   "n_calls": row["prefix_calls"], "kind": row["kind"],
                                   "classes": row["classes"]}
    totals = {}
    for kind in sorted({e["kind"] for e in entries.values()}):
        mine = {k: v for k, v in entries.items() if v["kind"] == kind}
        totals[kind] = {"n": len(mine), "ok": sum(v["ok"] for v in mine.values()),
                        "strict_ok": sum(v["strict_ok"] for v in mine.values()),
                        "excluded": sorted(k for k, v in mine.items() if not v["ok"])}
    doc = {"image": _image_id(IMAGE), "generated": km.utc_now(),
           "seconds": round(time.time() - t_all, 1),
           "rule": "replay reproduces the previous item's /tmp trace at position >= 1; "
                   "ok = every prefix call content_equal (rab.prefix) and the host-side "
                   "steps succeeded; strict_ok = every call equal after normalise",
           "events": entries, "totals": totals}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    return doc


def _image_id(image: str) -> str:
    try:
        out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             check=False, timeout=60)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _cell(text: str, width: int = EXCERPT_CHARS) -> str:
    text = text[:width].replace("\\", "\\\\").replace("|", "\\|").replace("`", "'")
    return text.replace("\r", "").replace("\n", " \u23ce ")


def _report(s: dict, events_path: Path) -> str:
    path = events_path.resolve()
    rel = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
    verdict = "PASS" if s["pass"] else "FAIL"
    lines = [
        "# Replay check (guide §1 \"Replay\", §3)",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        f"`uv run python tools/replay_check.py --check {s['n']}` (seed {s['seed']}), from `{rel}`.",
        f"Image `{IMAGE}` = `{s['image_id']}`. One fresh env-v2 container per event, "
        f"run in turn; every container removed after its event. {s['seconds']} s in all.",
        "",
        f"**Content: {s['passed']} of {s['n']} events reproduce every prefix output "
        f"(`content_equal`); the rule needs {s['need']}. {verdict}. "
        f"Strict (`normalise` only): {s['strict_passed']} of {s['n']}.**",
        "",
        "Strict: `normalise(recorded) == normalise(replayed)`, where `normalise` erases the "
        "date column of `ls -l` lines, the `N.NN` of a pytest `in N.NNs` summary and (a "
        "PREREG departure) the `/proc` link count with the link-column padding "
        "(`LS_DATE_RE`, `PYTEST_TIME_RE`, `LS_NLINK_RE` in `src/rab/prefix.py`). "
        "Content: strict, or the same lines where only `prog: message` diagnostics moved "
        "(`content_equal`; stdout and stderr reach `docker exec` on separate streams). "
        "Any other difference is a failure. The excerpt is a zero-context unified diff "
        "of the first call failing the strict comparison (`-` recorded, `+` replayed), "
        "first 200 characters, newlines shown as \u23ce.",
        "",
        "| event_id | kind | prefix calls | strict matched | content matched | ok | "
        "first strict failure | class | diff excerpt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        lines.append(f"| `{r['event_id']}` | {r['kind']} | {r['prefix_calls']} | "
                     f"{r['matched']} | {r['content_matched']} | {'yes' if r['ok'] else 'no'} | "
                     f"{r['first_fail'] or '-'} | {r['class'] or '-'} | "
                     f"{_cell(r['excerpt']) or '-'} |")
    failed = [(r, c) for r in s["rows"] for c in r["failed_calls"]]
    if failed:
        lines += ["", "## Every call failing the strict comparison", "",
                  "| event_id | call | class | content match | diff excerpt |",
                  "|---|---|---|---|---|"]
        for r, c in failed:
            lines.append(f"| `{r['event_id']}` | {c['index']} {c['name']} | {c['class']} | "
                         f"{'yes' if c['content_matched'] else 'no'} | {_cell(c['excerpt'])} |")
    lines += ["", "## `/work` after `touch -d <work_mtime>`", "",
              "| event_id | `ls -ld /work` |", "|---|---|"]
    for r in s["rows"]:
        lines.append(f"| `{r['event_id']}` | {_cell(r['work']) or '(no work_mtime on the event)'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/replay_check.py", description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", type=int, metavar="N",
                      help="replay N seeded drawn events and write results/replay_check.md")
    mode.add_argument("--sweep", action="store_true",
                      help="replay every eligible event and write data/replay_sweep.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--events", type=Path, default=EVENTS_PATH)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.sweep:
        out = args.out or SWEEP_PATH
        doc = sweep(out)
        for kind, t in doc["totals"].items():
            print(f"{kind}: {t['ok']} of {t['n']} ok, {t['strict_ok']} strict; "
                  f"excluded {t['excluded']}", file=sys.stderr)
        print(f"wrote {out}", file=sys.stderr)
        return 0
    out = args.out or REPLAY_REPORT
    s = replay_check(args.check, args.seed, args.events, out)
    print(f"{s['passed']} of {s['n']} events replayed clean by content, {s['strict_passed']} "
          f"strict (need {s['need']}): {'PASS' if s['pass'] else 'FAIL'}; wrote {out}",
          file=sys.stderr)
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
