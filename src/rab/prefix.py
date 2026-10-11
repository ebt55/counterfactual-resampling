"""Prefix rebuild and replay (build guide §3; the day-0 "Replay" check of §1).

Three jobs, all keyless:

* :func:`build_messages` turns a KM record's ``transcript[:k]`` back into the
  ``(system, list[Msg])`` pair the agent loop held just before the blocked call's
  result, so a continuation resumes from exactly what was sent.
* :func:`replay` re-executes the prefix tool calls ``tool_calls[:i]`` in a fresh env-v2
  ``DockerSandbox`` and checks each output against the recording after
  :func:`normalise`. The sandbox is returned open, so the caller can place a file and
  re-run the blocked call in the same container.
* :func:`replay_check` is the day-0 gate: 20 seeded events, one container each, a
  table in ``results/replay_check.md``. ``python -m rab.prefix --check 20``.

What :func:`build_messages` drops, relative to the recorded transcript, and why that is
still exactly what KM sent (``OpenAICompatProvider._to_api`` is the serialiser):

1. ``transcript[0]`` (the system entry) becomes the ``system`` string; its empty
   ``tool_calls`` / ``tool_call_id`` keys carry nothing.
2. Every entry's empty ``tool_calls: []`` and ``tool_call_id: None`` are not sent.
3. An assistant entry with tool calls and ``content == ""`` is sent as
   ``"content": null`` (``_to_api`` does this itself).
4. ``tool_calls[].args`` (a dict in the transcript) is sent as
   ``function.arguments = json.dumps(args, ensure_ascii=False)`` plus
   ``"type": "function"``. KM parsed the model's arguments string on receipt and
   re-serialised the dict on every later request, so the model's original string was
   never resent and is not recoverable; the re-serialised one is what was sent.
5. Reasoning: ``OpenAICompatProvider.chat`` reads only ``content``, ``tool_calls``,
   ``refusal``, ``finish_reason`` and ``usage`` from a response. The reasoning text is
   therefore in no record and in no later request; only ``usage.reasoning_tokens``
   shows it existed. Assistant text was ``.strip()``-ed before it was stored and before
   it was resent, so the stored text is the sent text.
6. ``Msg.tool_name`` is not stored in a transcript; ``harness.continuous.rebuild_messages``
   recovers it from ``tool_call_id``. ``_to_api`` never sends it.

Nothing else differs: the round trip is tested in ``tests/test_prefix.py``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rab import REPO_ROOT
from rab import km
from rab.constants import ENV_VERSION, SEED
from harness.agent import NUDGE_TEXT, ToolExecutor
from harness.config import IMAGE, ITEM_WALL_CLOCK_S, MAX_NUDGES, MAX_TURNS
from harness.continuous import rebuild_messages
from harness.providers.base import Msg
from harness.sandbox import DockerSandbox, docker_available, image_available

EVENTS_PATH = REPO_ROOT / "data" / "events.jsonl"
REPLAY_REPORT = REPO_ROOT / "results" / "replay_check.md"

#: Guide §1 pass rule: 19 of 20 events reproduce every prefix output after
#: normalisation. Kept as a share so ``--check N`` scales (ceil(0.95 * 20) == 19).
#: TODO(chunk A): move to ``rab.constants`` (chunk B may not edit that file).
REPLAY_PASS_SHARE = 0.95

#: Container names for replays: never ``dc_`` (``DockerSandbox.start`` runs
#: ``docker rm -f <name>`` first, so a KM-style name could clobber a KM container).
NAME_PREFIX = "rab_"
EXCERPT_CHARS = 200


# ------------------------------------------------------------------ messages

def build_messages(rec: dict, k: int) -> tuple[str, list[Msg]]:
    """``(system, messages)`` as KM held them before ``transcript[k]`` existed.

    ``system = transcript[0]["content"]`` (the frozen ``rules.md`` text);
    ``messages = transcript[1:k]`` as ``Msg`` objects via KM's own
    ``harness.continuous.rebuild_messages`` (``ToolCall(id, name, args)`` on assistant
    entries, ``tool_call_id`` and the recovered ``tool_name`` on tool entries;
    ``transcript[1]`` is the manifest with notes; nudge user messages stay).
    For an event, ``k`` is the blocked call's tool entry, so the last message is the
    assistant turn that issued the blocked call. See the module docstring for what
    the wire form drops.
    """
    transcript = rec["transcript"]
    if not 1 <= k <= len(transcript):
        raise ValueError(f"k={k} outside 1..{len(transcript)}")
    if transcript[0].get("role") != "system":
        raise ValueError("transcript[0] is not the system entry")
    system = transcript[0]["content"]
    return system, rebuild_messages([{"transcript": transcript[:k]}])


def prefix_counts(rec: dict, event: dict) -> tuple[int, int]:
    """``(prefix_turns, prefix_nudges)`` at the block point.

    ``prefix_turns`` = assistant entries in ``transcript[1:k]``: KM's ``n_turns`` counts
    ``provider.chat`` calls and each call appends exactly one assistant entry, the
    last of them being the turn that issued the blocked call. It must equal the
    blocked call's recorded ``tool_calls[i]["turn"]``; a mismatch raises.
    ``prefix_nudges`` = user entries in ``transcript[2:k]`` (``transcript[1]`` is the
    manifest; in the per-item arms every later user entry is a ``NUDGE_TEXT`` nudge,
    which is asserted).
    """
    k, i = int(event["k"]), int(event["i"])
    head = rec["transcript"][:k]
    turns = sum(1 for e in head[1:] if e["role"] == "assistant")
    users = [e for e in head[2:] if e["role"] == "user"]
    if any(e["content"] != NUDGE_TEXT for e in users):
        raise ValueError(f"{event.get('event_id')}: a user entry after the manifest is not a nudge")
    recorded_turn = rec["tool_calls"][i]["turn"]
    if turns != recorded_turn:
        raise ValueError(f"{event.get('event_id')}: {turns} assistant entries before k "
                         f"but tool_calls[{i}].turn == {recorded_turn}")
    return turns, len(users)


def continuation_budget(rec: dict, event: dict) -> dict:
    """What a continuation from the block point may still spend (guide §3).

    ``max_turns = 30 - prefix_turns``, ``max_nudges = 2 - prefix_nudges``,
    ``wall_clock_s = 600`` (fresh). 30 and 600 are the record's own
    ``model_config.max_turns`` / ``item_wall_clock_s`` (checked against KM's
    ``MAX_TURNS`` / ``ITEM_WALL_CLOCK_S``), 2 is KM's ``MAX_NUDGES``. See
    :func:`prefix_counts` for the two prefix definitions.
    """
    mc = rec["model_config"]
    max_turns = int(mc.get("max_turns", MAX_TURNS))
    wall = int(mc.get("item_wall_clock_s", ITEM_WALL_CLOCK_S))
    if (max_turns, wall) != (MAX_TURNS, ITEM_WALL_CLOCK_S):
        raise ValueError(f"record limits {max_turns}/{wall}s differ from KM's "
                         f"{MAX_TURNS}/{ITEM_WALL_CLOCK_S}s")
    turns, nudges = prefix_counts(rec, event)
    return {"max_turns": max_turns - turns, "max_nudges": MAX_NUDGES - nudges,
            "wall_clock_s": wall, "prefix_turns": turns, "prefix_nudges": nudges}


# ------------------------------------------------------------------ normalise

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

#: One ``ls -l`` line in GNU ls's default C-locale format, anchored at line start:
#: file type + 9 permission chars (+ optional ``.``/``+``/``@``), link count, owner and
#: group (one token under ``-g``/``-o``/``-G``), size (bytes, ``maj, min`` for a
#: device, or ``-h`` form like ``4.0K``), then the date column: ``Mon DD HH:MM`` (recent)
#: or ``Mon DD  YYYY`` (older than six months or in the future), then a space and the
#: name. Only the date column is replaced. Deliberately NOT covered, so they stay
#: failures: ``--full-time`` / ``--time-style`` / ISO dates, ``ls -s`` block prefixes,
#: ``find -ls`` / ``stat`` output, ``total N`` lines, and any date anywhere else.
LS_DATE_RE = re.compile(
    r"^(?P<head>[-bcdlps][rwxsStT-]{9}[.+@]?[ ]+\d+[ ]+(?:\S+[ ]+){1,2}"
    r"(?:\d+,[ ]*)?\d+(?:\.\d+)?[KMGTPE]?[ ]+)"
    rf"(?P<date>{_MONTH}[ ]+\d{{1,2}}[ ]+(?:\d{{2}}:\d{{2}}|\d{{4}}))"
    r"(?P<tail>[ ].*)$",
    re.M,
)
LS_DATE_TOKEN = "<ls-date>"

#: A pytest summary line: names a pytest outcome and ends its count with ``in N.NNs``.
#: Only the ``N.NN`` is replaced; the line, the counts and the ``=`` padding stay.
PYTEST_TIME_RE = re.compile(
    r"^(?P<head>.*\b(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|"
    r"deselected|no tests ran)\b.* in )\d+\.\d+(?P<tail>s\b.*)$",
    re.M,
)
PYTEST_TIME_TOKEN = "<t>"


def normalise(text: str) -> str:
    """Erase ``ls -l`` dates and pytest timings; leave every other character alone.

    Guide §1: ``normalise`` touches only ``ls -l`` dates and pytest timings
    (``in \\d+\\.\\d+s``); any other difference between a replayed and a recorded output
    is a replay failure by definition. See :data:`LS_DATE_RE` for the exact ``ls -l``
    shape (deliberately narrow) and :data:`PYTEST_TIME_RE` for the timing.
    """
    if not text:
        return text or ""
    text = LS_DATE_RE.sub(lambda m: m["head"] + LS_DATE_TOKEN + m["tail"], text)
    return PYTEST_TIME_RE.sub(lambda m: m["head"] + PYTEST_TIME_TOKEN + m["tail"], text)


def diff_excerpt(recorded: str, replayed: str, width: int = 4 * EXCERPT_CHARS) -> str:
    """The first lines that differ, as a zero-context unified diff ('-' recorded)."""
    if recorded == replayed:
        return ""
    lines = list(difflib.unified_diff(recorded.splitlines(), replayed.splitlines(),
                                      n=0, lineterm=""))[2:]
    text = "\n".join(lines) if lines else (
        f"line ends differ: recorded ends {recorded[-20:]!r}, replayed ends {replayed[-20:]!r}")
    return text[:width]


# ------------------------------------------------------------------ replay

@dataclass
class CallDiff:
    """One replayed prefix call (build plan "Interfaces"; chunk C moves it to types.py)."""

    index: int            # position in rec["tool_calls"]
    tool: str             # "bash" | "read_file" | "write_file" | "run_tests"
    matched: bool         # normalise(replayed) == normalise(recorded), same truncated flag
    diff_excerpt: str     # zero-context unified diff of the normalised texts, "" if matched


@dataclass
class Replay:
    """Outcome of :func:`replay` (build plan "Interfaces").

    ``sandbox`` is open: the prefix calls ``[:i]`` have run, ``/work`` is touched, the
    blocked call has NOT run. :meth:`close` removes the container. The fields after
    ``ok`` are extras with defaults that :func:`replay_check` reads.
    """

    sandbox: DockerSandbox
    item_key: str
    task_dir: Path
    batch_id: str
    calls: list[CallDiff]
    ok: bool
    event_id: str = ""
    replayed: list[str] = field(default_factory=list)   # raw replayed outputs, by index
    work_listing: str = ""                             # `ls -ld /work` after the touch
    notes: list[str] = field(default_factory=list)     # e.g. a failed touch

    @property
    def first_failure(self) -> CallDiff | None:
        return next((c for c in self.calls if not c.matched), None)

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self) -> "Replay":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def replay_batch_id(event_id: str) -> str:
    """``<event_id>-replay-<6 hex>``: unique per replay, so two slots of one event
    running at once never share a container name (``start()`` would ``rm -f`` the other)."""
    return f"{event_id}-replay-{uuid.uuid4().hex[:6]}"


def new_sandbox(batch_id: str, image: str = IMAGE) -> DockerSandbox:
    """A fresh env-v2 ``DockerSandbox`` named ``rab_<batch_id sanitised>``, not started."""
    return DockerSandbox(batch_id, tasks_dir=km.TASKS_DIR, image=image,
                         name_prefix=NAME_PREFIX, env_version=ENV_VERSION)


def replay(rec: dict, event: dict, sandbox: DockerSandbox | None = None, *,
           batch_id: str | None = None) -> Replay:
    """Rebuild the work dir at the block point and check every prefix output.

    1. A fresh ``DockerSandbox`` (env v2, the record's image, ``name_prefix="rab_"``,
       ``batch_id`` default :func:`replay_batch_id`) unless ``sandbox`` is given.
    2. ``prepare_item(item_key, task_dir, arm="baseline", env_variant="salient",
       position=rec["position"])``: the work dir is ``/work/item_NN``; ``task_dir`` is
       ``tasks/<item_key>`` (solvable) or ``tasks/impossible/<item_key>``.
    3. Each ``rec["tool_calls"][:event["i"]]`` through ``ToolExecutor.run(tool, input)``
       with the record's ``bash_timeout_s`` (60) and ``tool_output_limit`` (8000), so
       KM's own truncation applies; a call matches when
       ``normalise(replayed) == normalise(recorded)`` and the truncated flags agree.
       A mismatch does not stop the replay (later calls still run and are checked).
    4. ``touch -d "<work_mtime>" /work`` when the event carries ``work_mtime`` (the
       ``..`` line of a prefix ``ls -l`` in ``/work/item_NN`` is ``/work``); runs as the
       sandbox user ``dc``, who owns ``/work``. A failed touch makes ``ok`` false.

    The sandbox is left open in the result. If anything raises, a sandbox created
    here is removed before the error propagates; a passed-in sandbox is the caller's.
    """
    item_key, position = rec["item_key"], int(rec["position"])
    if (event["item_key"], int(event["position"])) != (item_key, position):
        raise ValueError(f"event {event['event_id']} is {event['item_key']}@{event['position']}, "
                         f"record is {item_key}@{position}")
    mc = rec["model_config"]
    if (mc.get("env_version"), mc.get("env_variant"), rec.get("arm")) != (
            ENV_VERSION, "salient", "baseline"):
        raise ValueError(f"{event['event_id']}: not a baseline / salient / env-v2 record")
    tdir = km.task_dir(item_key, rec.get("mutation") or "solvable")
    i = int(event["i"])

    own = sandbox is None
    if own:
        sandbox = new_sandbox(batch_id or replay_batch_id(event["event_id"]),
                              image=mc.get("image", IMAGE))
    try:
        sandbox.start()
        sandbox.prepare_item(item_key, tdir, arm="baseline", env_variant="salient",
                             position=position)
        executor = ToolExecutor(sandbox, item_key, int(mc["bash_timeout_s"]),
                                int(mc["tool_output_limit"]))
        calls: list[CallDiff] = []
        outputs: list[str] = []
        for idx, call in enumerate(rec["tool_calls"][:i]):
            out, truncated = executor.run(call["tool"], call["input"])
            outputs.append(out)
            a, b = normalise(call["output"]), normalise(out)
            matched = a == b and bool(truncated) == bool(call["truncated"])
            excerpt = diff_excerpt(a, b) if a != b else (
                "" if matched else f"truncated flag: recorded {call['truncated']}, replayed {truncated}")
            calls.append(CallDiff(idx, call["tool"], matched, excerpt))

        notes: list[str] = []
        listing = ""
        work_mtime = event.get("work_mtime")
        if work_mtime:
            res = sandbox.exec(item_key, f"touch -d {shlex.quote(str(work_mtime))} /work && ls -ld /work")
            listing = res.output.strip()
            if res.exit_code != 0:
                notes.append(f"touch -d {work_mtime!r} /work failed: {listing[:200]}")
        ok = all(c.matched for c in calls) and not notes
        return Replay(sandbox, item_key, tdir, sandbox.batch_id, calls, ok,
                      event_id=event["event_id"], replayed=outputs, work_listing=listing,
                      notes=notes)
    except BaseException:
        if own:
            sandbox.close()
        raise


# ------------------------------------------------------------------ day-0 check

def load_events(path: Path = EVENTS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def record_for(event: dict) -> dict:
    """The KM record an event came from (``results/runs/<run_id>/items.jsonl``)."""
    run_id = str(event["run_id"]).split("/")[0]
    rec = km.load_record(run_id, int(event["position"]))
    if rec["item_key"] != event["item_key"]:
        raise ValueError(f"{event['event_id']}: record at {run_id}@{event['position']} is "
                         f"{rec['item_key']}, event says {event['item_key']}")
    return rec


def draw(events: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """``random.Random(seed).sample`` of ``n`` events, from event ids in sorted order."""
    by_id = {e["event_id"]: e for e in events}
    return [by_id[eid] for eid in random.Random(seed).sample(sorted(by_id), n)]


_LS_NLINK = re.compile(r"^([-bcdlps][rwxsStT-]{9}[.+@]?)[ ]+(\d+)[ ]+(.*)$")
_STDERR_DIAG = re.compile(r"^[\w.+/-]+: .+$")   # "cat: data/x.md: No such file or directory"


def classify(recorded: str, replayed: str) -> str:
    """A first-pass label for a failed call (both texts already normalised).

    Mechanical, so a reader can check it; first match wins:

    * ``ordering (stdout/stderr interleaving)``: the same lines in another order, and
      every moved line is a ``prog: ...`` diagnostic; plain ``ordering`` otherwise.
    * ``timeout``: one side carries KM's ``[command timed out ...]`` head.
    * ``truncation boundary``: exactly one side was truncated.
    * ``lines only in the recording`` / ``lines only in the replay``.
    * ``ls -l link count (/proc nlink)``: paired ``ls -l`` lines equal once the link
      count and its column padding are dropped, and a ``proc`` entry's count differs
      (procfs reports every process in the Docker VM there); ``ls -l link count``
      if no ``proc`` line is involved.
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
        moved = gone + new
        if moved and all(_STDERR_DIAG.match(x) for x in moved):
            return "ordering (stdout/stderr interleaving)"
        return "ordering"
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


def _cell(text: str, width: int = EXCERPT_CHARS) -> str:
    text = text[:width].replace("\\", "\\\\").replace("|", "\\|").replace("`", "'")
    return text.replace("\r", "").replace("\n", " \u23ce ")


def replay_check(n: int = 20, seed: int = SEED, events_path: Path = EVENTS_PATH,
                 out_path: Path = REPLAY_REPORT) -> dict:
    """Day-0 gate (guide §1): replay ``n`` seeded events, one container each, in turn.

    Writes ``out_path`` (a per-event table and the pass count) and returns a summary.
    Every container is removed before the next event starts, whatever happens.
    """
    if not docker_available():
        raise RuntimeError("Docker is not available")
    if not image_available(IMAGE):
        raise RuntimeError(f"image {IMAGE} missing: docker build -t {IMAGE} harness/docker in KM_ROOT")
    events = load_events(events_path)
    picked = draw(events, n, seed)
    rows: list[dict] = []
    t_all = time.time()
    for j, ev in enumerate(picked):
        t0 = time.time()
        row = {"event_id": ev["event_id"], "kind": ev["kind"], "prefix_calls": int(ev["i"]),
               "matched": 0, "ok": False, "first_fail": "", "excerpt": "", "class": "",
               "work": "", "error": ""}
        res = None
        try:
            rec = record_for(ev)
            res = replay(rec, ev)
            row["matched"] = sum(c.matched for c in res.calls)
            row["ok"] = res.ok
            row["work"] = res.work_listing
            bad = res.first_failure
            if bad is not None:
                rec_out = normalise(rec["tool_calls"][bad.index]["output"])
                rep_out = normalise(res.replayed[bad.index])
                row["first_fail"] = f"{bad.index} ({bad.tool})"
                row["excerpt"] = bad.diff_excerpt
                row["class"] = classify(rec_out, rep_out)
                row["failed_calls"] = [
                    {"index": c.index, "name": c.tool,
                     "class": classify(normalise(rec["tool_calls"][c.index]["output"]),
                                       normalise(res.replayed[c.index])),
                     "excerpt": c.diff_excerpt}
                    for c in res.calls if not c.matched]
            elif res.notes:
                row["first_fail"] = "touch"
                row["excerpt"] = "; ".join(res.notes)
                row["class"] = "host-side touch"
        except Exception as exc:  # recorded, never fatal to the check
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["class"] = "replay error"
            row["excerpt"] = row["error"]
        finally:
            if res is not None:
                res.close()
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{j + 1}/{n}] {ev['event_id']} i={ev['i']} ok={row['ok']} "
              f"matched={row['matched']}/{ev['i']} {row['class']} {row['seconds']}s",
              file=sys.stderr, flush=True)

    passed = sum(r["ok"] for r in rows)
    need = math.ceil(REPLAY_PASS_SHARE * n)
    summary = {"n": n, "seed": seed, "passed": passed, "need": need,
               "pass": passed >= need, "seconds": round(time.time() - t_all, 1),
               "image_id": _image_id(IMAGE), "rows": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_report(summary, events_path), encoding="utf-8", newline="\n")
    return summary


def _image_id(image: str) -> str:
    try:
        out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
                             timeout=60)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _report(s: dict, events_path: Path) -> str:
    rel = events_path.resolve().relative_to(REPO_ROOT) if events_path.resolve().is_relative_to(
        REPO_ROOT) else events_path
    verdict = "PASS" if s["pass"] else "FAIL"
    lines = [
        "# Replay check (guide §1 \"Replay\", §3)",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        f"`uv run python -m rab.prefix --check {s['n']}` (seed {s['seed']}), "
        f"from `{rel.as_posix() if isinstance(rel, Path) else rel}`.",
        f"Image `{IMAGE}` = `{s['image_id']}`. One fresh env-v2 container per event, "
        f"run in turn; every container removed after its event. {s['seconds']} s in all.",
        "",
        f"**{s['passed']} of {s['n']} events reproduce every prefix output after "
        f"`normalise`; the rule needs {s['need']}. {verdict}.**",
        "",
        "`normalise` erases only the date column of `ls -l` lines and the `N.NN` of a "
        "pytest `in N.NNs` summary (`src/rab/prefix.py`, `LS_DATE_RE`, "
        "`PYTEST_TIME_RE`); any other difference is a failure. The excerpt is a "
        "zero-context unified diff of the first failing call (`-` recorded, `+` "
        "replayed), first 200 characters, newlines shown as \u23ce.",
        "",
        "| event_id | kind | prefix calls | matched | first failing call | class | diff excerpt |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        lines.append(f"| `{r['event_id']}` | {r['kind']} | {r['prefix_calls']} | "
                     f"{r['matched']} | {r['first_fail'] or '-'} | {r['class'] or '-'} | "
                     f"{_cell(r['excerpt']) or '-'} |")
    failed = [r for r in s["rows"] if not r["ok"]]
    if failed:
        lines += ["", "## Every failing call", "",
                  "| event_id | call | class | diff excerpt |", "|---|---|---|---|"]
        for r in failed:
            for c in r.get("failed_calls") or [{"index": r["first_fail"], "name": "",
                                                 "class": r["class"], "excerpt": r["excerpt"]}]:
                lines.append(f"| `{r['event_id']}` | {c['index']} {c['name']} | {c['class']} | "
                             f"{_cell(c['excerpt'])} |")
    lines += ["", "## `/work` after `touch -d <work_mtime>`", "",
              "| event_id | `ls -ld /work` |", "|---|---|"]
    for r in s["rows"]:
        lines.append(f"| `{r['event_id']}` | {_cell(r['work']) or '(no work_mtime on the event)'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rab.prefix", description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=int, metavar="N", required=True,
                    help="replay N seeded events and write results/replay_check.md")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--events", type=Path, default=EVENTS_PATH)
    ap.add_argument("--out", type=Path, default=REPLAY_REPORT)
    args = ap.parse_args(argv)
    s = replay_check(args.check, args.seed, args.events, args.out)
    print(f"{s['passed']} of {s['n']} events replayed clean (need {s['need']}): "
          f"{'PASS' if s['pass'] else 'FAIL'}; wrote {args.out}", file=sys.stderr)
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
