"""Prefix rebuild and replay (build guide §3; the day-0 "Replay" check of §1).

Four jobs, all keyless:

* :func:`build_messages` turns a KM record's ``transcript[:k]`` back into the
  ``(system, list[Msg])`` pair the agent loop held just before the blocked call's
  result, so a continuation resumes from exactly what was sent.
* :func:`replay` re-executes the prefix tool calls ``tool_calls[:i]`` in a fresh env-v2
  ``DockerSandbox`` and checks each output against the recording, strictly
  (:func:`normalise`) and by content (:func:`content_equal`). The sandbox is returned
  open, so the caller can place a file and re-run the blocked call in the same
  container.
* :func:`replay_check` is the day-0 gate: 20 seeded events, one container each, a
  table in ``results/replay_check.md``. ``python -m rab.prefix --check 20``.
* :func:`sweep` replays every eligible event (before the draw) and writes
  ``data/replay_sweep.json``, which ``rab.select`` reads to exclude events whose prefix
  does not replay. ``python -m rab.prefix --sweep``.

Two comparators, both recorded everywhere:

* strict, ``normalise(recorded) == normalise(replayed)`` (``CallDiff.matched``,
  ``Replay.strict_ok``): the guide's rule, ``ls -l`` dates and pytest timings, plus one
  departure, the ``/proc`` link count (see :func:`normalise`).
* content, :func:`content_equal` (``CallDiff.content_matched``, ``Replay.ok``): strict,
  or the same lines with only ``prog: message`` diagnostics moved, because
  ``docker exec`` carries stdout and stderr on separate streams and their merge order
  is a race (``results/replay_check.md``).

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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rab import REPO_ROOT
from rab import km
from rab.constants import ENV_VERSION, KINDS, REPLAY_PASS_SHARE, SEED
from harness.agent import NUDGE_TEXT, ToolExecutor
from harness.config import IMAGE, ITEM_WALL_CLOCK_S, MAX_NUDGES, MAX_TURNS
from harness.continuous import rebuild_messages
from harness.providers.base import Msg
from harness.sandbox import DockerSandbox, docker_available, image_available

EVENTS_PATH = REPO_ROOT / "data" / "events.jsonl"
SWEEP_PATH = REPO_ROOT / "data" / "replay_sweep.json"
REPLAY_REPORT = REPO_ROOT / "results" / "replay_check.md"

#: Guide §1 pass rule (19 of 20 events), kept as a share so ``--check N`` scales;
#: defined in ``rab.constants``.

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

#: Departure from the guide (PREREG): an ``ls -l`` line already recognised by
#: :data:`LS_DATE_RE` (its date is the token now), split at the link count. procfs
#: reports the number of processes in the whole Docker VM as ``/proc``'s link count,
#: and that count also sets the padding of the link column for every line of the same
#: listing. So on every ``ls -l`` line the padding between the permission string and the
#: link count becomes one space, and on a line whose name is exactly ``proc`` the link
#: count becomes :data:`PROC_NLINK_TOKEN`. No other link count is touched (``/tmp``'s
#: 1 vs 2 stays a failure: it records whether anything was written there).
LS_NLINK_RE = re.compile(
    r"^(?P<perm>[-bcdlps][rwxsStT-]{9}[.+@]?)[ ]+(?P<nlink>\d+)"
    r"(?P<rest>[ ]+(?:\S+[ ]+){1,2}(?:\d+,[ ]*)?\d+(?:\.\d+)?[KMGTPE]?[ ]+"
    + re.escape(LS_DATE_TOKEN) + r"(?P<name>[ ].*))$",
    re.M,
)
PROC_NLINK_TOKEN = "<nproc>"

#: A ``prog: message`` diagnostic line, e.g. ``cat: data/x.md: No such file or directory``.
DIAGNOSTIC_RE = re.compile(r"^\S+: .*")


def _nlink(m: re.Match) -> str:
    nlink = PROC_NLINK_TOKEN if m["name"] == " proc" else m["nlink"]
    return f"{m['perm']} {nlink}{m['rest']}"


def normalise(text: str) -> str:
    """Erase ``ls -l`` dates, pytest timings and ``/proc``'s link count.

    Guide §1: ``normalise`` touches only ``ls -l`` dates and pytest timings
    (``in \\d+\\.\\d+s``); any other difference between a replayed and a recorded output
    is a replay failure. One recorded departure (PREREG): the ``/proc`` link count and
    the link-column padding it drives (:data:`LS_NLINK_RE`). See :data:`LS_DATE_RE` for
    the exact ``ls -l`` shape (deliberately narrow) and :data:`PYTEST_TIME_RE`.
    """
    if not text:
        return text or ""
    text = LS_DATE_RE.sub(lambda m: m["head"] + LS_DATE_TOKEN + m["tail"], text)
    text = LS_NLINK_RE.sub(_nlink, text)
    return PYTEST_TIME_RE.sub(lambda m: m["head"] + PYTEST_TIME_TOKEN + m["tail"], text)


def content_equal(recorded: str, replayed: str) -> bool:
    """True when the outputs differ at most by where ``prog: message`` lines landed.

    Either ``normalise`` makes them equal, or the normalised texts have the same length
    and the same multiset of lines, and the lines that do not match
    :data:`DIAGNOSTIC_RE` appear in the same order on both sides (so every line whose
    position moved is a diagnostic). ``docker exec`` carries the container's stdout and
    stderr on separate streams and ``ToolExecutor`` merges them as they arrive, so where
    a stderr line lands among stdout lines is a race; a changed byte in any line, or a
    moved non-diagnostic line, is still a difference.
    """
    a, b = normalise(recorded), normalise(replayed)
    if a == b:
        return True
    la, lb = a.splitlines(), b.splitlines()
    if len(a) != len(b) or Counter(la) != Counter(lb):
        return False
    return ([x for x in la if not DIAGNOSTIC_RE.match(x)]
            == [x for x in lb if not DIAGNOSTIC_RE.match(x)])


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

    index: int             # position in rec["tool_calls"]
    tool: str              # "bash" | "read_file" | "write_file" | "run_tests"
    matched: bool          # strict: normalise() equal and the same truncated flag
    content_matched: bool  # content_equal() and the same truncated flag
    diff_excerpt: str      # zero-context unified diff of the normalised texts, "" if equal


@dataclass
class Replay:
    """Outcome of :func:`replay` (build plan "Interfaces").

    ``sandbox`` is open: the prefix calls ``[:i]`` have run, ``/work`` is touched, the
    blocked call has NOT run. :meth:`close` removes the container. ``ok`` is every call
    ``content_matched`` (and the touch worked); ``strict_ok`` is every call ``matched``.
    The fields after ``ok`` are extras with defaults.
    """

    sandbox: DockerSandbox
    item_key: str
    task_dir: Path
    batch_id: str
    calls: list[CallDiff]
    ok: bool
    strict_ok: bool = False
    event_id: str = ""
    replayed: list[str] = field(default_factory=list)   # raw replayed outputs, by index
    work_listing: str = ""                             # `ls -ld /work` after the touch
    notes: list[str] = field(default_factory=list)     # e.g. a failed touch

    @property
    def first_failure(self) -> CallDiff | None:
        """First call that fails the strict comparison."""
        return next((c for c in self.calls if not c.matched), None)

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self) -> Replay:
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
       KM's own truncation applies; each call is compared strictly and by content
       (module docstring). A mismatch does not stop the replay.
    4. ``touch -d "<work_mtime>" /work`` when the event carries ``work_mtime`` (the
       ``..`` line of a prefix ``ls -l`` in ``/work/item_NN`` is ``/work``); runs as the
       sandbox user ``dc``, who owns ``/work``. A failed touch makes both oks false.

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
            same_flag = bool(truncated) == bool(call["truncated"])
            a, b = normalise(call["output"]), normalise(out)
            matched = a == b and same_flag
            content = content_equal(call["output"], out) and same_flag
            excerpt = diff_excerpt(a, b) if a != b else (
                "" if same_flag else f"truncated flag: recorded {call['truncated']}, replayed {truncated}")
            calls.append(CallDiff(idx, call["tool"], matched, content, excerpt))

        notes: list[str] = []
        listing = ""
        work_mtime = event.get("work_mtime")
        if work_mtime:
            res = sandbox.exec(item_key, f"touch -d {shlex.quote(str(work_mtime))} /work && ls -ld /work")
            listing = res.output.strip()
            if res.exit_code != 0:
                notes.append(f"touch -d {work_mtime!r} /work failed: {listing[:200]}")
        ok = all(c.content_matched for c in calls) and not notes
        strict_ok = all(c.matched for c in calls) and not notes
        return Replay(sandbox, item_key, tdir, sandbox.batch_id, calls, ok, strict_ok,
                      event_id=event["event_id"], replayed=outputs, work_listing=listing,
                      notes=notes)
    except BaseException:
        if own:
            sandbox.close()
        raise


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


# ------------------------------------------------------------------ check and sweep

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
        if res.notes:
            row["failed_calls"].append({"index": "touch", "name": "", "class": "host-side touch",
                                        "content_matched": False, "excerpt": "; ".join(res.notes)})
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
    doc = {"image": _image_id(IMAGE), "generated": _utc_now(),
           "seconds": round(time.time() - t_all, 1),
           "rule": "ok = every prefix call content_equal (rab.prefix) and the /work touch "
                   "succeeded; strict_ok = every call equal after normalise",
           "events": entries, "totals": totals}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    return doc


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        f"`uv run python -m rab.prefix --check {s['n']}` (seed {s['seed']}), from `{rel}`.",
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
    ap = argparse.ArgumentParser(prog="rab.prefix", description=__doc__.splitlines()[0])
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
