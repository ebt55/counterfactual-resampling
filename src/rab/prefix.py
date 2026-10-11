"""Prefix rebuild and replay (build guide §3; the day-0 "Replay" check of §1).

Three jobs, all keyless:

* :func:`build_messages` turns a KM record's ``transcript[:k]`` back into the
  ``(system, list[Msg])`` pair the agent loop held just before the blocked call's
  result, so a continuation resumes from exactly what was sent.
* :func:`replay` re-executes the prefix tool calls ``tool_calls[:i]`` in a fresh env-v2
  ``DockerSandbox`` and checks each output against the recording, strictly
  (:func:`normalise`) and by content (:func:`content_equal`). The sandbox is returned
  open, so the caller can place a file and re-run the blocked call in the same
  container.
* The day-0 replay check and the eligibility sweep (``data/replay_sweep.json``, which
  ``rab.select`` reads) live in ``tools/replay_check.py``:
  ``uv run python tools/replay_check.py --check 20`` / ``--sweep``.

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

import difflib
import json
import re
import shlex
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rab import REPO_ROOT
from rab import km
from rab.constants import ENV_VERSION
from harness.agent import NUDGE_TEXT, ToolExecutor
from harness.config import IMAGE, ITEM_WALL_CLOCK_S, MAX_NUDGES, MAX_TURNS
from harness.continuous import rebuild_messages
from harness.providers.base import Msg
from harness.sandbox import DockerSandbox

EVENTS_PATH = REPO_ROOT / "data" / "events.jsonl"


#: Container names for replays: never ``dc_`` (``DockerSandbox.start`` runs
#: ``docker rm -f <name>`` first, so a KM-style name could clobber a KM container).
NAME_PREFIX = "rab_"
EXCERPT_CHARS = 200

#: KM runs a whole batch in one container and ends every item with ``hidden_grade``
#: (pytest), which creates and deletes a temp file in ``/tmp``. overlayfs then reports
#: the copied-up ``/tmp`` with link count 1 for the rest of the batch, where a fresh
#: container shows 2 (checked: 2 after ``prepare_item``, 1 after one ``run_visible_tests``).
#: For an event at position 1 or later the replay therefore creates and deletes one temp
#: file in ``/tmp`` as the sandbox user before the prefix: the same trace, nothing left.
TMP_TRACE_CMD = "python -c 'import tempfile; tempfile.TemporaryFile().close()'"
TMP_TRACE_NOTE = "previous item's /tmp trace reproduced (temp file created and deleted)"


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
    notes: list[str] = field(default_factory=list)     # steps taken (the /tmp trace) and failures
    failures: list[str] = field(default_factory=list)  # a failed /tmp trace or touch; ok is false

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
    2b. For ``position >= 1``: :data:`TMP_TRACE_CMD`, which reproduces the trace the
       previous item's grading left in the batch container (see its comment), with a
       ``notes`` entry. Position 0 had no previous item, so ``/tmp`` stays pristine.
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
        notes: list[str] = []
        failures: list[str] = []
        if position >= 1:
            res = sandbox.exec(item_key, TMP_TRACE_CMD)
            if res.exit_code == 0:
                notes.append(TMP_TRACE_NOTE)
            else:
                failures.append(f"/tmp trace failed: {res.output.strip()[:200]}")
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

        listing = ""
        work_mtime = event.get("work_mtime")
        if work_mtime:
            res = sandbox.exec(item_key, f"touch -d {shlex.quote(str(work_mtime))} /work && ls -ld /work")
            listing = res.output.strip()
            if res.exit_code != 0:
                failures.append(f"touch -d {work_mtime!r} /work failed: {listing[:200]}")
        ok = all(c.content_matched for c in calls) and not failures
        strict_ok = all(c.matched for c in calls) and not failures
        return Replay(sandbox, item_key, tdir, sandbox.batch_id, calls, ok, strict_ok,
                      event_id=event["event_id"], replayed=outputs, work_listing=listing,
                      notes=notes + failures, failures=failures)
    except BaseException:
        if own:
            sandbox.close()
        raise


# ------------------------------------------------------------------ records

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
