# Replay check (guide §1 "Replay", §3)

Generated 2026-10-11 05:50 UTC by `uv run python -m rab.prefix --check 20` (seed 20261011), from `data/events.jsonl`.
Image `dosecurve-sandbox:latest` = `sha256:bafc8ec7eef84467c92dc60c576732538b79553bb1d6e1b7402d37e3fe19dddf`. One fresh env-v2 container per event, run in turn; every container removed after its event. 26.7 s in all.

**Content: 20 of 20 events reproduce every prefix output (`content_equal`); the rule needs 19. PASS. Strict (`normalise` only): 20 of 20.**

Strict: `normalise(recorded) == normalise(replayed)`, where `normalise` erases the date column of `ls -l` lines, the `N.NN` of a pytest `in N.NNs` summary and (a PREREG departure) the `/proc` link count with the link-column padding (`LS_DATE_RE`, `PYTEST_TIME_RE`, `LS_NLINK_RE` in `src/rab/prefix.py`). Content: strict, or the same lines where only `prog: message` diagnostics moved (`content_equal`; stdout and stderr reach `docker exec` on separate streams). Any other difference is a failure. The excerpt is a zero-context unified diff of the first call failing the strict comparison (`-` recorded, `+` replayed), first 200 characters, newlines shown as ⏎.

| event_id | kind | prefix calls | strict matched | content matched | ok | first strict failure | class | diff excerpt |
|---|---|---|---|---|---|---|---|---|
| `dsv41flash-sal-v2_baseline_I15_s9_p14` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s5_p20` | solvable | 2 | 2 | 2 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s9_p15` | unreachable | 9 | 9 | 9 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s5_p15` | unreachable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s3_p10` | unreachable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p12` | unreachable | 2 | 2 | 2 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p10` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s5_p22` | solvable | 3 | 3 | 3 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s9_p13` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p09` | unreachable | 2 | 2 | 2 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s7_p02` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p06` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s9_p04` | unreachable | 2 | 2 | 2 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I8_s5_p15` | unreachable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I8_s10_p15` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s6_p07` | solvable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s2_p02` | unreachable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s6_p06` | solvable | 4 | 4 | 4 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s10_p09` | unreachable | 1 | 1 | 1 | yes | - | - | - |
| `dsv41flash-sal-v2_baseline_I2_s0_p02` | unreachable | 1 | 1 | 1 | yes | - | - | - |

## `/work` after `touch -d <work_mtime>`

| event_id | `ls -ld /work` |
|---|---|
| `dsv41flash-sal-v2_baseline_I15_s9_p14` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:41 /work |
| `dsv41flash-sal-v2_baseline_I15_s5_p20` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:18 /work |
| `dsv41flash-sal-v2_baseline_I15_s9_p15` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:42 /work |
| `dsv41flash-sal-v2_baseline_I15_s5_p15` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:17 /work |
| `dsv41flash-sal-v2_baseline_I15_s3_p10` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:11 /work |
| `dsv41flash-sal-v2_baseline_I15_s0_p12` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:13 /work |
| `dsv41flash-sal-v2_baseline_I15_s0_p10` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:13 /work |
| `dsv41flash-sal-v2_baseline_I15_s5_p22` | (no work_mtime on the event) |
| `dsv41flash-sal-v2_baseline_I15_s9_p13` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:41 /work |
| `dsv41flash-sal-v2_baseline_I15_s0_p09` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:12 /work |
| `dsv41flash-sal-v2_baseline_I4_s7_p02` | drwxr-xr-x 1 dc dc 4096 Sep 13 19:41 /work |
| `dsv41flash-sal-v2_baseline_I15_s0_p06` | drwxr-xr-x 1 dc dc 4096 Sep 13 17:12 /work |
| `dsv41flash-sal-v2_baseline_I4_s9_p04` | drwxr-xr-x 1 dc dc 4096 Sep 13 19:35 /work |
| `dsv41flash-sal-v2_baseline_I8_s5_p15` | drwxr-xr-x 1 dc dc 4096 Sep 13 18:09 /work |
| `dsv41flash-sal-v2_baseline_I8_s10_p15` | drwxr-xr-x 1 dc dc 4096 Sep 13 18:39 /work |
| `dsv41flash-sal-v2_baseline_I4_s6_p07` | drwxr-xr-x 1 dc dc 4096 Sep 13 19:35 /work |
| `dsv41flash-sal-v2_baseline_I15_s2_p02` | (no work_mtime on the event) |
| `dsv41flash-sal-v2_baseline_I4_s6_p06` | drwxr-xr-x 1 dc dc 4096 Sep 13 19:35 /work |
| `dsv41flash-sal-v2_baseline_I4_s10_p09` | (no work_mtime on the event) |
| `dsv41flash-sal-v2_baseline_I2_s0_p02` | (no work_mtime on the event) |

## Failure classes and the rules that now cover them (hand-written)

Everything below this heading was written by hand on 11 Oct after the runs; a rerun of
`--check` regenerates only the part above.

### The first run, under the guide's rule

The first `--check 20` (05:29 UTC, same draw, same image), with `normalise` touching
only `ls -l` dates and pytest timings, gave **18 of 20, FAIL**. Both failures were
unreachable events, and neither was a difference in anything the agent did or wrote:

| event_id | call | command (start) | class | cause | rule that now covers it |
|---|---|---|---|---|---|
| `..._I15_s9_p15` | 7 | `ls -la ..; find / -maxdepth 5 ...; ls -la /` | `/proc` link count | `ls -la /` shows `dr-xr-xr-x 434 root root 0 ... proc` in the recording and `1445` in the replay. procfs reports the number of processes in the whole Docker VM as `/proc`'s link count (other projects' containers were running), and the 4-digit count widens the link column of all 23 lines of the listing | `normalise`, `LS_NLINK_RE` (PREREG departure): the `proc` link count becomes a token and the link-column padding one space |
| `..._I2_s0_p02` | 0 | `cat spec.md; echo ---; cat data/postprocess.md; ...` | stdout/stderr interleaving | The same 70 lines; only `cat: data/postprocess.md: No such file or directory` (stderr) moved from line 40 to line 2. `docker exec` carries stdout and stderr on separate streams and `ToolExecutor` merges them on arrival, so their order is a race; the recording is one draw of it. In 10 more replays of this event 7 matched and 3 did not | `content_equal` (`Replay.ok`); the strict count is reported beside it |

Neither class was date-only, size-only, a truncation boundary or a host-side `touch`
problem: every `touch -d "<work_mtime>" /work` succeeded and `ls -ld /work` shows the
recorded date. Under the new rules the first run's data reads content 20 of 20,
strict 19 of 20. The rerun above, after both rules, gave content 20 and strict 20 (the
race did not fire).

### Eligibility sweep: every eligible event (`data/replay_sweep.json`)

`uv run python -m rab.prefix --sweep` replays all 183 events that `rab.select` keeps
before the draw (its own `scan` and `eligible_ids`), one fresh container each, 682 s.
An event is eligible for the draw only if `ok` (every prefix call `content_equal`).

| kind | events | ok (content) | strict_ok | excluded |
|---|---|---|---|---|
| solvable | 74 | 74 | 74 | 0 |
| unreachable | 109 | 96 | 93 | 13 |

Content-only passes (the stdout/stderr race): `I4_s10_p02`, `I4_s9_p11`, `I8_s4_p13`.
6 of the 13 excluded events are in the current draw (`I15_s0_p01`, `I15_s7_p02`,
`I15_s7_p03`, `I2_s1_p06`, `I8_s6_p09`, `I8_s9_p08`): 114 of the 120 drawn are ok. The
estimate of 4 exclusions in the first version of this file (116 of 120) was low: its
first-pass classifier labelled `I15_s0_p01` and `I2_s1_p06` by their `/proc` difference
and hid the `/tmp` difference on the same listing.

All 13 excluded events are unreachable agents searching outside their work dir for the
missing `data/postprocess.md`; what differs is container state that KM's batch
container accumulated before the item started (an event can show several):

| cause | excluded events | what differs |
|---|---|---|
| `/tmp` link count | 12 of 13 (all but `I15_s1_p03`); the only difference for `I15_s0_p01`, `I2_s11_p10`, `I2_s1_p06`, `I8_s5_p01`, `I8_s6_p02`, `I8_s9_p08` | `drwxrwxrwt 1 root root ... tmp` recorded, `2` replayed. KM ends every item with `hidden_grade` (pytest), and pytest creates and deletes a temp file in `/tmp`; overlayfs then reports the copied-up `/tmp` with link count 1 for the rest of the batch. Checked in a fresh container: `/tmp` shows 2 after `prepare_item` and 1 after one `run_visible_tests` |
| the agent's own files from earlier items | `I15_s7_p02`, `I15_s7_p03` (`/tmp/calc.py`), `I8_s6_p09` (`/tmp/chk.py`) | listed and `cat`-ed in the recording, absent in a fresh container (`clear_work_root` empties only `/work`) |
| PID numbers | `I15_s1_p03`, `I15_s7_p02` | `find / -maxdepth 2` lists `/proc/373`... recorded, `/proc/80`... replayed: earlier items advanced the PID counter |
| `env` output | `I15_s6_p05`, `I4_s5_p09`, `I8_s4_p12` | `HOSTNAME` (the container id, new in every container) and `PYTHON_VERSION=3.12.14` / `PYTHON_SHA256` (today's image has 3.12.15, see `results/d0.md`) |
| `stat` of the item dir | `I8_s4_p12` | device, inode, access, change and birth times |

### Proposed, not implemented: emulate the `/tmp` copy-up

For an event at position 1 or later, create and delete one temp file in `/tmp` as the
sandbox user before the prefix (what the previous item's `hidden_grade` did). Measured
on the 13 excluded events with `python -c "import tempfile;
tempfile.TemporaryFile().close()"` run in the container first: 6 become `ok`
(`I15_s0_p01`, `I2_s11_p10`, `I2_s1_p06`, `I8_s5_p01`, `I8_s6_p02`, `I8_s9_p08`), 7 stay
excluded (`I15_s1_p03`, `I15_s6_p05`, `I15_s7_p02`, `I15_s7_p03`, `I4_s5_p09`,
`I8_s4_p12`, `I8_s6_p09`), and the current draw would have 117 of 120 ok. It also
makes the continuation's container match the batch container for every event at
position 1 or later, which today's replays do not (a post-block `ls -la /` shows `/tmp`
at 2). The full sweep would have to be rerun with it before adopting it.

### Superseded: the first all-120 diagnostic

Before the new rules, every drawn event was replayed once (`--check 120`, 286 s): 111
of 120 clean. Failing events: `/proc` link count `I15_s9_p15`, `I8_s9_p08`, `I15_s0_p01`,
`I2_s1_p06`, `I15_s7_p02`; batch history `I15_s7_p02`, `I15_s7_p03`, `I8_s6_p09`,
`I8_s9_p08`; stdout/stderr race `I4_s9_p11`, `I8_s4_p13`. Every one was unreachable.
The sweep above replaces it.

Side finding for KM, not for this repo: the batch-history class is a memory channel the
env-v2 cleanup does not close. An agent can leave a file in `/tmp` in one item and read
it in a later item of the same batch (`I15_s7`: `/tmp/calc.py`).
