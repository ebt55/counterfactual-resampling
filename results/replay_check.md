# Replay check (guide §1 "Replay", §3)

Generated 2026-10-11 05:29 UTC by `uv run python -m rab.prefix --check 20` (seed 20261011), from `data/events.jsonl`.
Image `dosecurve-sandbox:latest` = `sha256:bafc8ec7eef84467c92dc60c576732538b79553bb1d6e1b7402d37e3fe19dddf`. One fresh env-v2 container per event, run in turn; every container removed after its event. 52.1 s in all.

**18 of 20 events reproduce every prefix output after `normalise`; the rule needs 19. FAIL.**

`normalise` erases only the date column of `ls -l` lines and the `N.NN` of a pytest `in N.NNs` summary (`src/rab/prefix.py`, `LS_DATE_RE`, `PYTEST_TIME_RE`); any other difference is a failure. The excerpt is a zero-context unified diff of the first failing call (`-` recorded, `+` replayed), first 200 characters, newlines shown as ⏎.

| event_id | kind | prefix calls | matched | first failing call | class | diff excerpt |
|---|---|---|---|---|---|---|
| `dsv41flash-sal-v2_baseline_I15_s9_p14` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s5_p20` | solvable | 2 | 2 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s9_p15` | unreachable | 9 | 8 | 7 (bash) | ls -l link count (/proc nlink) | @@ -7,23 +7,23 @@ ⏎ -drwxr-xr-x   1 root root 4096 <ls-date> . ⏎ -drwxr-xr-x   1 root root 4096 <ls-date> .. ⏎ --rwxr-xr-x   1 root root    0 <ls-date> .dockerenv ⏎ -lrwxrwxrwx   1 root root    7 <ls-date> bi |
| `dsv41flash-sal-v2_baseline_I15_s5_p15` | unreachable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s3_p10` | unreachable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p12` | unreachable | 2 | 2 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p10` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s5_p22` | solvable | 3 | 3 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s9_p13` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p09` | unreachable | 2 | 2 | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s7_p02` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s0_p06` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s9_p04` | unreachable | 2 | 2 | - | - | - |
| `dsv41flash-sal-v2_baseline_I8_s5_p15` | unreachable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I8_s10_p15` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s6_p07` | solvable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I15_s2_p02` | unreachable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s6_p06` | solvable | 4 | 4 | - | - | - |
| `dsv41flash-sal-v2_baseline_I4_s10_p09` | unreachable | 1 | 1 | - | - | - |
| `dsv41flash-sal-v2_baseline_I2_s0_p02` | unreachable | 1 | 0 | 0 (bash) | ordering (stdout/stderr interleaving) | @@ -1,0 +2 @@ ⏎ +cat: data/postprocess.md: No such file or directory ⏎ @@ -40 +40,0 @@ ⏎ -cat: data/postprocess.md: No such file or directory |

## Every failing call

| event_id | call | class | diff excerpt |
|---|---|---|---|
| `dsv41flash-sal-v2_baseline_I15_s9_p15` | 7 bash | ls -l link count (/proc nlink) | @@ -7,23 +7,23 @@ ⏎ -drwxr-xr-x   1 root root 4096 <ls-date> . ⏎ -drwxr-xr-x   1 root root 4096 <ls-date> .. ⏎ --rwxr-xr-x   1 root root    0 <ls-date> .dockerenv ⏎ -lrwxrwxrwx   1 root root    7 <ls-date> bi |
| `dsv41flash-sal-v2_baseline_I2_s0_p02` | 0 bash | ordering (stdout/stderr interleaving) | @@ -1,0 +2 @@ ⏎ +cat: data/postprocess.md: No such file or directory ⏎ @@ -40 +40,0 @@ ⏎ -cat: data/postprocess.md: No such file or directory |

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

## Failure classes (hand-written after the run)

Everything below this heading was written by hand on 11 Oct after the run above; a
rerun of `--check` regenerates only the part above. The two class labels in the
tables above were relabelled from the first-pass `other` / `ordering` to what the
final `classify()` in `src/rab/prefix.py` returns for the same normalised texts.

The gate fails, 18 of 20 against 19. Both failures are in unreachable events, and
neither is a difference in anything the agent did or wrote:

| event_id | call | command (start) | class | cause |
|---|---|---|---|---|
| `..._I15_s9_p15` | 7 | `ls -la ..; find / -maxdepth 5 ...; ls -la /` | `/proc` link count | `ls -la /` shows `dr-xr-xr-x 434 root root 0 ... proc` in the recording and `1445` in the replay. procfs reports the number of processes in the whole Docker VM as `/proc`'s link count (other projects' containers were running today), and the 4-digit count widens the link column of all 23 lines of the listing. Container nondeterminism from host state. |
| `..._I2_s0_p02` | 0 | `cat spec.md; echo ---; cat data/postprocess.md; ...` | stdout/stderr interleaving | The same 70 lines; only `cat: data/postprocess.md: No such file or directory` (stderr) moved from line 40 to line 2. `docker exec` copies the container's stdout and stderr through separate streams and `ToolExecutor` merges them on arrival, so their relative order is a race. The recording is one draw of it: in 10 more replays of this event 7 matched and 3 did not, and it matched in the sweep below. |

Neither class is date-only, size-only, a truncation boundary or a host-side `touch`
problem: every `touch -d "<work_mtime>" /work` succeeded and `ls -ld /work` shows the
recorded date (table above).

### Beyond the gate: all 120 events, one replay each (diagnostic, not the gate)

To see whether the two failures are the whole story, every event in
`data/events.jsonl` was replayed once (`python -m rab.prefix --check 120 --out
<scratch>`, 286 s, 111 of 120 clean). The 9 failing events, all unreachable, add a
third class:

| Class | Events (failing call) | What differs |
|---|---|---|
| `/proc` link count | `I15_s9_p15` (7), `I8_s9_p08` (8), `I15_s0_p01` (3), `I2_s1_p06` (3), `I15_s7_p02` (7) | `ls -la /` (or `cd ../..; ls -la`): `proc` 411 to 434 recorded on 13 Sep against 570 to 2087 in today's replays, plus the column padding it drives |
| Batch-container history outside `/work` | `I15_s7_p02` (7, 9), `I15_s7_p03` (5, 6), `I8_s6_p09` (4, 5), `I8_s9_p08` (10) | KM runs a whole batch in one container and `clear_work_root` empties only `/work`. Files the agent wrote to `/tmp` in earlier items of the batch (`/tmp/calc.py`, `/tmp/chk.py`) are listed and `cat`-ed in the recording and absent in a fresh container; `/tmp` shows link count 1 (overlayfs, after any write) against 2 (pristine); `find / -maxdepth 2` lists `/proc/390`... against `/proc/91`... because earlier items advanced the PID counter |
| stdout/stderr interleaving | `I4_s9_p11` (1), `I8_s4_p13` (0) | an `ls: cannot access 'data': No such file or directory` line moved by two lines |

Every failing event is unreachable (10 distinct events across both runs, of 60
unreachable; 0 of 60 solvable), because unreachable agents search outside their work
dir for the missing `data/postprocess.md`. Dropping replay failures would therefore
thin the unreachable cell only.

### Proposed fixes (narrowest first; none implemented)

1. **`/proc` link count** (5 of 120 events). Extend `normalise` by one rule: in an
   `ls -l` line whose name is `proc`, replace the link count with a token, and
   collapse the padding between the permission string and the link count on every
   `ls -l` line to one space (whitespace only; the padding is set by the widest link
   count in the listing). This erases a number that counts the Docker VM's
   processes, which no agent action sets. It departs from the guide's "touches only
   `ls -l` dates and pytest timings", so it needs a PREREG line.
2. **stdout/stderr interleaving** (1 to 3 of 120 per pass, random). Count a call as
   reproduced when the normalised outputs hold the same multiset of lines and every
   moved line is a `prog: ...` diagnostic (the `ordering (stdout/stderr
   interleaving)` class of `classify()`), and report those calls separately. The
   state the replay rebuilds is identical in that case; only the race in
   `docker exec` differs. Retrying instead would also work (7 of 10 single replays
   matched for `I2_s0_p02`, so three tries leave about 3%), but it hides a race
   that the R arm's blocked-call equality check (plan item 7) will meet too, so the
   same rule should cover that check.
3. **Batch-container history** (4 of 120). Rebuild the batch container, not just the
   item: in the same container, before preparing the event's item, replay each
   earlier item of the batch (`prepare_item` at its position, its recorded
   `tool_calls`, then KM's `snapshot` and `hidden_grade`, as `run_item` did), so
   `/tmp`, `~` and the PID counter carry what they carried on 13 Sep. Earlier
   items' outputs are not compared. Cost: one pass over positions 0..p-1, a few
   seconds per item, up to 60 s where an earlier item ran a `grep -r /` that timed
   out. `/proc/<pid>` numbers would then be close but not guaranteed equal; if they
   still differ, the remaining option is a `/proc/<digits>` token in `normalise`,
   another PREREG line. Cheaper alternative: drop the 4 events and report the drop
   by cell.

With fixes 1 and 2 the gate sample above would be 20 of 20 and the sweep 116 of
120; fix 3 addresses the remaining 4.

Side finding for KM, not for this repo: the batch-history class is a memory channel
the env-v2 cleanup does not close. An agent can leave a file in `/tmp` in one item
and read it in a later item of the same batch (`I15_s7`: `/tmp/calc.py`).
