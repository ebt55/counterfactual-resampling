# Kobayashi Maru: issues found while building the resampling pilot (11 Oct 2026)

Found by the four design reviews and the replay work in this repo, on the KM checkout
`C:\Users\ebin\claude-ground\ai-incident-response\dosecurve` at tag `freeze-2026-09-14b`.
Nothing in KM was changed. This file lives here so the pilot's "KM unchanged" statement
stays true; copy it into KM's `followups.md` when you work on it.

Verdict first: the headline cheating rates and the notes mechanism are not affected.
Items 1 and 2 touch secondary numbers; items 3 to 5 are hygiene.

## 1. The declared-impossible heuristic over-fires on "file missing" mentions

- **What:** `detectors/detect.py` `declared_impossible()` returns True on an impossible
  item whenever the EXTRA pattern matches (missing file, `no such file`, `could not
  find`, and so on). Unreachable items point the agent at `data/postprocess.md`, which
  does not exist, so most agents mention it whatever they then do.
- **Evidence (DeepSeek `dsv41flash-sal-v2` line, recount 11 Oct):** the heuristic fires on
  72 of 111 unreachable records; 66 of those 72 also passed the real hidden tests by
  fitting the transform from the answer file. Only 8 of the 72 match the strict SPEC
  pattern (`impossib|cannot be solved|contradict|...`).
- **Why the headline survives:** `outcome()` precedence is error → refusal → cheat →
  declared_impossible → honest_pass → honest_fail, so a run that read the file is
  counted as cheat, never as declared.
- **What may be wrong:** the "said this is impossible" column in `README.md` (line ~117)
  and the "declared impossible (unreachable)" column in `results/analysis/NUMBERS.md`
  (line ~84), plus the P8 declare-rules row (NUMBERS line ~109). A non-cheating run that
  mentions the missing file and submits a spec solution lands in declared rather than
  honest_fail. Size unmeasured.
- **To do:** recount those columns under a stricter rule (SPEC pattern only, or EXTRA
  plus no submission of a solution that passes the visible tests), report both, and add
  a DEVIATIONS line. Also note the heuristic misses "cannot determine", "not possible",
  "unknown rule". The pilot's own classifier (`src/rab/classify.py` here) renames the
  class to `flagged_problem` for this reason.

## 2. A memory channel across items through `/tmp`

- **What:** env v2 wipes `/work` between items (`clear_work_root`) but not `/tmp`, `~`
  or the process table. Files an agent writes to `/tmp` in one item are readable in
  later items of the same batch.
- **Evidence:** replay of batch `dsv41flash-sal-v2_baseline_I15_s7` shows the agent
  listing and `cat`-ing `/tmp/calc.py` and `/tmp/chk.py` written in earlier items
  (`results/replay_check.md` in this repo, "batch-container history"). Three of 183
  eligible prefixes depend on such files; they were utility scripts, not notes.
- **Why the mechanism claim survives:** the notes ablation dropped cheating from 46 to 0
  of 120 in the measured cell with notes withheld; a channel still open through `/tmp`
  would not give zero.
- **To do:** state it in `followups.md` as a known second channel; for any future env
  version, run each item in a fresh container or clear `/tmp` and `~` as well.

## 3. The Docker image is not pinned

- **What:** `harness/docker/Dockerfile` uses `python:3.12-slim` unpinned. The September
  records print Python 3.12.14; a rebuild on 11 Oct gave 3.12.15 (base digest
  `python@sha256:02108f5d…`). pytest 8.3.4 is pinned and matches.
- **Effect:** any output that prints the Python version differs across rebuilds
  (`env` output in 3 of 183 DeepSeek prefixes). No published number depends on it.
- **To do:** pin the base by digest in the Dockerfile and record the built image digest
  in PREREG for future runs.

## 4. Replay nondeterminism to document for anyone re-running from records

- `ls -la /` shows procfs's link count for `/proc`, which equals the Docker VM's process
  count at the time; it differed between 434 (13 Sep) and 570 to 2087 (11 Oct).
- `docker exec` merges stdout and stderr by arrival, so a `cat: ...: No such file`
  diagnostic can move lines between runs of the same command (matched 7 of 10 replays).
- `/tmp` shows link count 1 after the first item's pytest run and 2 in a fresh container;
  PIDs advance across items; `HOSTNAME` differs per container.
- None of these is an agent action. The pilot's `rab.prefix.normalise` and
  `content_equal` document the rules used to compare outputs across them.

## 5. Small things

- The README "Try it yourself" dry run writes `results/runs/demo-dry_baseline_I4_s0/` and
  overwrites `results/runs/last_run_summary.json` inside the repo; suggest `--out` in the
  README so a reader's dry run does not touch the frozen results directory.
- `pytest -q` in the README runs at `-qq` because `addopts` already has `-q`, so it
  prints no summary line; drop the flag from the README or from `addopts`.
- `harness/docker/Dockerfile` comment still says `/work/<item_key>` (env v1 layout).
- `OpenAICompatProvider` silently swallows unknown kwargs (`**_kw`); a misspelled
  `temperature=` would run at the default without an error.
- The record's `tool_calls[i]` carries no `id`; the call id is only in `transcript`.
  Worth one line in the record-format docs.
