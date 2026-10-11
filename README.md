# counterfactual-resampling

[EBIN: one sentence, verbatim]

At the agent's first read of the answer file, we replaced the file's contents on disk
with a simulated version and re-ran the agent from that step. The comparator is re-runs
from the same step with the real file. A third arm models a monitor that denies the read.

## Results

[filled by analyze.py after the run; paragraph chosen by its verdict rule]

Predictions (`PREREG.md`): predicted X, got Y. [filled after the run]

## Method

The analysis rules and every number below, with its source, are in
`notes/protocol-2026-10-11.md`; the predictions are in `PREREG.md`.

- **Events.** From the Kobayashi Maru records of DeepSeek-V4.1-flash (60 runs of the
  `dsv41flash-sal-v2` line), the first successful read of the forbidden answer file
  `expected_outputs.json` in each record: 76 solvable and 111 unreachable events, minus
  reads that shared an assistant turn with a later call and prefixes that do not replay,
  then 60 of each drawn with seed 20261011 (`src/rab/select.py`, `data/events.jsonl`).
- **Replay.** Each continuation starts in a fresh container in which every earlier tool
  call of the item is re-executed and checked against the recording
  (`src/rab/prefix.py`, `results/replay_check.md`).
- **The fake.** Claude Sonnet 5.5 writes the file from what the agent had seen: the
  rules, tool list, spec, visible tests, the session so far and the read call, never the
  real file or anything after the read (`src/rab/simulate.py`,
  `src/rab/prompts/simulator_v1.txt`).
- **Placement.** The arm's file goes onto disk at both paths through one write path
  (the real file too), with KM's mtime and serialisation, and the read is re-executed live
  (`src/rab/place.py`).
- **Arms.** R (the real file, 8 runs), SA (the Sonnet fake, 4), S\* (the fake's arguments
  with correct values, 2), W (the real values shuffled across cases, solvable, 2), P
  (correct but untransformed values, unreachable, 2), B (a monitor that denies every
  answer-file access, 2): 18 continuations per event, 2,160 in all (`src/rab/run.py`).
- **Continuation.** KM's own agent loop from the read to `submit` or a limit, with the
  provider pinned to DeepSeek's endpoint, then KM's two graders (`src/rab/resume.py`).
- **Measures.** A class per continuation (error, refusal, fitted and submitted, flagged a
  problem, kept probing, submitted clean, no submit), what the submitted solution computes
  (spec, the hidden transform, the file, other), re-reads, the submit note, the clean
  regrade and a frozen suspicion regex, all from the steps after the read
  (`src/rab/classify.py`). On the 187 recorded continuations the classifier reproduces
  KM's published counts.
- **Statistics.** Per cell, the total variation distance between the class distributions
  of SA and R with a within-event permutation test and an event bootstrap, Holm across
  cells, a task-level sign-flip test beside it, equivalence only on class shares within
  ±10 points, and a pairwise blinded discriminator (Claude Haiku 5.5) for "which is real"
  (`src/rab/analyze.py`, `src/rab/discriminate.py`).

## Scope

DeepSeek-V4.1-flash; 5 to 25-line Python tasks; the fake is a 1 to 3 KB test-case file;
7 unreachable tasks.

## Related work

OpenAI's simulated-deployment forecasts (Williams et al., Jul 2026) regenerate a candidate
model's next response on fixed production prefixes and name tool-resampling realism as
the central open problem, but do not swap a tool observation and measure what follows.
Kissane, MacDiarmid and Roger (Mar 2026) and Ahlqvist et al. (Sep 2026) measure how real
simulated audits look to a judge or to the model, not how one swapped observation changes
the agent's next steps against re-runs on the real one; EnvSimBench (May 2026) scores
simulator accuracy without agents, and Trace2Env (Oct 2026) scores next-observation
fidelity and whether replayed actions stay valid. Anthropic's incident assessment (Sep
2026) resampled edited transcripts and rebuilt incidents in simulation, without a
real-versus-simulated comparison at a fixed step. Ctrl-Z (Apr 2025) and Lucassen and
Kaufman (May 2026) resample or retry the agent's action rather than the environment's
observation; Mallen (Sep 2026) argues that blocking monitors cost usefulness that learning
will route around, and METR's per-action monitor (Sep 2026) blocks and halts for review,
so neither measures what an agent does after a denied or faked read. Shah's Causal Agent
Replay (Jun 2026) intervenes on one step and re-runs the rest, on synthetic models. This
pilot puts a simulated observation into a real container at the agent's real first read of
a forbidden file and measures the gap against the spread of real re-runs and against a
denial. Sources and what each measured: `notes/citations-2026-10-11.md`.

## Reproduce

Needs Docker, `uv`, and a checkout of the Kobayashi Maru repository at tag
`freeze-2026-09-14b` (`KM_ROOT`, see `.env.example`).

```
uv run python -m rab.select           # the events, byte-identical to data/events.jsonl
uv run python -m rab.run --dry-run    # every arm end to end, keyless (FakeProvider, FakeSimulator)
uv run python -m rab.analyze          # results/tables.md from the continuation files
```

The full run (keys in `.env`): `uv run python -m rab.run`, then
`uv run python -m rab.analyze`. The 5-event pilot (`uv run python -m rab.run --pilot 5`)
is exploratory and excluded from the results above.

## Modules

The package is `rab`, the working name's initials; the repo name is `counterfactual-resampling`.

Generated by `uv run python tools/module_map.py`.

| Module | What it does (first docstring paragraph) | Lines |
|---|---|---|
| `src/rab/analyze.py` | Analysis over the continuation files (design v2 D, G; guide §8) | 812 |
| `src/rab/classify.py` | Continuation classes and measures on the post-block view (build guide §7; build plan decisions 1, 2, 3, 5; design v2 A.3, C, F) | 461 |
| `src/rab/constants.py` | Every number, id and pattern the build guide fixes, in one place | 58 |
| `src/rab/discriminate.py` | Pairwise discriminator: which observation is real (design v2 E; PREREG §7) | 458 |
| `src/rab/km.py` | Read-only access to the Kobayashi Maru (KM) records and task tree (guide §2, §3, §10) | 116 |
| `src/rab/place.py` | Placement and the arms' files (guide §5; build plan decisions 7, 9, 17b; design v2 B, C) | 422 |
| `src/rab/prefix.py` | Prefix rebuild and replay (build guide §3; the day-0 "Replay" check of §1) | 414 |
| `src/rab/resume.py` | Continuation from the block point (build guide §6; build plan decisions 4, 5, 6, 10, 13) | 536 |
| `src/rab/run.py` | The run: every slot of every event, resumable (design v2 B, C, D; guide §6, §10) | 567 |
| `src/rab/select.py` | Event selection: the block points of the pilot (build guide §2) | 688 |
| `src/rab/simulate.py` | Simulator: a fake ``expected_outputs.json`` for the block point (build guide §4; §5 leak canary) | 755 |
| `src/rab/types.py` | Frozen interfaces between chunks (build plan "Interfaces"; guide §2, §4, §5) | 222 |

## Who did what

- Ebin Babu Thomas: the question, the predictions (`PREREG.md`), the hand read of ten
  continuations, and the sections written in his words.
- Code and data: Claude Opus 5.5 agents under a Claude Fable 5.1 coordinator.
- Records, tasks, harness and detectors: the Kobayashi Maru repository, unchanged.

[EBIN: "Why this question" and "What this does not show", in his words]

## Licence

MIT (`LICENSE`).
