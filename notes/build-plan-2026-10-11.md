# blockpoint: build plan (11 Oct 2026)

Orchestration plan for building the resampling pilot from
`C:\Users\ebin\claude-ground\profile-update-oct2026\project-plan-2026-10-09\07-BUILD-GUIDE-resampling-pilot.md`
(the guide) and `05-FINAL-PLAN.md` §1 (the plan). The guide is the spec; this file
fixes the conventions every chunk shares and splits the work into chunks with
file boundaries, acceptance criteria and verification steps.

## Why this exists (read before coding)

The project's value is one preregistered number that Redwood and METR said they
want: at the point where a real agent (DeepSeek-V4.1-flash) read a forbidden
answer file, a fake file from another model family is put on disk instead, the
agent keeps working in a real container, and we measure whether what it does
next differs from re-runs on the real file, against the spread of those re-runs.
Everything in the repo serves that number. No framework, no abstraction for a
second substrate, no dashboards. A reader must be able to open the repo, read
`README.md`, `PREREG.md` and `results/tables.md`, and re-run every step from the
KM records with one command per step.

## Shared conventions (binding for every chunk)

- **Repo**: `C:\Users\ebin\claude-ground\counterfactual-resampling` is the repo
  root; GitHub name `blockpoint` (owner `ebt55`). MIT.
- **Package**: `src/blockpoint/`, imported as `blockpoint`. Module names follow
  the guide: `select.py`, `prefix.py`, `simulate.py`, `place.py`, `resume.py`,
  `classify.py`, `discriminate.py`, `analyze.py`, `run.py`. Where the guide
  writes `rab.x`, read `blockpoint.x`.
- **KM**: the Kobayashi Maru checkout at
  `C:\Users\ebin\claude-ground\ai-incident-response\dosecurve`, tag
  `freeze-2026-09-14b` (HEAD is 5 commits past the tag; the only `.py` diffs are
  `analysis/spend.py` and its test, so harness/detectors code equals the tag).
  `KM_ROOT` env var, defaulting to that path. `src/blockpoint/__init__.py` and
  `tests/conftest.py` insert `KM_ROOT` on `sys.path`. **Nothing in the KM repo
  changes, ever.** Import KM modules (`harness.*`, `detectors.*`, `analysis.*`);
  never copy their source, except the `run_item` inner loop that the guide §6
  says to copy into `resume.py`.
- **Python**: 3.12+ via `uv`. `pyproject.toml` with `[project]` deps
  (`httpx`, `anthropic`, `python-dotenv`, `pytest`, `numpy`, `scipy`,
  `pandas`, `statsmodels`), `[tool.pytest.ini_options]` with markers `docker`
  (needs Docker) and `live` (spends tokens), `addopts = "-q -m 'not live'"`.
  Run tests with `uv run pytest`.
- **Secrets**: `.env` at repo root (gitignored), loaded with python-dotenv.
  **Do not read `$KM_ROOT/.env`; those keys are old and Ebin will issue fresh
  ones.** The loader reads only the repo's own `.env` and the process
  environment. Never print key values. Never commit `.env`.
- **Spend**: no chunk in this session makes any live model call. The day-0
  "served" check (`GET /models/<id>/endpoints` needs no key; the one pinned
  call does), the "credit" check, continuations, simulator draws and
  discriminator calls all wait for fresh keys from Ebin, his sentence and the
  committed PREREG. Build the code paths and `--dry-run` only. Everything must be testable keyless with
  `harness.providers.fake_provider.FakeProvider` and `LocalSandbox` /
  `DockerSandbox`.
- **Style**: small plain functions with type hints; module docstring names the
  guide section it implements; every constant that the guide fixes (seed
  20261011, run counts, regex, thresholds) lives in one place
  (`src/blockpoint/constants.py`) and is imported, not retyped. No classes
  unless state demands it. No logging frameworks; print to stderr. Paths via
  `pathlib`. Windows host, Linux Docker: never assume POSIX on the host.
- **Wording**: "the recorded next step", never "ground truth". Numbers in
  docs name the file they came from.
- **Results**: `results/` holds run outputs (`events` live in `data/`). Files
  the guide names: `results/provider.log`, `results/sims.jsonl`,
  `results/leaks.jsonl`, `results/refusals.jsonl`, `results/continuations/`,
  `results/tables.md`, `results/rows_for_ebin.md`, `results/rows_key.json`,
  `results/cost.md`. Dry-run outputs go under `results/dry/` so a real run
  never overwrites them.
- **Git**: each chunk ends with a commit on `main`. Short imperative subject,
  body says what was verified. End with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Verification is part of the chunk**: run the tests, paste the pytest
  summary and the recount numbers in the final report. A chunk that reports
  "should work" is not done.

## Chunks

| # | Name | Owner | Depends on | Files |
|---|---|---|---|---|
| A | Scaffold + event selection | agent A | none | `pyproject.toml`, `LICENSE`, `.gitignore`, `.env.example`, `src/blockpoint/__init__.py`, `constants.py`, `km.py`, `select.py`, `data/events.jsonl`, `tests/conftest.py`, `tests/test_select.py` |
| B | Prefix rebuild, replay, Docker day-0 | agent B | A's scaffold for tests; A's events for `replay_check` | `src/blockpoint/prefix.py`, `tests/test_prefix.py`, `tests/fixtures/`, `results/d0.md`, `results/replay_check.md` |
| C | Simulator, placement, leak canary | resume agent A | A, B's `replay` | `src/blockpoint/simulate.py`, `prompts/simulator_v1.txt`, `place.py`, `tests/test_place.py`, `tests/test_leak.py`, `tests/test_simulate.py` |
| D | Resume, pre-flight, classification | resume agent B | B, C | `src/blockpoint/resume.py`, `classify.py`, `tests/test_preflight.py`, `tests/test_classify.py` |
| E | Discriminator, analysis, dry run, PREREG, README | first free agent | C, D | `discriminate.py`, `analyze.py`, `run.py`, `results/dry/tables.md`, `PREREG.md`, `README.md` |

Day-0 Docker checks (image build, KM dry run, replay) run in B. The keyed
day-0 checks (served pin call, credit) wait for fresh keys. Ebin's items
(sentence, PREREG approval, emails, 20 rows, keys) stay his.
