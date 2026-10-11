# counterfactual-resampling: preregistration

[EBIN: the question in one sentence]

## Predictions

Proposals from the analysis protocol; edit, keep or strike each.

- Positive control: on unreachable items the clean-regrade pass rate is at least 30
  points lower under P (a plausible spec-correct file without the transform) than under
  R (the real file). [EBIN: keep / change / strike]
- Denying the read (B) moves the class distribution away from R's beyond the R-vs-R
  permutation null, in both cells. [EBIN: keep / change / strike]
- On solvable items, the share of submit notes that name the answer file is within ±10
  points of R's under SA. [EBIN: keep / change / strike]
- A blinded model picks the real observation in 55% to 80% of SA-vs-R pairs, and near 50%
  for S\* vs R (same surface, correct values). [EBIN: keep / change / strike]
- Uncalled: whether W (real values shuffled, solvable) or P changes what the agent
  submits. [EBIN: keep / change / strike]

[EBIN: one prediction I would bet against, and why]

## The null

A simulated answer file at the first read does not move what this agent does next beyond
the spread of re-runs on the real file, and a blinded model cannot tell the two apart.

## Sample

60 solvable and 60 unreachable events, 18 runs each (R 8, SA 4, S\* 2, W or P 2, B 2):
2,160 continuations. The 5-event pilot is exploratory and excluded.

## Stopping rule

Stop at $15 of non-Anthropic cash. Re-run only infrastructure failures (API, Docker,
harness); turn limits and timeouts are results.

Analysis rules: `notes/protocol-2026-10-11.md` at commit `[hash]`.
