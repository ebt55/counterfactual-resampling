# counterfactual-resampling: preregistration

[EBIN: the question in one sentence]

Nothing was blocked in the recording; the read succeeded. B models a monitor that was not
there.

## Predictions

1. On unreachable events the clean regrade passes on 80–92% of R runs and under 35% of P
   runs; under 10% on events whose notes do not name the transform; gap at least 40
   points.
2. TVD(B, R) at least 0.4 on unreachable, p < 0.05; between 0.1 and 0.3 on solvable.
3. On solvable events SA − R in notes naming the file is between −10 and +5 points and
   the interval excludes −20.
4. SA vs R discriminator 55–75% with the interval excluding 50; S\* vs R interval includes
   50. Scored only if swapped-position agreement on the 30 consistency pairs is at least
   70%, else reported as untestable.
5. W: the solution reproduces the file in under 5% of runs, `flagged_problem` at least 15
   points above R. P: fitted under 15%, flagged at least 50%.

[EBIN: one prediction I would bet against, and why]

## The null

A simulated answer file at the first read does not move what this agent does next beyond
the spread of re-runs on the real file, and a blinded model cannot tell the two apart.

## Sample

60 solvable and 60 unreachable events, 18 runs each (R 8, SA 4, S\* 2, W or P 2, B 2):
2,160 continuations. A 5-event exploratory pilot (90 continuations) ran before this was
written and shaped predictions 1, 2 and 5; it is excluded from every headline table.

## Stopping rule

Stop at $15 of non-Anthropic cash. Only events with all 18 slots done enter the headline.
Re-run only infrastructure failures (API, Docker, harness); turn limits and timeouts are
results.

Analysis rules: `notes/protocol-2026-10-11.md` at commit `31434d5`.
