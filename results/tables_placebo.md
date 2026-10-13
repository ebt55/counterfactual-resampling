# Placebo (false-positive check on R alone)

In each seeded random split every event's R runs are halved, one half relabelled, and the primary test is run on the halves; the rejection rate should sit near the level.

| cell | events | splits | permutations per split | rejected at 0.05 | rejected at 0.025 |
|---|---|---|---|---|---|
| solvable | 60 | 200 | 1,000 | 9.5 | 5.5 |
| unreachable | 60 | 200 | 1,000 | 4.0 | 0.0 |
