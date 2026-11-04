# Weight-space programming report: `carry_home`

## Weights changed

4,809 of 163,874 weights (2.935%).

| tensor | changed | of | max abs delta |
|---|---:|---:|---:|
| agent_0.W1 | 360 | 11,520 | 6.354 |
| agent_0.b1 | 8 | 256 | 9.399 |
| agent_0.W2 | 4,032 | 65,536 | 3.381 |
| agent_0.b2 | 256 | 256 | 2.388 |
| agent_0.W3 | 136 | 4,352 | 5.119 |
| agent_0.b3 | 17 | 17 | 0.4895 |

## Held-out matches (200 per opponent, seeds 900000+, same seeds for both policies)

### vs heuristic:hard

| metric | original | edited |
|---|---:|---:|
| W / D / L | 0 / 81 / 119 | 33 / 69 / 98 |
| mean score (blue : red) | 0.00 : 0.77 | 0.41 : 0.81 |
| score difference | -0.77 | -0.40 |
| flag grabs / match | 1.09 | 1.26 |
| captures per grab | 0.00 | 0.33 |
| carrying steps / match | 45.9 | 30.7 |
| speed toward own half while carrying (m/s) | 0.28 | 1.68 |
| stop action while carrying | 19% | 1% |
| times tagged / match | 2.86 | 2.75 |
| out-of-bounds / match | 0.80 | 0.85 |
| tags made / match | 2.43 | 2.50 |
| time on enemy half | 29% | 29% |
| loiter rate | 0.0% | 0.0% |
| spin rate | 0.0% | 0.0% |
| dead-stop rate | 0.0% | 0.0% |

Paired change in score difference: +0.36 (95% bootstrap CI +0.24 to +0.50); captures per match: +0.41 (+0.33 to +0.50).

### vs heuristic:medium

| metric | original | edited |
|---|---:|---:|
| W / D / L | 0 / 189 / 11 | 4 / 182 / 14 |
| mean score (blue : red) | 0.00 : 0.06 | 0.03 : 0.08 |
| score difference | -0.06 | -0.05 |
| flag grabs / match | 0.38 | 0.41 |
| captures per grab | 0.00 | 0.07 |
| carrying steps / match | 9.4 | 5.0 |
| speed toward own half while carrying (m/s) | 0.25 | 1.55 |
| stop action while carrying | 21% | 1% |
| times tagged / match | 4.29 | 4.34 |
| out-of-bounds / match | 0.58 | 0.66 |
| tags made / match | 6.33 | 6.33 |
| time on enemy half | 21% | 20% |
| loiter rate | 0.0% | 0.0% |
| spin rate | 0.0% | 0.0% |
| dead-stop rate | 0.0% | 0.0% |

Paired change in score difference: +0.01 (95% bootstrap CI -0.04 to +0.04); captures per match: +0.03 (+0.01 to +0.06).

### vs checkpoint mappo_2v2_selfplay_final (as red)

| metric | original | edited |
|---|---:|---:|
| W / D / L | 1 / 197 / 2 | 28 / 172 / 0 |
| mean score (blue : red) | 0.01 : 0.01 | 0.15 : 0.01 |
| score difference | -0.01 | +0.14 |
| flag grabs / match | 0.57 | 0.57 |
| captures per grab | 0.01 | 0.26 |
| carrying steps / match | 49.1 | 14.3 |
| speed toward own half while carrying (m/s) | 0.30 | 1.72 |
| stop action while carrying | 14% | 0% |
| times tagged / match | 3.83 | 3.98 |
| out-of-bounds / match | 0.53 | 0.55 |
| tags made / match | 3.38 | 3.39 |
| time on enemy half | 22% | 20% |
| loiter rate | 0.0% | 0.0% |
| spin rate | 0.0% | 0.0% |
| dead-stop rate | 0.0% | 0.0% |

Paired change in score difference: +0.15 (95% bootstrap CI +0.10 to +0.20); captures per match: +0.14 (+0.10 to +0.20).
