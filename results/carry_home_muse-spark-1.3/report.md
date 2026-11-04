# Weight-space programming report: `carry_home`

## Run

- Editor: `muse-spark-1.3` (reasoning effort `high`, protocol `responses`)
- Turns: 66, wall time 29.7 min, end: submitted
- Budgets used: 62 python runs, 28 practice matches
- Tokens: {'input': 2939194, 'output': 113202, 'reasoning': 71642}
- Audit flags (patterns that could mean fitting instead of writing): [{'turn': 37, 'match': 'lstsq'}]

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

## Held-out matches (40 per opponent, seeds 900000+, same seeds for both policies)

### vs heuristic:hard

| metric | original | edited |
|---|---:|---:|
| W / D / L | 0 / 16 / 24 | 6 / 13 / 21 |
| mean score (blue : red) | 0.00 : 0.70 | 0.42 : 0.80 |
| score difference | -0.70 | -0.38 |
| flag grabs / match | 1.02 | 1.20 |
| captures per grab | 0.00 | 0.35 |
| carrying steps / match | 54.5 | 31.0 |
| speed toward own half while carrying (m/s) | 0.25 | 1.65 |
| stop action while carrying | 20% | 1% |
| times tagged / match | 2.92 | 2.62 |
| out-of-bounds / match | 0.70 | 0.80 |
| tags made / match | 2.42 | 2.52 |
| time on enemy half | 28% | 30% |
| loiter rate | 0.0% | 0.1% |
| spin rate | 0.0% | 0.0% |
| dead-stop rate | 0.0% | 0.0% |

Paired change in score difference: +0.33 (95% bootstrap CI +0.07 to +0.60); captures per match: +0.42 (+0.23 to +0.65).

### vs heuristic:medium

| metric | original | edited |
|---|---:|---:|
| W / D / L | 0 / 39 / 1 | 0 / 39 / 1 |
| mean score (blue : red) | 0.00 : 0.03 | 0.00 : 0.03 |
| score difference | -0.03 | -0.03 |
| flag grabs / match | 0.35 | 0.25 |
| captures per grab | 0.00 | 0.00 |
| carrying steps / match | 9.9 | 2.2 |
| speed toward own half while carrying (m/s) | 0.17 | 1.36 |
| stop action while carrying | 22% | 1% |
| times tagged / match | 4.42 | 4.40 |
| out-of-bounds / match | 0.45 | 0.60 |
| tags made / match | 6.55 | 6.78 |
| time on enemy half | 21% | 19% |
| loiter rate | 0.0% | 0.0% |
| spin rate | 0.0% | 0.0% |
| dead-stop rate | 0.0% | 0.0% |

Paired change in score difference: +0.00 (95% bootstrap CI -0.07 to +0.07); captures per match: +0.00 (+0.00 to +0.00).

### vs checkpoint mappo_2v2_selfplay_final (as red)

| metric | original | edited |
|---|---:|---:|
| W / D / L | 0 / 39 / 1 | 6 / 34 / 0 |
| mean score (blue : red) | 0.00 : 0.03 | 0.15 : 0.00 |
| score difference | -0.03 | +0.15 |
| flag grabs / match | 0.60 | 0.65 |
| captures per grab | 0.00 | 0.23 |
| carrying steps / match | 49.5 | 15.6 |
| speed toward own half while carrying (m/s) | 0.35 | 1.71 |
| stop action while carrying | 14% | 0% |
| times tagged / match | 3.90 | 4.12 |
| out-of-bounds / match | 0.53 | 0.53 |
| tags made / match | 3.10 | 3.40 |
| time on enemy half | 22% | 18% |
| loiter rate | 0.0% | 0.0% |
| spin rate | 0.0% | 0.0% |
| dead-stop rate | 0.0% | 0.0% |

Paired change in score difference: +0.17 (95% bootstrap CI +0.07 to +0.30); captures per match: +0.15 (+0.05 to +0.28).

## Editor's own report

CAUSE. The attacker (agent_0) grabs the flag but its carry-mode policy is blind to the way home. On recorded baseline traces, when has_flag=1 the circular correlation between the homeward bearing (own_home_bearing) and the chosen turn is ~0 (-0.06..+0.12 across matches; attack-mode homing is weak but grabs still happen by approach). Sweeping own_home/scrimmage bearings over their full range on fixed carry observations usually does not change the argmax at all, and when it does the choice does not track the bearing (e.g. bearing +160° -> STOP; many bearings -> a single stuck action). Consequences measured by the harness: 66.9 steps/match carrying at only 0.24 m/s toward the own half, 17% stop actions while carrying, carriers wander (typical carry nets only ~5 m toward home) and every carry episode ends in a tag — 0.00 captures/grab.

EDIT (agent_0 only, agent_1 defender untouched). Installed a gated "return-home" reflex through 8 dead h1 units + 8 dead h2 units (saturated, near-zero variance, tiny output norms):
- Each donor h1 unit k computes tanh(2*own_home_b + 2*scrim_b... precisely (g/2)*b2+(g/2)*b12 + 6*hf - 4*t_k - 6), i.e. a bearing threshold detector that is pinned to a constant (-1) when has_flag=-1 and becomes tanh(4*(avg_bearing - t_k)) when carrying. Thresholds at ±0.2,±0.4,±0.6,±0.8 (≈±36°,±72°,±108°,±144°) form a thermometer code for the homeward bearing.
- Each donor h2 unit copies one h1 unit (single weight 3.0, passthrough), giving a clean staircase code.
- The 8 donor W3 columns implement the telescoping vote V[:,m]=(G/2)(onehot(d(m+1))-onehot(d(m))) with bin->action map [0,7,6,5,4,3,2,1,0] (full-speed turn closest to each bin center; G=10), stop row zeroed.
- Preservation: donor h1 downstream columns were zeroed with their constant drive folded into all 256 b2 biases (using exact old means); non-carry donor-h2 constants and old W3 contributions folded into b3. Verified: non-carry logits argmax agreement 99.8% (median abs diff 0.0025), carry-mode argmax equals the desired homing action on 97% of real carry observations; mean stop probability while carrying fell to ~0-3% with ~93% full-speed probability.
Weight change: 2.93% total (agent_0 only: 8 W1 rows, 8 b1, 8 W2 columns + 8 W2 rows, 256 b2 comps, 8 W3 columns, b3). Shapes/dtypes unchanged.

EFFECT (20 practice matches vs heuristic:hard): captures/grab 0.00 -> ~0.37, speed toward own half 0.24 -> ~1.75 m/s, carry stops 17% -> ~0%, mean score 0.00:0.75 -> ~0.60:x with multiple wins; grabs, tags made, OOB and enemy-half fraction essentially unchanged (OOB events all occur with has_flag=-1, i.e. base policy). Residual failures are interceptions by defenders, not wandering. The reflex keys only on has_flag and homeward bearings, so it is independent of start positions; defense was not touched.
