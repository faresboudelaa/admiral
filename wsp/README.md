# Weight-space programming with Muse Spark

Muse Spark reads the raw weights of a trained PyQuaticus MAPPO policy and edits them directly. It uses no gradients, optimiser or training. The harness then scores the edited policy against the original on held-out matches.

This follows the setup of *Weight-Space Programming* (MIT CSAIL, Nov 2026), where GPT-6 Astra repaired a Kinetix PPO policy by editing 3.5% of its 171,664 weights. Here the editor is Muse Spark and the target is our own 2v2 capture-the-flag policy (163,874 weights).

## Run it

Put the Model API key in `admiral/.env` (never commit it):

```
MODEL_API_KEY=...
```

Start Docker Desktop. Build the sandbox image once (this is the only step that uses the network):

```bash
docker build -t admiral-wsp-sandbox:1 wsp/docker
```

Then, from `admiral/`:

```bash
.venv/Scripts/python.exe -m wsp.run
```

Defaults: task `carry_home`, model `muse-spark-1.3`, reasoning effort `high`, Responses API, a budget of 100 Python runs and 50 practice matches, and 40 held-out matches per opponent. Useful flags:

| flag | what it does |
|---|---|
| `--effort max` | maximum reasoning effort (muse-spark-1.3 and muse-wonder-1.0 only) |
| `--model muse-wonder-1.0` | the long-horizon agentic preview model |
| `--protocol chat` | fall back to Chat Completions if the Responses API misbehaves |
| `--task from_scratch` | write a policy into the untrained network instead |
| `--match-budget 0` | a "blind" edit: weights and recorded matches only, no feedback |
| `--scripted-test --heldout 4` | check the harness end to end without an API key |
| `--sandbox local` | run the kernel as a plain child process; allowed only with `--scripted-test` |

Every run writes to `runs/<task>_<model>_<time>/`:

| file | contents |
|---|---|
| `prompt.txt` | the exact first message the model saw |
| `transcript.md` | readable log: model text, every tool call and its output |
| `events.jsonl` | raw API responses and token usage per turn |
| `tool_calls.jsonl` | every tool call with its full output and timing |
| `edited.pt` | the edited checkpoint (same format as the legacy MAPPO checkpoints) |
| `run.json` | budgets used, tokens, wall time, audit flags, % weights changed |
| `report.md` / `heldout.json` | held-out comparison of the original and edited policies |

To re-score any checkpoint:

```bash
.venv/Scripts/python.exe -m wsp.heldout --edited runs/<run>/edited.pt --episodes 100
```

To see where agent_0 goes while it carries the flag, before and after the edit:

```bash
.venv/Scripts/python.exe -m wsp.carry_paths --edited runs/<run>/edited.pt --episodes 40
```

The heuristic opponents use randomness that the start seed does not fix, so a replay starts every match like the scored run but does not always play out identically.

## The tasks

**`carry_home` (default): a repair.** The target is `checkpoints/legacy/mappo_2v2_selfplay_final.pt`, a self-play policy trained for 1,500 updates. Against the heuristic "hard" team, agent_0 grabs the enemy flag about 1.5 times per match but has never scored in any measured match. While it carries the flag, it moves toward its own half at about 0.2 m/s (max speed is 2.0) and picks the *stop* action about 17% of the time. Every carry ends in a tag. This is a single, localised fault, like Kinetix's "stuck on the ramp". A capture only needs the carrier to cross back over the scrimmage line, so the needed behaviour is simple: when `has_flag` is on, steer toward `own_home_bearing` at full speed.

**`from_scratch`: writing a policy.** `mappo reward hacking.pt` turned out **not** to be a trained policy. It has `update_count` 0, every bias exactly 0, and an output layer at the 0.01 orthogonal-init scale, so it acts almost uniformly at random. That makes it a target for writing a policy from scratch (the paper's MinAtar experiment), not for a repair.

## How the harness maps to the paper

- **Weights in a sandbox, not in the prompt.** As in the paper's Go and Qwen experiments, the weights sit in a persistent Python kernel as numpy arrays (`W`, with the original in `W0`). The model inspects them with code: activations on recorded states, unit statistics, and so on.
- **Budgeted forward passes.** Seeing the policy play costs matches from a fixed budget (`evaluate`). Running the network offline on recorded states is free, which matches the paper's "inspect activations on recorded states".
- **Free starting evidence.** The prompt includes the environment description, the observation and action tables, and statistics from 8 recorded matches of the original policy. The kernel holds those 8 matches as traces, like the one recorded episode Astra got for Kinetix.
- **Neutral instructions.** The system message is the one line the authors used ("Answer the user's request."). Everything else is in the task prompt.
- **Held-out scoring.** The editor never sees the held-out matches: they use different start seeds and include two opponents it never played (heuristic "medium" and the original policy playing red). Both policies play the same seeds (a paired comparison), and the report gives bootstrap 95% CIs.
- **Isolated execution.** The model's code runs in a Docker container (`wsp/docker/Dockerfile`, numpy only). The container has no network, a read-only root filesystem and no Linux capabilities. It is limited to 4 GB of memory, 2 CPUs and 128 processes, and sees only this run's `sandbox/` folder. The container is the security boundary, as bubblewrap was for the authors. Nothing coming back is trusted: weights return as `.npz` files without pickle, and the host checks them against the original shapes before any match uses them. Tested: there is no route to the internet and DNS fails, no host files or environment variables are visible, writes outside `/work` fail, and forged weight files are rejected.
- **No-training audit.** Inside the container, the kernel cannot import the simulator, torch, jax, sklearn or network clients. Code containing `lstsq`, `pinv`, `.backward(`, `optimizer`, `minimize(` and similar is flagged in `run.json` for manual review.

## Caveats

- **Two things run on the host.** The match simulator and the held-out scoring run on the host, but they only receive validated float arrays from the sandbox, never code.
- **The audit catches obvious fitting only.** It is a pattern check, so a determined regression written by hand would get past it. The transcript is the real audit.
- **Report sample sizes.** 40 held-out matches per opponent separates large effects, not small ones. Raise `--episodes` before quoting a number.
- **Both outcomes are results.** A failed repair is still the first test of weight-space programming with a non-OpenAI editor.

## Files

| file | job |
|---|---|
| `sim.py` | checkpoint ↔ numpy, numpy forward pass, opponents, seeded matches, behaviour metrics |
| `matches.py` | runs batches of matches in parallel processes |
| `sandbox_worker.py`, `sandbox.py` | the editor's persistent Python kernel (in Docker) and its host-side driver |
| `docker/Dockerfile` | the sandbox image: python 3.10 slim and numpy 1.26.4, running as a non-root user |
| `tasks.py` | task presets and the prompt |
| `editor.py` | tool definitions, Responses / Chat / scripted clients, the edit loop |
| `heldout.py` | paired held-out comparison and `report.md` |
| `carry_paths.py` | replays held-out start positions and records each flag carry's path (`carries.json`) |
| `run.py` | the entry point |
| `_compat.py` | stubs `pymoos`, which PyQuaticus imports but simulation never uses (it needs MSVC to build on Windows) |

## Gotchas found while building this

- PyQuaticus overwrites the arrays of the previous observation in place on every `env.step`. Copy observations before stepping if you store them. The legacy training code was not affected, because it stacked a copy before stepping.
- Positions in `env.state` are also updated in place, so copy them before keeping a history.
- Headings are nautical (0° = +y, 90° = +x, clockwise). Observation bearings are relative to the boat's own heading. A positive action heading turns clockwise, so the action whose heading change is closest to a target's bearing steers toward it. This was checked against recorded traces.
