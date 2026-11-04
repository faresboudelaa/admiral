![pyquaticus](pyquaticus.png)

# Pyquaticus — APM_5DA01_TP

This is a fork of [pyquaticus](https://github.com/mit-ll-trusted-autonomy/pyquaticus) for the class **APM_5DA01_TP** on multiagent systems.

Pyquaticus is a Capture-the-Flag environment for autonomous marine vehicles. Two teams (blue and red) compete to grab the opposing team's flag and return it to their base, while tagging opponents who venture into their territory.

## Setup

```bash
uv venv
uv pip install -e ".[ray,torch]"
```

Test and play the environment manually with keyboard controls (blue team: arrow keys, red team: WASD):

```bash
uv run python test/arrowkeys_test.py
```

## Training with RLlib (Ray)

[rl_test/train_2v2.py](rl_test/train_2v2.py) sets up a 2v2 self-play PPO run using RLlib.

```bash
uv run python rl_test/train_2v2.py
```

The reference throughput in a laptop is around ~200 SPS (steps per second).

## Training with PufferLib (faster)

PufferLib achieves better throughput (~700 SPS) through more efficient vectorization.
Install it separately:

```bash
uv pip install pufferlib==3.0.0 --no-build-isolation --no-binary pufferlib --reinstall
```

Train and evaluate:

```bash
uv run python rl_test/puffer_2v2.py --train   # train; saves checkpoints, logs to WandB
uv run python rl_test/puffer_2v2.py           # evaluate and render to pyquaticus/videos/
```

Pass `--model <path>` to evaluate a specific checkpoint or model.

---

## Project

The project is open-ended: extend the environment and/or the learning algorithm and evaluate the impact. It can be based on **this repo** or on [pufferquaticus](https://codeberg.org/pauhmolins/pufferquaticus), a fork of PufferLib that implements a faster but simpler version of the environment.

### Ideas

- **Use a different algorithm**: try MAPPO, IPPO, or other MARL algorithms.
- **Modify the observation space**: add or remove features and study the effect on learned behavior.
- **Modify the reward function**: shape rewards to encourage different emergent behaviors.
- **Add role specialization**: assign each agent a fixed role (attacker / defender) and train specialized policies.

### Working in this repo

- **Algorithm:** play around with the RLlib algorithm in [rl_test/train_2v2.py](rl_test/train_2v2.py) and tweak different options ([documentation](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.multi_agent.html)).
- **Observation space / environment changes:** modify [pyquaticus/envs/pyquaticus.py](pyquaticus/envs/pyquaticus.py), for instance `state_to_obs()` to change the local observation computation.
- **Rewards:** edit or add a function in [pyquaticus/utils/rewards.py](pyquaticus/utils/rewards.py), then point `reward_config` in [rl_test/train_2v2.py](rl_test/train_2v2.py) to it.
- **Role specialization:** assign different reward functions in [rl_test/train_2v2.py](rl_test/train_2v2.py) or augment the observation space with a one-hot encoding of the role in [pyquaticus/envs/pyquaticus.py](pyquaticus/envs/pyquaticus.py).

### Tips
- Start with small changes and test them out in a quick training run (e.g. 10k steps) to verify they work as expected.
- Use WandB to track experiments and compare results across different configurations (if using [rl_test/puffer_2v2.py](rl_test/puffer_2v2.py))
- You can debug functions (e.g., reward or observation computation) by adding `breakpoint()` where needed. This will drop you into an interactive [pdb](https://docs.python.org/3/library/pdb.html) session. Note that this can be tricky when running with multiple workers, so you may want to test with a single environment (e.g., running [test/arrowkeys_test.py](test/arrowkeys_test.py)).