"""Run batches of seeded matches in parallel worker processes."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wsp.sim import ActorSet, MatchConfig

_env = None
_cfg = None
_opp_cache = {}


def _init(cfg: MatchConfig):
    global _env, _cfg
    from wsp.sim import make_env
    _cfg = cfg
    _env = make_env(cfg)


def _play(args):
    actors, opponent, seed, trace_path = args
    from wsp.sim import NetTeam, load_actors, make_opponent, run_episode
    n = _cfg.team_size
    blue_ids = [f"agent_{i}" for i in range(n)]
    red_ids = [f"agent_{n + i}" for i in range(n)]
    if opponent.startswith("ckpt:"):
        if opponent not in _opp_cache:
            _opp_cache[opponent] = load_actors(opponent[5:])
        red = NetTeam(_opp_cache[opponent], red_ids)
    else:
        red = make_opponent(opponent, _env, red_ids)
    ep = run_episode(_env, _cfg, NetTeam(actors, blue_ids), red, seed=seed, record=trace_path is not None)
    if trace_path is not None:
        np.savez_compressed(trace_path, **ep.trace)
    return ep.metrics


class MatchRunner:
    def __init__(self, cfg: MatchConfig, workers: int | None = None):
        self.cfg = cfg
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
        self.pool = ProcessPoolExecutor(self.workers, initializer=_init, initargs=(cfg,))

    def run(self, actors: ActorSet, opponent: str, seeds: list[int],
            trace_paths: list[Path] | None = None) -> list[dict]:
        tps = trace_paths or [None] * len(seeds)
        jobs = [(actors, opponent, s, (str(t) if t else None)) for s, t in zip(seeds, tps)]
        return list(self.pool.map(_play, jobs))

    def close(self):
        self.pool.shutdown(cancel_futures=True)


class _Ep:
    """Adapter so sim.summarize() can take plain metric dicts."""
    def __init__(self, m):
        self.metrics = m


def summarize_metrics(ms: list[dict]) -> dict:
    from wsp.sim import summarize
    return summarize([_Ep(m) for m in ms])
