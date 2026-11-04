"""
Carry paths: replay the held-out start positions against one opponent and record
where agent_0 goes while it holds the enemy flag, for the original and edited policy.

    python -m wsp.carry_paths --edited runs/<run>/edited.pt --episodes 40

Writes carries.json next to --edited: per match, every carry's path (metres),
its length in steps and how it ended (captured = crossed back over the scrimmage
line, tagged, or time ran out). The heuristic opponents draw on randomness the
start seed does not fix, so a replay starts each match like the scored held-out
run but does not always play out step for step the same.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from wsp.heldout import HELDOUT_SEED, opponent_label
from wsp.matches import MatchRunner
from wsp.sim import MatchConfig, load_actors, make_env
from wsp.tasks import TASKS


def carries(trace: dict, agent: int = 0) -> list[dict]:
    """Contiguous has_flag stretches of one blue agent, with how each ended."""
    hf = trace["has_flag"][:, agent].astype(bool)
    tagged = trace["is_tagged"][:, agent].astype(bool)
    pos = trace["pos"][:, agent]
    out, t, n = [], 0, len(hf)
    while t < n:
        if not hf[t]:
            t += 1
            continue
        start = t
        while t < n and hf[t]:
            t += 1
        if t >= n:
            outcome = "time"
        elif tagged[t] or tagged[t - 1]:
            outcome = "tagged"
        else:
            outcome = "captured"
        path = pos[max(start - 1, 0):min(t + 1, n)]  # from the grab to the step it ended
        out.append({"steps": t - start, "outcome": outcome, "path": np.round(path, 2).tolist()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="carry_home", choices=sorted(TASKS))
    ap.add_argument("--edited", required=True, help="edited checkpoint (.pt)")
    ap.add_argument("--original", help="defaults to the task's checkpoint")
    ap.add_argument("--opponent", help="defaults to the task's first held-out opponent")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", help="output directory (default: next to --edited)")
    args = ap.parse_args()

    task = TASKS[args.task]
    policies = {"original": load_actors(args.original or task.ckpt), "edited": load_actors(args.edited)}
    opponent = args.opponent or task.heldout_opponents[0]
    out = Path(args.out or Path(args.edited).parent)
    traces = out / "carry_traces"
    traces.mkdir(parents=True, exist_ok=True)
    seeds = list(range(HELDOUT_SEED, HELDOUT_SEED + args.episodes))

    runner = MatchRunner(MatchConfig(team_size=policies["original"].team_size), args.workers)
    try:
        metrics = {name: runner.run(actors, opponent, seeds,
                                    trace_paths=[traces / f"{name}_{s}.npz" for s in seeds])
                   for name, actors in policies.items()}
    finally:
        runner.close()

    env = make_env(MatchConfig())
    geometry = {"field_m": np.asarray(env.env_size).tolist(),
                "scrimmage": np.asarray(env.scrimmage_coords).tolist(),
                "flag_home": np.asarray(env.state["flag_home"]).tolist()}
    result = {"opponent": opponent, "seeds": seeds, "geometry": geometry, "matches": {}}
    for name in policies:
        result["matches"][name] = [
            {"seed": s, "blue_score": m["blue_score"], "red_score": m["red_score"],
             "carries": carries(dict(np.load(traces / f"{name}_{s}.npz")))}
            for s, m in zip(seeds, metrics[name])]
    (out / "carries.json").write_text(json.dumps(result), encoding="utf-8")

    print(f"vs {opponent_label(opponent)}, {len(seeds)} start positions:")
    for name, rows in result["matches"].items():
        cs = [c for r in rows for c in r["carries"]]
        ends = {k: sum(c["outcome"] == k for c in cs) for k in ("captured", "tagged", "time")}
        print(f"  {name}: {len(cs)} carries, {ends['captured']} captured, {ends['tagged']} tagged, "
              f"{ends['time']} cut off by time")
    print(f"Paths: {out / 'carries.json'}")


if __name__ == "__main__":
    main()
