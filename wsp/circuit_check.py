"""
Circuit check: run the original and edited agent_0 networks on the states
recorded by carry_paths.py and measure what the edit changed.

    python -m wsp.circuit_check --edited runs/<run>/edited.pt

Reads <run>/carry_traces/*.npz and writes circuit_check.json next to --edited:
- turn chosen vs direction home while carrying (mean probability per 45-degree
  sector, both speeds merged, plus stop), for both networks;
- a knock-out: the edited network with its changed W3 columns zeroed;
- top-action agreement and mean total-variation distance against the original,
  on carrying and non-carrying states.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from wsp.sim import forward, load_actors
from wsp.tasks import TASKS

HEADINGS = [180, 135, 90, 45, 0, -45, -90, -135]  # actions a and a + 8 turn by HEADINGS[a]; 16 is stop
OBS_HOME_BEARING, OBS_HAS_FLAG = 2, 15


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def turn_probs(p: np.ndarray) -> np.ndarray:
    """(B, 17) action probabilities -> (B, 9): the 8 turn headings, both speeds merged, then stop."""
    return np.concatenate([p[:, :8] + p[:, 8:16], p[:, 16:]], 1)


def sector(deg: np.ndarray) -> np.ndarray:
    """Index into HEADINGS of the 45-degree sector each bearing falls in."""
    diff = np.abs((deg[:, None] - np.array(HEADINGS)[None, :] + 180) % 360 - 180)
    return diff.argmin(1)


def agreement(a: dict, b: dict, obs: np.ndarray) -> dict:
    pa, pb = softmax(forward(a, obs)), softmax(forward(b, obs))
    return {"n": len(obs), "top_action_agree": float((pa.argmax(1) == pb.argmax(1)).mean()),
            "mean_total_variation": float(0.5 * np.abs(pa - pb).sum(1).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="carry_home", choices=sorted(TASKS))
    ap.add_argument("--edited", required=True, help="edited checkpoint (.pt)")
    ap.add_argument("--original", help="defaults to the task's checkpoint")
    ap.add_argument("--agent", default="agent_0")
    args = ap.parse_args()

    run = Path(args.edited).parent
    orig = load_actors(args.original or TASKS[args.task].ckpt).nets[args.agent]
    edit = load_actors(args.edited).nets[args.agent]
    changed = np.flatnonzero(np.any(orig["W3"] != edit["W3"], axis=0))
    knockout = {k: v.copy() for k, v in edit.items()}
    knockout["W3"][:, changed] = 0.0

    traces = sorted((run / "carry_traces").glob("*.npz"))
    if not traces:
        raise SystemExit(f"no traces in {run / 'carry_traces'}: run python -m wsp.carry_paths first")
    i = int(args.agent.split("_")[1])
    obs = []
    for f in traces:
        with np.load(f) as z:
            obs.append(z["obs"][z["action"][:, i] >= 0, i])
    obs = np.concatenate(obs)
    carrying = obs[obs[:, OBS_HAS_FLAG] > 0]
    other = obs[obs[:, OBS_HAS_FLAG] <= 0]

    s = sector(carrying[:, OBS_HOME_BEARING] * 180.0)
    counts = np.bincount(s, minlength=len(HEADINGS))
    out = {"agent": args.agent, "traces": len(traces), "headings": HEADINGS, "sector_counts": counts.tolist(),
           "knocked_out_W3_columns": changed.tolist(), "turn_vs_home": {}}
    for name, net in [("original", orig), ("edited", edit)]:
        p = softmax(forward(net, carrying))
        t = turn_probs(p)
        m = np.stack([t[s == j].mean(0) if counts[j] else np.zeros(9) for j in range(len(HEADINGS))], 1)
        out["turn_vs_home"][name] = {
            "matrix_turn_by_sector": m.round(4).tolist(),
            "toward_home": float(t[np.arange(len(s)), s].mean()),
            "stop": float(p[:, 16].mean()), "full_speed": float(p[:, :8].sum(1).mean())}
    out["carrying"] = {"edited": agreement(orig, edit, carrying), "knockout": agreement(orig, knockout, carrying)}
    out["not_carrying"] = {"edited": agreement(orig, edit, other), "knockout": agreement(orig, knockout, other)}
    (run / "circuit_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    for name, r in out["turn_vs_home"].items():
        print(f"{name}: turns toward home {r['toward_home']:.1%}, stops {r['stop']:.1%}, "
              f"full speed {r['full_speed']:.1%} ({len(carrying)} carrying states)")
    for part in ("carrying", "not_carrying"):
        for name, r in out[part].items():
            print(f"{part.replace('_', ' ')}, original vs {name}: same top action {r['top_action_agree']:.2%}, "
                  f"mean total variation {r['mean_total_variation']:.4f} (n={r['n']})")
    print(f"Check: {run / 'circuit_check.json'}")


if __name__ == "__main__":
    main()
