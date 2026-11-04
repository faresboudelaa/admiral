"""
Record rendered footage of agent_0's first flag carry, for the original and an
edited policy, from the same start position. Used for the demo video.

    python -m wsp.record_carry --edited results/carry_home_muse-spark-1.3/edited.pt \
        --seeds 900000 900001 900002 --out video_frames

For each seed and policy it plays a match against heuristic "hard", renders
every step (rgb_array), and keeps the window from 45 steps before the grab to
20 steps after the carry ends. Frames go to <out>/<policy>_<seed>/NNNNN.jpg,
with a meta.json describing how the carry ended.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from wsp._compat import ensure_pymoos  # noqa: E402

ensure_pymoos()

import numpy as np  # noqa: E402

from wsp.sim import ROOT, MatchConfig, NetTeam, HeuristicTeam, load_actors  # noqa: E402
from wsp.tasks import TASKS  # noqa: E402


def make_render_env(cfg: MatchConfig):
    from pyquaticus import pyquaticus_v0
    from pyquaticus.config import config_dict_std

    c = dict(config_dict_std)
    c.update(sim_speedup_factor=cfg.sim_speedup, max_score=cfg.max_score, max_time=cfg.max_time,
             tagging_cooldown=cfg.tagging_cooldown, tag_on_oob=True, render_agent_ids=True)
    n = cfg.team_size
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=c, render_mode="rgb_array",
                                      reward_config={f"agent_{i}": None for i in range(2 * n)},
                                      team_size=n)
    env.reset(seed=0)
    return env


def grab_frame(env):
    """Render the current state and return it as an (H, W, 3) uint8 array."""
    import pygame

    env.render()   # rgb_array mode draws onto env.screen and returns None in this PyQuaticus version
    return np.transpose(np.array(pygame.surfarray.pixels3d(env.screen), dtype=np.uint8), (1, 0, 2))


def record(actors, seed: int, out_dir: Path, cfg: MatchConfig, want: str = "any", before: int = 45,
           after: int = 20, max_steps: int = 800):
    """Play one match, then save the window around one of agent_0's carries.

    want: "captured" keeps the first carry that scores, "longest" the longest carry,
          "any" the first carry.
    """
    import io

    from PIL import Image
    from scenario_init import random_reset

    env = make_render_env(cfg)
    n = cfg.team_size
    blue_ids = [f"agent_{i}" for i in range(n)]
    red_ids = [f"agent_{n + i}" for i in range(n)]
    blue, red = NetTeam(actors, blue_ids), HeuristicTeam(env, red_ids, "hard")
    obs, info = random_reset(env, scenario=cfg.scenario, seed=seed)
    rng = np.random.default_rng(seed)

    def jpeg(frame):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    def pos0():
        p = env.env_to_screen(np.array(env.state["agent_position"][0], dtype=float))
        return [round(float(p[0]), 1), round(float(p[1]), 1), bool(env.state["agent_has_flag"][0])]

    frames, carries, grab = [jpeg(grab_frame(env))], [], None
    track = [pos0()]
    for t in range(max_steps):
        b_act, _ = blue.act(obs, info, rng)
        r_act, _ = red.act(obs, info, rng)
        had, caps = bool(env.state["agent_has_flag"][0]), int(env.state["captures"][0])
        obs, _, term, trunc, info = env.step({**b_act, **r_act})
        frames.append(jpeg(grab_frame(env)))
        track.append(pos0())
        has = bool(env.state["agent_has_flag"][0])
        if has and not had:
            grab = t + 1
        if had and not has and grab is not None:
            carries.append(dict(grab=grab, end=t + 1,
                                outcome="captured" if int(env.state["captures"][0]) > caps else "tagged"))
            grab = None
        if any(term.get(a, False) or trunc.get(a, False) for a in blue_ids):
            break
    env.close()
    score = [int(x) for x in env.state["captures"]]
    if not carries:
        return None
    if want == "captured":
        pick = next((c for c in carries if c["outcome"] == "captured"), None)
    elif want == "longest":
        pick = max(carries, key=lambda c: c["end"] - c["grab"])
    else:
        pick = carries[0]
    if pick is None:
        return None
    lo, hi = max(0, pick["grab"] - before), min(len(frames), pick["end"] + after + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames[lo:hi]):
        (out_dir / f"{i:05d}.jpg").write_bytes(f)
    meta = dict(seed=seed, grab_frame=pick["grab"] - lo, end_frame=pick["end"] - lo, outcome=pick["outcome"],
                carry_steps=pick["end"] - pick["grab"], n_frames=hi - lo, match_score=score,
                all_carries=[(c["end"] - c["grab"], c["outcome"]) for c in carries],
                agent0_screen=track[lo:hi])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edited", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[900000])
    ap.add_argument("--out", default="video_frames")
    ap.add_argument("--want-original", default="longest", choices=["any", "longest", "captured"])
    ap.add_argument("--want-edited", default="captured", choices=["any", "longest", "captured"])
    a = ap.parse_args()
    cfg = MatchConfig()
    pols = {"original": load_actors(TASKS["carry_home"].ckpt), "edited": load_actors(a.edited)}
    for seed in a.seeds:
        for name, actors in pols.items():
            want = a.want_original if name == "original" else a.want_edited
            meta = record(actors, seed, Path(a.out) / f"{name}_{seed}", cfg, want=want)
            print(name, seed, meta if meta else "no grab")


if __name__ == "__main__":
    main()
