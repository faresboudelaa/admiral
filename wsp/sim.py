"""
Simulation layer for weight-space programming (WSP).

Everything the editor's tools and the held-out evaluation need to run a policy
in PyQuaticus: checkpoint <-> numpy weights, a numpy forward pass, opponents,
seeded matches, and per-episode behaviour metrics.

The actor architecture is the one from legacy/mappo_projectv1.py:
    obs -> Linear(256) -> tanh -> Linear(256) -> tanh -> Linear(17) logits
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "legacy") not in sys.path:
    sys.path.insert(0, str(ROOT / "legacy"))

from wsp._compat import ensure_pymoos  # noqa: E402

ensure_pymoos()

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")

from pyquaticus import pyquaticus_v0  # noqa: E402
from pyquaticus.config import ACTION_MAP, config_dict_std  # noqa: E402

ACT_DIM = 17
OBS_DIM_TO_TEAM = {29: 1, 45: 2, 61: 3, 77: 4}

# numpy name <-> torch state_dict name
LAYERS = [("W1", "net.0.weight"), ("b1", "net.0.bias"),
          ("W2", "net.2.weight"), ("b2", "net.2.bias"),
          ("W3", "net.4.weight"), ("b3", "net.4.bias")]

# Behaviour detectors, same thresholds as the training rewards in legacy/mappo_projectv1.py
LOITER_WINDOW = 15     # steps
LOITER_RADIUS = 0.30   # metres
SPIN_WINDOW = 6        # steps
SPIN_THRESHOLD = 180.0 # degrees of cumulative turn while nearly stopped
SPIN_MAX_SPEED = 0.05
DEAD_STOP_DIST = 0.01  # metres moved in one step


def action_table() -> list[str]:
    """Human-readable meaning of each discrete action index."""
    rows = []
    for i, (spd, hdg) in enumerate(ACTION_MAP):
        if spd == 0.0:
            rows.append(f"{i}: stop (speed 0)")
        else:
            rows.append(f"{i}: speed {spd:.1f} x max, turn {hdg:+d} deg relative to current heading")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoints <-> numpy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActorSet:
    """The actor weights of one MAPPO checkpoint, as float32 numpy arrays.

    `nets` maps a checkpoint actor name (e.g. "agent_0") to {"W1": ..., "b3": ...}.
    With a shared policy there is one net and every blue agent uses it.
    """
    nets: dict
    shared: bool
    obs_dim: int
    source: str = ""

    @property
    def team_size(self) -> int:
        return OBS_DIM_TO_TEAM[self.obs_dim]

    def net_for(self, agent_index: int) -> dict:
        names = sorted(self.nets)
        return self.nets[names[0]] if self.shared else self.nets[names[agent_index]]

    def copy(self) -> "ActorSet":
        return ActorSet({k: {n: a.copy() for n, a in v.items()} for k, v in self.nets.items()},
                        self.shared, self.obs_dim, self.source)

    def n_weights(self) -> int:
        return int(sum(a.size for v in self.nets.values() for a in v.values()))


def load_actors(path: str | Path) -> ActorSet:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    nets = {}
    for key, sd in payload.items():
        if key.startswith("actor_") and isinstance(sd, dict) and "net.0.weight" in sd:
            nets[key[len("actor_"):]] = {n: sd[t].detach().cpu().numpy().astype(np.float32).copy()
                                         for n, t in LAYERS}
    if not nets:
        raise ValueError(f"No actor_* state dicts found in {path}")
    shared = bool(payload.get("shared_policy", len(nets) == 1)) or len(nets) == 1
    obs_dim = next(iter(nets.values()))["W1"].shape[1]
    return ActorSet(nets, shared, obs_dim, str(path))


def save_actors(actors: ActorSet, src_ckpt: str | Path, out_path: str | Path, note: str = ""):
    """Write `actors` into a copy of `src_ckpt`, so legacy MAPPO.load() still works."""
    payload = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)
    for name, net in actors.nets.items():
        payload[f"actor_{name}"] = {t: torch.from_numpy(np.ascontiguousarray(net[n], dtype=np.float32))
                                    for n, t in LAYERS}
    payload["wsp_edit_note"] = note
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))


def forward(net: dict, obs: np.ndarray, hidden: bool = False):
    """Batched actor forward pass. obs: (B, obs_dim) or (obs_dim,)."""
    x = np.asarray(obs, dtype=np.float32)
    h1 = np.tanh(x @ net["W1"].T + net["b1"])
    h2 = np.tanh(h1 @ net["W2"].T + net["b2"])
    logits = h2 @ net["W3"].T + net["b3"]
    return (logits, h1, h2) if hidden else logits


def weight_diff(a: ActorSet, b: ActorSet, tol: float = 0.0) -> dict:
    """How much of `b` differs from `a`: overall and per tensor."""
    per = {}
    changed = total = 0
    for name in a.nets:
        for n, _ in LAYERS:
            x, y = a.nets[name][n], b.nets[name][n]
            d = np.abs(x - y) > tol
            per[f"{name}.{n}"] = {"changed": int(d.sum()), "size": int(d.size),
                                  "max_abs_delta": float(np.abs(x - y).max())}
            changed += int(d.sum())
            total += int(d.size)
    return {"changed": changed, "total": total, "pct": 100.0 * changed / max(total, 1), "per_tensor": per}


# ─────────────────────────────────────────────────────────────────────────────
#  Environment and opponents
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchConfig:
    team_size: int = 2
    max_time: float = 600.0
    max_score: int = 3
    sim_speedup: int = 8
    tagging_cooldown: float = 30.0
    scenario: str = "on_sides"   # random own-half starts, see legacy/scenario_init.py


def make_env(cfg: MatchConfig):
    c = dict(config_dict_std)
    c.update(sim_speedup_factor=cfg.sim_speedup, max_score=cfg.max_score, max_time=cfg.max_time,
             tagging_cooldown=cfg.tagging_cooldown, tag_on_oob=True)
    n = cfg.team_size
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=c, render_mode=None,
                                      reward_config={f"agent_{i}": None for i in range(2 * n)},
                                      team_size=n)
    env.reset(seed=0)
    return env


def obs_feature_names(env) -> list[str]:
    """Names of the observation vector entries, in order (normalised obs, roughly in [-1, 1])."""
    return [name for name, _, _ in obs_spec(env)]


def obs_spec(env) -> list[tuple[str, float, float]]:
    """(name, raw_low, raw_high) per observation entry. The env maps [low, high] linearly onto [-1, 1]."""
    out = []
    for k, box in env.agent_obs_normalizer._bounds.items():
        name = k if isinstance(k, str) else f"{k[0]}.{k[1]}"
        lo, hi = np.asarray(box.low).ravel(), np.asarray(box.high).ravel()
        for j in range(lo.size):
            out.append((name if lo.size == 1 else f"{name}[{j}]", float(lo[j]), float(hi[j])))
    return out


class NetTeam:
    """A team driven by an ActorSet (blue = the edited policy, or a red self-play opponent)."""

    def __init__(self, actors: ActorSet, agent_ids: list[str], greedy: bool = False):
        self.actors, self.agent_ids, self.greedy = actors, agent_ids, greedy

    def act(self, obs: dict, info: dict, rng: np.random.Generator):
        actions, logits = {}, {}
        for i, aid in enumerate(self.agent_ids):
            o = obs.get(aid)
            if o is None:
                continue
            lg = forward(self.actors.net_for(i), o)
            logits[aid] = lg
            if self.greedy:
                actions[aid] = int(np.argmax(lg))
            else:
                p = np.exp(lg - lg.max())
                actions[aid] = int(rng.choice(ACT_DIM, p=p / p.sum()))
        return actions, logits


class HeuristicTeam:
    def __init__(self, env, agent_ids: list[str], mode: str = "hard"):
        from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
        self.agents = {a: Heuristic_CTF_Agent(a, env, mode=mode, continuous=False) for a in agent_ids}

    def act(self, obs, info, rng):
        return {a: ag.compute_action(obs, info) for a, ag in self.agents.items()}, {}


class RandomTeam:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids

    def act(self, obs, info, rng):
        return {a: int(rng.integers(ACT_DIM)) for a in self.agent_ids}, {}


def make_opponent(spec: str, env, red_ids: list[str]):
    """spec: 'heuristic:hard' | 'heuristic:medium' | 'heuristic:easy' | 'random' | 'ckpt:<path>'"""
    if spec.startswith("heuristic"):
        mode = spec.split(":", 1)[1] if ":" in spec else "hard"
        return HeuristicTeam(env, red_ids, mode)
    if spec == "random":
        return RandomTeam(red_ids)
    if spec.startswith("ckpt:"):
        return NetTeam(load_actors(spec[5:]), red_ids)
    raise ValueError(f"Unknown opponent spec {spec!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  Episodes and behaviour metrics
# ─────────────────────────────────────────────────────────────────────────────

def _ang(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class Episode:
    seed: int
    metrics: dict
    trace: dict = field(default_factory=dict)   # per-step arrays, blue agents only


def run_episode(env, cfg: MatchConfig, blue: NetTeam, red, seed: int, record: bool = False) -> Episode:
    from scenario_init import random_reset

    n = cfg.team_size
    blue_ids = [f"agent_{i}" for i in range(n)]
    obs, info = random_reset(env, scenario=cfg.scenario, seed=seed)
    rng = np.random.default_rng(seed)

    st = env.state
    pos_hist = {a: [] for a in blue_ids}
    hdg_hist = {a: [] for a in blue_ids}
    tr = {k: [] for k in ["obs", "action", "logits", "pos", "heading", "speed", "on_side",
                          "has_flag", "is_tagged", "oob", "loiter", "spin", "dead_stop"]}
    m = {a: dict(steps=0, active_steps=0, dead_stop=0, loiter=0, spin=0, enemy_side=0,
                 crossings_out=0, oob_events=0, tagged_events=0, tags_made=0, grabs=0,
                 near_enemy_flag_no_flag=0, dist=0.0, action_counts=[0] * ACT_DIM,
                 carry_steps=0, carry_progress=0.0, carry_stop=0)
         for a in blue_ids}
    enemy_flag = np.asarray(st["flag_home"][1], dtype=float)
    # captures happen on crossing back over the (vertical) scrimmage line onto blue's half
    scrim_x = float(np.asarray(env.scrimmage_coords)[0][0])
    home_dir = float(np.sign(st["flag_home"][0][0] - scrim_x))
    catch_r = float(env.catch_radius)

    steps = 0
    while True:
        b_act, b_logits = blue.act(obs, info, rng)
        r_act, _ = red.act(obs, info, rng)
        prev = {k: np.array(st[k], copy=True) for k in
                ["agent_position", "agent_on_sides", "agent_oob", "agent_is_tagged", "agent_has_flag"]}
        obs_before = {k: np.array(v, copy=True) for k, v in obs.items()}  # env.step overwrites obs arrays in place
        obs, _, term, trunc, info = env.step({**b_act, **r_act})
        st = env.state
        steps += 1

        step_rec = {k: [] for k in tr}
        for i, a in enumerate(blue_ids):
            mm = m[a]
            p = np.array(st["agent_position"][i], dtype=float)  # env mutates in place
            spd = float(st["agent_speed"][i])
            hdg = float(st["agent_heading"][i])
            tagged, oob = bool(st["agent_is_tagged"][i]), bool(st["agent_oob"][i])
            on_side, has_flag = bool(st["agent_on_sides"][i]), bool(st["agent_has_flag"][i])
            moved = float(np.linalg.norm(p - prev["agent_position"][i]))
            mm["steps"] += 1
            mm["dist"] += moved
            if a in b_act:
                mm["action_counts"][b_act[a]] += 1
            mm["tagged_events"] += int(tagged and not prev["agent_is_tagged"][i])
            mm["oob_events"] += int(oob and not prev["agent_oob"][i])
            mm["grabs"] += int(has_flag and not prev["agent_has_flag"][i])
            mm["tags_made"] += int(st["agent_made_tag"][i] is not None)
            mm["crossings_out"] += int(bool(prev["agent_on_sides"][i]) and not on_side)
            if prev["agent_has_flag"][i] and not prev["agent_is_tagged"][i]:
                # this step's action was chosen while carrying the enemy flag
                p0 = prev["agent_position"][i]
                mm["carry_steps"] += 1
                mm["carry_progress"] += float((p[0] - p0[0]) * home_dir)
                mm["carry_stop"] += int(b_act.get(a) == ACT_DIM - 1)

            lo = sp = ds = False
            if tagged or oob:
                pos_hist[a].clear()
                hdg_hist[a].clear()
            else:
                mm["active_steps"] += 1
                mm["enemy_side"] += int(not on_side)
                near_flag = np.linalg.norm(p - enemy_flag) < 1.5 * catch_r
                mm["near_enemy_flag_no_flag"] += int(near_flag and not has_flag)
                pos_hist[a].append(p)
                hdg_hist[a].append(hdg)
                pos_hist[a] = pos_hist[a][-LOITER_WINDOW:]
                hdg_hist[a] = hdg_hist[a][-SPIN_WINDOW:]
                ds = moved < DEAD_STOP_DIST
                if len(pos_hist[a]) == LOITER_WINDOW:
                    P = np.array(pos_hist[a])
                    lo = float(np.max(np.linalg.norm(P - P.mean(0), axis=1))) < LOITER_RADIUS
                if spd <= SPIN_MAX_SPEED and len(hdg_hist[a]) == SPIN_WINDOW:
                    h = hdg_hist[a]
                    sp = sum(_ang(h[j], h[j - 1]) for j in range(1, len(h))) > SPIN_THRESHOLD
                mm["dead_stop"] += int(ds)
                mm["loiter"] += int(lo)
                mm["spin"] += int(sp)

            if record:
                step_rec["obs"].append(np.asarray(obs_before.get(a, np.zeros(blue.actors.obs_dim)), np.float32))
                step_rec["action"].append(b_act.get(a, -1))
                step_rec["logits"].append(b_logits.get(a, np.full(ACT_DIM, np.nan, np.float32)))
                step_rec["pos"].append(p)
                step_rec["heading"].append(hdg)
                step_rec["speed"].append(spd)
                step_rec["on_side"].append(on_side)
                step_rec["has_flag"].append(has_flag)
                step_rec["is_tagged"].append(tagged)
                step_rec["oob"].append(oob)
                step_rec["loiter"].append(lo)
                step_rec["spin"].append(sp)
                step_rec["dead_stop"].append(ds)
        if record:
            for k, v in step_rec.items():
                tr[k].append(v)

        if any(term.get(a, False) or trunc.get(a, False) for a in blue_ids) or not obs:
            break

    gs = info[blue_ids[0]]["global_state"]
    bs, rs = int(gs["blue_team_score"]), int(gs["red_team_score"])
    agents = {}
    for a, mm in m.items():
        act_n = max(mm["active_steps"], 1)
        agents[a] = dict(
            dead_stop_rate=mm["dead_stop"] / act_n, loiter_rate=mm["loiter"] / act_n,
            spin_rate=mm["spin"] / act_n, enemy_side_rate=mm["enemy_side"] / act_n,
            near_enemy_flag_without_grab_rate=mm["near_enemy_flag_no_flag"] / act_n,
            mean_speed=mm["dist"] / max(mm["steps"], 1) / (cfg.sim_speedup * float(config_dict_std["tau"])),
            crossings_into_enemy_half=mm["crossings_out"], oob_events=mm["oob_events"],
            times_tagged=mm["tagged_events"], tags_made=mm["tags_made"], flag_grabs=mm["grabs"],
            action_hist=[c / max(mm["steps"], 1) for c in mm["action_counts"]],
        )
    team = {k: float(np.mean([agents[a][k] for a in blue_ids]))
            for k in ["dead_stop_rate", "loiter_rate", "spin_rate", "enemy_side_rate",
                      "near_enemy_flag_without_grab_rate", "mean_speed"]}
    for k in ["crossings_into_enemy_half", "oob_events", "times_tagged", "tags_made", "flag_grabs"]:
        team[k] = int(sum(agents[a][k] for a in blue_ids))
    team["carry_steps"] = int(sum(m[a]["carry_steps"] for a in blue_ids))
    team["carry_progress_m"] = float(sum(m[a]["carry_progress"] for a in blue_ids))
    team["carry_stop_actions"] = int(sum(m[a]["carry_stop"] for a in blue_ids))
    metrics = dict(seed=seed, steps=steps, step_seconds=cfg.sim_speedup * float(config_dict_std["tau"]),
                   blue_score=bs, red_score=rs,
                   outcome="win" if bs > rs else "loss" if bs < rs else "draw",
                   team=team, agents=agents)
    trace = {k: np.asarray(v) for k, v in tr.items()} if record else {}
    return Episode(seed, metrics, trace)


def summarize(episodes: list[Episode]) -> dict:
    """Aggregate a batch of episodes (means over episodes, W/D/L counts)."""
    if not episodes:
        return {}
    ms = [e.metrics for e in episodes]
    out = dict(n=len(ms),
               wins=sum(m["outcome"] == "win" for m in ms),
               draws=sum(m["outcome"] == "draw" for m in ms),
               losses=sum(m["outcome"] == "loss" for m in ms),
               mean_score_diff=float(np.mean([m["blue_score"] - m["red_score"] for m in ms])),
               mean_blue_score=float(np.mean([m["blue_score"] for m in ms])),
               mean_red_score=float(np.mean([m["red_score"] for m in ms])),
               mean_steps=float(np.mean([m["steps"] for m in ms])))
    for k in ms[0]["team"]:
        out[k] = float(np.mean([m["team"][k] for m in ms]))
    # pooled over every carrying step in the batch (episodes without a grab add nothing)
    cs = sum(m["team"]["carry_steps"] for m in ms)
    step_s = ms[0]["step_seconds"]
    out["carry_homeward_speed_mps"] = (sum(m["team"]["carry_progress_m"] for m in ms) / (cs * step_s)) if cs else None
    out["carry_stop_rate"] = (sum(m["team"]["carry_stop_actions"] for m in ms) / cs) if cs else None
    out["captures_per_grab"] = (out["mean_blue_score"] / out["flag_grabs"]) if out["flag_grabs"] else None
    hist = np.mean([[np.mean([m["agents"][a]["action_hist"][j] for a in m["agents"]]) for j in range(ACT_DIM)]
                    for m in ms], axis=0)
    out["action_hist"] = [round(float(x), 4) for x in hist]
    return out
