"""
scenario_init.py
────────────────
Randomised scenario generator for PyQuaticus.

Usage
─────
    from scenario_init import random_reset

    obs, info = random_reset(env)                        # pick scenario at random
    obs, info = random_reset(env, scenario="flag_run")   # force a specific scenario
    obs, info = random_reset(env, seed=42)               # reproducible

Available scenarios
───────────────────
    "scattered"      – Fully random valid positions across the whole field.
    "on_sides"       – All agents start on their own half (default-like but random).
    "flag_run"       – One attacker is carrying the enemy flag and sprinting home;
                       one or more defenders are chasing.
    "flag_grab"      – An attacker is right next to the enemy flag (about to grab);
                       a defender is nearby but not yet in tag range.
    "interception"   – Attacker has the flag; a defender is placed directly on
                       the shortest path between the attacker and the attacker's home.
    "standoff"       – All agents cluster near the scrimmage line, heading across.
    "multi_threat"   – Multiple attackers deep in enemy territory simultaneously.
    "chase"          – One attacker is deep in enemy territory (no flag), one
                       defender is right behind them.
    "tag_recovery"   – One attacker is freshly tagged and walking home; their
                       teammate (if any) is already in enemy territory.
    "last_chance"    – Scores are asymmetric (one team is one point from winning);
                       attacker has the flag.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from pyquaticus.structs import Team
import pyquaticus


# ──────────────────────────────────────────────────────────────────────────────
# Internal geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scrimmage_sign(env) -> dict:
    """
    Returns the cross-product sign that tells whether a position is on a given
    team's own side of the scrimmage line. Derived purely from flag positions so
    it works without accessing private env attributes.
    """
    scrim_start = env.scrimmage_coords[0]
    scrim_vec   = env.scrimmage_coords[1] - scrim_start

    signs = {}
    for team in Team:
        flag_home = env.flags[int(team)].home
        cp = np.cross(scrim_vec, flag_home - scrim_start)
        signs[team] = np.sign(cp)

    return signs


def _is_on_side(pos: np.ndarray, team: Team, scrim_start, scrim_vec, signs) -> bool:
    cp = np.cross(scrim_vec, np.asarray(pos) - scrim_start)
    return (np.sign(cp) == signs[team]) or (np.sign(cp) == 0)


def _random_pos_on_side(
    env,
    team: Team,
    scrim_start,
    scrim_vec,
    signs,
    existing_positions: list[np.ndarray],
    *,
    own_side: bool = True,
    flag_keepout: bool = True,
    max_tries: int = 2000,
) -> np.ndarray:
    """
    Sample a valid random position on the requested half of the field.

    Parameters
    ──────────
    own_side     : True  → sample on `team`'s own half
                   False → sample on the opponent's half
    flag_keepout : Avoid the flag keepout exclusion zones of all flags.
    """
    target_team   = team if own_side else (Team.RED_TEAM if team == Team.BLUE_TEAM else Team.BLUE_TEAM)
    env_size      = env.env_size
    flag_keepout_r = env.flag_keepout_radius
    min_r         = np.max(env.agent_radius)

    for _ in range(max_tries):
        # Sample uniformly inside the field
        pos = np.array([
            np.random.uniform(min_r, env_size[0] - min_r),
            np.random.uniform(min_r, env_size[1] - min_r),
        ])

        # Side check
        if not _is_on_side(pos, target_team, scrim_start, scrim_vec, signs):
            continue

        # Flag keepout
        if flag_keepout:
            too_close = False
            for flag in env.flags:
                if np.linalg.norm(pos - flag.home) < flag_keepout_r:
                    too_close = True
                    break
            if too_close:
                continue

        # Agent overlap check
        ok = True
        for other in existing_positions:
            if np.linalg.norm(pos - other) < 2 * min_r + 0.05:
                ok = False
                break
            
    
        
        if ok:
            return pos

    raise RuntimeError(
        f"Could not sample a valid position after {max_tries} tries "
        f"(team={team}, own_side={own_side}). "
        "Try increasing max_tries or reducing the number of agents."
    )


def _random_heading() -> float:
    return float(np.random.uniform(-180, 180))


def _toward_heading(src: np.ndarray, dst: np.ndarray) -> float:
    """Magnetic heading (CW from North) from src toward dst, ± small noise."""
    delta = dst - src
    angle = float(np.degrees(np.arctan2(delta[0], delta[1])))   # north = +y
    noise = float(np.random.uniform(-20, 20))
    return float((angle + noise + 360) % 360 - 180)


def _random_speed(env, agent_idx: int, fraction_range=(0.5, 1.0)) -> float:
    lo, hi = fraction_range
    return float(env.max_speeds[agent_idx] * np.random.uniform(lo, hi))


# ──────────────────────────────────────────────────────────────────────────────
# Scenario builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_scattered(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """All agents at independent random valid positions across the full field."""
    positions, headings, speeds = {}, {}, {}
    placed = []

    for agent_id in agents:
        idx  = agents.index(agent_id)
        team = Team.BLUE_TEAM if idx in blue_inds else Team.RED_TEAM
        own  = np.random.rand() < 0.5   # random side
        pos  = _random_pos_on_side(env, team, scrim_start, scrim_vec, signs,
                                   placed, own_side=own)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _random_heading()
        speeds[agent_id]    = _random_speed(env, idx)

    return dict(agent_position=positions, agent_heading=headings, agent_speed=speeds)


def _build_on_sides(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """All agents start on their own side (randomised positions)."""
    positions, headings, speeds = {}, {}, {}
    placed = []

    for agent_id in agents:
        idx  = agents.index(agent_id)
        team = Team.BLUE_TEAM if idx in blue_inds else Team.RED_TEAM
        pos  = _random_pos_on_side(env, team, scrim_start, scrim_vec, signs,
                                   placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _random_heading()
        speeds[agent_id]    = _random_speed(env, idx)

    return dict(agent_position=positions, agent_heading=headings, agent_speed=speeds)


def _build_flag_run(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    One blue attacker (agent_0) is carrying the red flag and heading home.
    Red defenders are chasing. If team_size > 1, extra blue agents are on their
    own side; extra red agents are positioned between the carrier and the blue flag home.
    """
    positions, headings, speeds = {}, {}, {}
    has_flag_dict = {a: False for a in agents}
    placed = []

    blue_home = env.flags[int(Team.BLUE_TEAM)].home
    red_home  = env.flags[int(Team.RED_TEAM)].home

    # Pick the flag carrier: first blue agent
    carrier_id  = agents[blue_inds[0]]
    carrier_idx = blue_inds[0]

    # Carrier is on the red side (enemy territory) with the flag
    carrier_pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                       placed, own_side=False)
    placed.append(carrier_pos)
    positions[carrier_id]    = carrier_pos
    headings[carrier_id]     = _toward_heading(carrier_pos, blue_home)
    speeds[carrier_id]       = _random_speed(env, carrier_idx, (0.6, 1.0))
    has_flag_dict[carrier_id] = True

    # Extra blue agents: on their own side
    for idx in blue_inds[1:]:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _toward_heading(pos, red_home)
        speeds[agent_id]    = _random_speed(env, idx)

    # Red defenders: placed between the carrier and the blue home (chasing)
    for idx in red_inds:
        agent_id = agents[idx]
        # Bias toward the carrier's area on red's own side
        alpha = np.random.uniform(0.1, 0.6)
        chase_target = alpha * carrier_pos + (1 - alpha) * blue_home
        # Clamp inside the field
        chase_target = np.clip(chase_target, env.agent_radius[idx],
                               env.env_size - env.agent_radius[idx])

        # Try exact target first, fall back to random on red side
        if _is_on_side(chase_target, Team.RED_TEAM, scrim_start, scrim_vec, signs):
            candidate = chase_target
            ok = all(np.linalg.norm(candidate - p) > 2 * np.max(env.agent_radius) + 0.05
                     for p in placed)
        else:
            ok = False

        if not ok:
            candidate = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                            placed, own_side=True)

        placed.append(candidate)
        positions[agent_id] = candidate
        headings[agent_id]  = _toward_heading(candidate, carrier_pos)
        speeds[agent_id]    = _random_speed(env, idx, (0.5, 1.0))

    has_flag_arr = [has_flag_dict[a] for a in agents]
    return dict(agent_position=positions, agent_heading=headings,
                agent_speed=speeds, agent_has_flag=has_flag_arr)


def _build_flag_grab(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    Blue attacker (agent_0) is right next to the red flag — about to grab it.
    A red defender is nearby but outside tag range.
    """
    positions, headings, speeds = {}, {}, {}
    placed = []

    red_home  = env.flags[int(Team.RED_TEAM)].home
    blue_home = env.flags[int(Team.BLUE_TEAM)].home
    r_agent   = np.max(env.agent_radius)

    # Attacker: within catch_radius of the red flag, on red's side
    angle    = np.random.uniform(0, 2 * np.pi)
    dist     = np.random.uniform(env.flag_keepout_radius + r_agent + 0.05,
                                  env.catch_radius * 0.9)
    attacker_pos = red_home + dist * np.array([np.cos(angle), np.sin(angle)])
    attacker_pos = np.clip(attacker_pos, r_agent, env.env_size - r_agent)

    attacker_id  = agents[blue_inds[0]]
    attacker_idx = blue_inds[0]
    placed.append(attacker_pos)
    positions[attacker_id] = attacker_pos
    headings[attacker_id]  = _toward_heading(attacker_pos, red_home)
    speeds[attacker_id]    = _random_speed(env, attacker_idx, (0.3, 0.8))

    # Extra blue agents: on their own side
    for idx in blue_inds[1:]:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _random_heading()
        speeds[agent_id]    = _random_speed(env, idx)

    # Red defender: close to the flag but outside tag range
    for idx in red_inds:
        agent_id = agents[idx]
        def_dist = np.random.uniform(env.catch_radius * 1.5, env.catch_radius * 4.0)
        def_angle = np.random.uniform(0, 2 * np.pi)
        def_pos   = red_home + def_dist * np.array([np.cos(def_angle), np.sin(def_angle)])
        def_pos   = np.clip(def_pos, r_agent, env.env_size - r_agent)

        ok = _is_on_side(def_pos, Team.RED_TEAM, scrim_start, scrim_vec, signs) and \
             all(np.linalg.norm(def_pos - p) > 2 * r_agent + 0.05 for p in placed)
        if not ok:
            def_pos = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                          placed, own_side=True)

        placed.append(def_pos)
        positions[agent_id] = def_pos
        headings[agent_id]  = _toward_heading(def_pos, attacker_pos)
        speeds[agent_id]    = _random_speed(env, idx, (0.5, 1.0))

    return dict(agent_position=positions, agent_heading=headings, agent_speed=speeds)


def _build_interception(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    Blue carrier has the red flag. A red defender is placed on the direct path
    between the carrier and the blue home — classic interception geometry.
    """
    positions, headings, speeds = {}, {}, {}
    has_flag_dict = {a: False for a in agents}
    placed = []

    blue_home = env.flags[int(Team.BLUE_TEAM)].home

    # Carrier on red's side
    carrier_id  = agents[blue_inds[0]]
    carrier_idx = blue_inds[0]
    carrier_pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                       placed, own_side=False)
    placed.append(carrier_pos)
    positions[carrier_id]    = carrier_pos
    headings[carrier_id]     = _toward_heading(carrier_pos, blue_home)
    speeds[carrier_id]       = _random_speed(env, carrier_idx, (0.5, 1.0))
    has_flag_dict[carrier_id] = True

    # Extra blue agents
    for idx in blue_inds[1:]:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _random_heading()
        speeds[agent_id]    = _random_speed(env, idx)

    # Interceptors: place between carrier and blue home on the path
    for i, idx in enumerate(red_inds):
        agent_id = agents[idx]
        # Alpha controls how far along the path the defender is placed
        alpha     = np.random.uniform(0.25, 0.65)
        intercept = (1 - alpha) * carrier_pos + alpha * blue_home
        intercept = np.clip(intercept, env.agent_radius[idx],
                            env.env_size - env.agent_radius[idx])

        ok = all(np.linalg.norm(intercept - p) > 2 * np.max(env.agent_radius) + 0.05
                 for p in placed)
        if not ok:
            intercept = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                             placed, own_side=False)

        placed.append(intercept)
        positions[agent_id] = intercept
        headings[agent_id]  = _toward_heading(intercept, carrier_pos)
        speeds[agent_id]    = _random_speed(env, idx, (0.5, 1.0))

    has_flag_arr = [has_flag_dict[a] for a in agents]
    return dict(agent_position=positions, agent_heading=headings,
                agent_speed=speeds, agent_has_flag=has_flag_arr)


def _build_standoff(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    All agents clustered within a few catch_radii of the scrimmage line, facing across.
    Simulates the tense mid-field stand-off at game start / after reset.
    """
    positions, headings, speeds = {}, {}, {}
    placed = []

    scrim_mid = (env.scrimmage_coords[0] + env.scrimmage_coords[1]) / 2
    scrim_vec_norm = (env.scrimmage_coords[1] - env.scrimmage_coords[0])
    scrim_vec_norm = scrim_vec_norm / (np.linalg.norm(scrim_vec_norm) + 1e-9)
    perp  = np.array([-scrim_vec_norm[1], scrim_vec_norm[0]])   # perpendicular to scrim

    def place_near_scrim(team, own_side):
        """Sample within 1–6 catch_radii of the scrimmage line on the correct side."""
        sign = 1.0 if own_side else -1.0
        # Determine which perp direction is "own side" for this team
        test = scrim_mid + 0.5 * env.catch_radius * perp
        if not _is_on_side(test, team, scrim_start, scrim_vec, signs):
            sign *= -1.0

        for _ in range(1000):
            along  = np.random.uniform(-0.45, 0.45) * np.linalg.norm(
                env.scrimmage_coords[1] - env.scrimmage_coords[0]
            )
            across = sign * np.random.uniform(0.5, 5.0) * env.catch_radius
            pos    = scrim_mid + along * scrim_vec_norm + across * perp
            pos    = np.clip(pos, np.max(env.agent_radius), env.env_size - np.max(env.agent_radius))
            if (
                _is_on_side(pos, team, scrim_start, scrim_vec, signs) == own_side
                and all(np.linalg.norm(pos - p) > 2 * np.max(env.agent_radius) + 0.05
                        for p in placed)
            ):
                return pos
        # Fallback
        return _random_pos_on_side(env, team, scrim_start, scrim_vec, signs,
                                   placed, own_side=own_side)

    blue_home = env.flags[int(Team.BLUE_TEAM)].home
    red_home  = env.flags[int(Team.RED_TEAM)].home

    for idx in blue_inds:
        agent_id = agents[idx]
        pos = place_near_scrim(Team.BLUE_TEAM, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _toward_heading(pos, red_home)
        speeds[agent_id]    = _random_speed(env, idx, (0.0, 0.5))

    for idx in red_inds:
        agent_id = agents[idx]
        pos = place_near_scrim(Team.RED_TEAM, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _toward_heading(pos, blue_home)
        speeds[agent_id]    = _random_speed(env, idx, (0.0, 0.5))

    return dict(agent_position=positions, agent_heading=headings, agent_speed=speeds)


def _build_multi_threat(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    Multiple blue attackers deep on the red side simultaneously — no flags grabbed yet.
    Red defenders scramble on their own side.
    """
    positions, headings, speeds = {}, {}, {}
    placed = []
    red_home = env.flags[int(Team.RED_TEAM)].home

    for idx in blue_inds:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=False)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _toward_heading(pos, red_home)
        speeds[agent_id]    = _random_speed(env, idx, (0.4, 1.0))

    for idx in red_inds:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        # Defender heads toward the nearest attacker
        attacker_positions = [positions[agents[i]] for i in blue_inds]
        nearest = min(attacker_positions, key=lambda p: np.linalg.norm(pos - p))
        headings[agent_id] = _toward_heading(pos, nearest)
        speeds[agent_id]   = _random_speed(env, idx, (0.5, 1.0))

    return dict(agent_position=positions, agent_heading=headings, agent_speed=speeds)


def _build_chase(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    Blue attacker deep in red territory (no flag), one red defender right behind them.
    """
    positions, headings, speeds = {}, {}, {}
    placed = []
    red_home = env.flags[int(Team.RED_TEAM)].home
    r_agent  = np.max(env.agent_radius)

    # Attacker: deep on red side
    attacker_id  = agents[blue_inds[0]]
    attacker_idx = blue_inds[0]
    attacker_pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                        placed, own_side=False)
    placed.append(attacker_pos)
    positions[attacker_id] = attacker_pos
    headings[attacker_id]  = _toward_heading(attacker_pos, red_home)
    speeds[attacker_id]    = _random_speed(env, attacker_idx, (0.6, 1.0))

    # Primary chaser: behind the attacker (between attacker and scrimmage line)
    chaser_id  = agents[red_inds[0]]
    chaser_idx = red_inds[0]
    chase_offset_dist = np.random.uniform(env.catch_radius * 2, env.catch_radius * 5)
    chase_dir   = attacker_pos - red_home
    chase_dir   = chase_dir / (np.linalg.norm(chase_dir) + 1e-9)
    chase_pos   = attacker_pos - chase_offset_dist * chase_dir
    chase_pos   = np.clip(chase_pos, r_agent, env.env_size - r_agent)

    ok = (
        _is_on_side(chase_pos, Team.RED_TEAM, scrim_start, scrim_vec, signs)
        and all(np.linalg.norm(chase_pos - p) > 2 * r_agent + 0.05 for p in placed)
    )
    if not ok:
        chase_pos = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                         placed, own_side=True)

    placed.append(chase_pos)
    positions[chaser_id] = chase_pos
    headings[chaser_id]  = _toward_heading(chase_pos, attacker_pos)
    speeds[chaser_id]    = _random_speed(env, chaser_idx, (0.5, 1.0))

    # Extra agents: random on their own side
    for idx in blue_inds[1:]:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _random_heading()
        speeds[agent_id]    = _random_speed(env, idx)

    for idx in red_inds[1:]:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _random_heading()
        speeds[agent_id]    = _random_speed(env, idx)

    return dict(agent_position=positions, agent_heading=headings, agent_speed=speeds)


def _build_tag_recovery(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    One blue attacker is freshly tagged and walking home (is_tagged=True, full cooldown).
    Their teammate (if any) is already deep in enemy territory.
    Red defenders are on their own side.
    """
    positions, headings, speeds = {}, {}, {}
    is_tagged_dict   = {a: False for a in agents}
    cooldown_dict    = {a: env.tagging_cooldown for a in agents}
    placed = []

    blue_home = env.flags[int(Team.BLUE_TEAM)].home
    red_home  = env.flags[int(Team.RED_TEAM)].home

    # Tagged agent: on red side, heading home
    tagged_id  = agents[blue_inds[0]]
    tagged_idx = blue_inds[0]
    tagged_pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                      placed, own_side=False)
    placed.append(tagged_pos)
    positions[tagged_id]   = tagged_pos
    headings[tagged_id]    = _toward_heading(tagged_pos, blue_home)
    speeds[tagged_id]      = env.max_speeds[tagged_idx]   # running home at max speed
    is_tagged_dict[tagged_id] = True
    cooldown_dict[tagged_id]  = 0.0                       # cooldown depleted (just tagged)

    # Extra blue agents: in enemy territory
    for idx in blue_inds[1:]:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.BLUE_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=False)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _toward_heading(pos, red_home)
        speeds[agent_id]    = _random_speed(env, idx, (0.4, 1.0))

    # Red defenders: on own side
    for idx in red_inds:
        agent_id = agents[idx]
        pos = _random_pos_on_side(env, Team.RED_TEAM, scrim_start, scrim_vec, signs,
                                  placed, own_side=True)
        placed.append(pos)
        positions[agent_id] = pos
        headings[agent_id]  = _toward_heading(pos, tagged_pos)
        speeds[agent_id]    = _random_speed(env, idx, (0.3, 0.9))

    is_tagged_arr = [is_tagged_dict[a] for a in agents]
    cooldown_arr  = [cooldown_dict[a]  for a in agents]
    return dict(
        agent_position=positions,
        agent_heading=headings,
        agent_speed=speeds,
        agent_is_tagged=is_tagged_arr,
        agent_tagging_cooldown=cooldown_arr,
    )


def _build_last_chance(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs) -> dict:
    """
    One team is one capture away from winning. Blue carrier has the flag.
    Red defenders know the urgency — they're racing back.
    Score state is embedded via captures in the init_dict.
    """
    base = _build_flag_run(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs)

    # One team is max_score - 1
    blue_score = env.max_score - 1
    red_score  = int(np.random.choice([0, max(0, env.max_score - 2)]))
    base["captures"] = [blue_score, red_score]

    return base


# ──────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ──────────────────────────────────────────────────────────────────────────────

_SCENARIOS = {
    "scattered":     _build_scattered,
    "on_sides":      _build_on_sides,
    "flag_run":      _build_flag_run,
    "flag_grab":     _build_flag_grab,
    "interception":  _build_interception,
    "standoff":      _build_standoff,
    "multi_threat":  _build_multi_threat,
    "chase":         _build_chase,
    "tag_recovery":  _build_tag_recovery,
    "last_chance":   _build_last_chance,
}

# Sampling weights: rarer / harder scenarios get a small boost so the agent
# sees them during training. Adjust to taste.
_WEIGHTS = {
    "scattered":     1.0,
    "on_sides":      1.0,
    "flag_run":      1.5,
    "flag_grab":     1.5,
    "interception":  1.2,
    "standoff":      1.0,
    "multi_threat":  1.2,
    "chase":         1.2,
    "tag_recovery":  0.8,
    "last_chance":   0.8,
}


# ──────────────────────────────────────────────────────────────────────────────
# init_dict normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _dicts_to_arrays(init_dict: dict, agents: list) -> dict:
    """
    Convert any per-agent dict values in init_dict into ordered arrays.

    PyQuaticus._set_state_from_init_dict has a bug on the dict path:
        val = init_dict.get(state_var).get(agent_id, None)
        if val == None:          # crashes when val is a numpy array
    Passing arrays forces the safe isinstance(...) branch that uses plain
    integer indexing and never does an equality check against None.

    Keys converted (if present and currently a dict):
        agent_position, agent_speed, agent_heading   → list of values in agents order
        agent_has_flag, agent_is_tagged,
        agent_tagging_cooldown, agent_collisions     → list of values in agents order

    All other keys (captures, tags, grabs, …) are left untouched.
    """
    per_agent_keys = (
        "agent_position",
        "agent_speed",
        "agent_heading",
        "agent_has_flag",
        "agent_is_tagged",
        "agent_tagging_cooldown",
        "agent_collisions",
    )

    out = dict(init_dict)   # shallow copy — don't mutate the builder's dict

    for key in per_agent_keys:
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, dict):
            # Re-order to match env.agents; missing agents get a sensible default
            ordered = [val[agent_id] for agent_id in agents]
            out[key] = ordered

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def random_reset(
    env,
    scenario: Optional[str] = None,
    seed: Optional[int] = None,
    normalize_obs: Optional[bool] = None,
    normalize_state: Optional[bool] = None,
) -> tuple:
    """
    Reset the environment with a randomly generated scenario.

    Parameters
    ──────────
    env            : A fully initialised PyQuaticusEnv instance (already reset
                     at least once so geometry is set up).
    scenario       : One of the keys in _SCENARIOS, or None to pick randomly.
    seed           : Optional random seed for reproducibility.
    normalize_obs  : Passed through to env.reset(). Defaults to env's current setting.
    normalize_state: Passed through to env.reset(). Defaults to env's current setting.

    Returns
    ───────
    obs, info — same as env.reset().

    Example
    ───────
        from scenario_init import random_reset
        obs, info = random_reset(env)
        obs, info = random_reset(env, scenario="flag_run", seed=0)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # ── Validate / pick scenario ──────────────────────────────────────────────
    if scenario is not None:
        if scenario not in _SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Choose from: {sorted(_SCENARIOS.keys())}"
            )
        chosen = scenario
    else:
        keys    = list(_WEIGHTS.keys())
        weights = np.array([_WEIGHTS[k] for k in keys], dtype=float)
        weights /= weights.sum()
        chosen  = np.random.choice(keys, p=weights)

    # ── Extract structural info from env ──────────────────────────────────────
    agents     = env.agents
    blue_inds  = env.agent_inds_of_team[Team.BLUE_TEAM].tolist()
    red_inds   = env.agent_inds_of_team[Team.RED_TEAM].tolist()

    scrim_start = env.scrimmage_coords[0]
    scrim_vec   = env.scrimmage_coords[1] - scrim_start
    signs       = _scrimmage_sign(env)

    # ── Build init_dict ───────────────────────────────────────────────────────
    builder  = _SCENARIOS[chosen]
    init_dict = builder(env, agents, blue_inds, red_inds, scrim_start, scrim_vec, signs)

    # ── Convert dict values → ordered arrays so the env takes the safe branch ─
    # PyQuaticus's _set_state_from_init_dict does `if val == None` on the dict
    # path, which raises a ValueError when val is a numpy array. Passing arrays
    # (in env.agents order) forces the isinstance(list/tuple/ndarray) branch
    # that uses simple indexing instead.
    init_dict = _dicts_to_arrays(init_dict, agents)

    # ── Call env.reset ────────────────────────────────────────────────────────
    options = {"init_dict": init_dict}
    if normalize_obs is not None:
        options["normalize_obs"] = normalize_obs
    if normalize_state is not None:
        options["normalize_state"] = normalize_state
        

    try:    
        obs, info = env.reset(options=options)
    except Exception as e:
        return random_reset(env, scenario=scenario, seed=seed+1 if seed is not None else None,
                            normalize_obs=normalize_obs, normalize_state=normalize_state)


    return obs, info


def list_scenarios() -> list[str]:
    """Return all available scenario names."""
    return sorted(_SCENARIOS.keys())