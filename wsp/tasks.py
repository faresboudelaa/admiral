"""
Task presets for weight-space programming on PyQuaticus policies, and the
prompt the editor model receives.

carry_home   (default) Repair the trained 2v2 self-play policy: it grabs the
             enemy flag but never brings it home. Closest to the Kinetix
             "stuck on the ramp" repair in the WSP blog post.
from_scratch Write a working policy into an untrained network (the file named
             "mappo reward hacking.pt" is the actor at initialisation:
             update_count 0, all biases exactly 0, output layer at init scale).
"""

from dataclasses import dataclass
from pathlib import Path

from wsp.sim import ROOT, action_table

LEGACY_APM = ROOT.parent / "APM_5DA01_TP-main_04-pyquaticus"


@dataclass
class Task:
    name: str
    ckpt: Path
    practice_opponent: str
    heldout_opponents: list
    goal: str
    symptom_intro: str


TASKS = {
    "carry_home": Task(
        name="carry_home",
        ckpt=ROOT / "checkpoints" / "legacy" / "mappo_2v2_selfplay_final.pt",
        practice_opponent="heuristic:hard",
        heldout_opponents=["heuristic:hard", "heuristic:medium",
                           "ckpt:" + str(ROOT / "checkpoints" / "legacy" / "mappo_2v2_selfplay_final.pt")],
        symptom_intro=(
            "The policy was trained with MAPPO self-play for 1,500 updates, with a shaped reward "
            "(agent_0 was rewarded as an attacker, agent_1 as a defender). It reaches and grabs the "
            "enemy flag regularly, but it does not bring it back to its own half."),
        goal=(
            "Make the blue team bring the enemy flag home (score captures) in held-out matches, "
            "without breaking what already works (reaching and grabbing the flag, defending, "
            "staying in bounds). Prefer a small, targeted edit that fixes the cause."),
    ),
    "from_scratch": Task(
        name="from_scratch",
        ckpt=LEGACY_APM / "mappo reward hacking.pt",
        practice_opponent="heuristic:hard",
        heldout_opponents=["heuristic:hard", "heuristic:medium", "random"],
        symptom_intro=(
            "This network has never been trained: it is the actor at initialisation (orthogonal "
            "weights, all biases 0, output layer scaled by 0.01), so its logits are almost flat and "
            "it acts nearly uniformly at random. Both blue agents use this one shared actor "
            "(W has a single entry)."),
        goal=(
            "Write a policy into these weights that plays capture-the-flag well: grab the enemy "
            "flag, bring it home, avoid being tagged, stay in bounds and defend. You may overwrite "
            "any or all weights."),
    ),
}


TRACE_DOC = """\
load_trace(name) returns a dict of numpy arrays, indexed [step, blue_agent, ...]:
  obs      (T, n, D) float32  observation each blue agent saw when choosing its action
  action   (T, n)    int      action chosen (sampled from softmax(logits)); -1 if the agent had no obs
  logits   (T, n, 17)         logits the policy produced for that observation
  pos      (T, n, 2) metres   position after the step (x along the 160 m axis, y along the 80 m axis)
  heading  (T, n)    degrees  heading after the step (nautical: 0 = +y, 90 = +x, clockwise)
  speed    (T, n)    m/s      speed after the step (max 2.0)
  on_side, has_flag, is_tagged, oob   (T, n) bool, state after the step
  loiter, spin, dead_stop             (T, n) bool, behaviour detectors (same as the training reward)
Blue agent index i in these arrays is W key "agent_{i}" (with a shared policy both use the one actor)."""


def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def baseline_lines(s: dict) -> str:
    return "\n".join([
        f"- matches: {s['n']}  (W/D/L {s['wins']}/{s['draws']}/{s['losses']}), "
        f"mean score {s['mean_blue_score']:.2f} : {s['mean_red_score']:.2f}",
        f"- flag grabs per match (team): {s['flag_grabs']:.2f};  captures per grab: {_fmt(s['captures_per_grab'])}",
        f"- while carrying the flag: {s['carry_steps']:.1f} steps per match; speed toward own half "
        f"{_fmt(s['carry_homeward_speed_mps'])} m/s (max boat speed 2.0); stop action chosen "
        f"{_fmt(None if s['carry_stop_rate'] is None else 100 * s['carry_stop_rate'], 0)}% of carrying steps",
        f"- times tagged per match (team): {s['times_tagged']:.2f};  out-of-bounds per match: {s['oob_events']:.2f};  "
        f"tags made: {s['tags_made']:.2f}",
        f"- fraction of time on the enemy half: {s['enemy_side_rate']:.2f};  mean speed {s['mean_speed']:.2f} m/s",
        f"- action frequencies (index: share): "
        + ", ".join(f"{i}: {p:.2f}" for i, p in enumerate(s["action_hist"])),
    ])


def build_prompt(task: Task, *, actors, spec, geometry: dict, baseline: dict, baseline_traces: list,
                 python_budget: int, episode_budget: int) -> str:
    n_agents = len(actors.nets)
    keys = ", ".join(f'W["{k}"]' for k in sorted(actors.nets))
    D = actors.obs_dim
    obs_rows = "\n".join(
        f"  {i:2d}  {name:<34s} raw [{lo:g}, {hi:g}]" for i, (name, lo, hi) in enumerate(spec))
    act_rows = "\n".join("  " + r for r in action_table())
    per_actor = actors.n_weights() // n_agents
    return f"""\
# Weight-space programming: edit a capture-the-flag policy directly in its weights

You will change the behaviour of a neural-network policy by reading and editing its weights.
You may not train it: no gradient descent, no optimiser, no fitting weights to data by regression
or least squares. Every change must come from your own analysis of the weights and of the policy's
behaviour, written into the weight arrays with numpy.

## The game
PyQuaticus (MIT Lincoln Laboratory) is maritime capture-the-flag for autonomous boats.
- Field: {geometry['env_size'][0]:g} m x {geometry['env_size'][1]:g} m. The scrimmage line is x = {geometry['scrimmage_x']:g}.
  Blue (the team you control) defends its flag at {geometry['blue_home']} on the x < {geometry['scrimmage_x']:g} half;
  red's flag is at {geometry['red_home']} on the other half.
- Teams of {actors.team_size} boats. Max speed 2.0 m/s. One decision every {geometry['step_seconds']:g} s of match time.
- Grab: get within 10 m of the enemy flag. Capture (1 point): carry it back across the scrimmage line onto
  your own half (x < {geometry['scrimmage_x']:g} for blue).
- Tag: a boat on its own half can tag an opponent within 10 m (then a 30 s cooldown). A tagged boat drops
  the flag (it returns to its home) and is driven back to its own flag automatically before it can play
  again. Leaving the field counts as being tagged.
- A match lasts {geometry['max_time']:g} s ({geometry['max_steps']} decisions) or until a team has {geometry['max_score']} points.
- The practice opponent is a scripted team ("{task.practice_opponent}"). Held-out scoring also uses other opponents.

## The policy
{n_agents} actor network(s): {keys}. {"Each blue boat has its own actor (agent_i drives blue boat i)." if n_agents > 1 else "Both blue boats use this one shared actor."}
Architecture of each actor (float32):
  h1     = tanh(W1 @ obs + b1)     W1: (256, {D}),  b1: (256,)
  h2     = tanh(W2 @ h1  + b2)     W2: (256, 256),  b2: (256,)
  logits = W3 @ h2 + b3            W3: (17, 256),   b3: (17,)
  action ~ softmax(logits)          (sampled during matches)
{per_actor:,} weights per actor, {actors.n_weights():,} in total. The shapes must stay as they are.

## Observation ({D} values per boat)
Each value is mapped linearly from its raw range onto [-1, 1] (raw low -> -1, raw high -> +1; booleans: false -1, true +1).
Headings are nautical (0 deg = +y, 90 deg = +x, clockwise). Bearings are degrees relative to the boat's own
heading, in [-180, 180] (0 = straight ahead, positive = clockwise).
"teammate_*" and "opponent_*" entries describe the other boats from this boat's point of view.
{obs_rows}

## Actions (17)
Each action sets a speed and a heading change relative to the current heading. A target at bearing b
is steered towards by the action whose heading change is closest to b.
{act_rows}

## What is wrong
{task.symptom_intro}
Measured by the harness before you start, on {baseline['n']} practice matches against the practice opponent
(recorded; load them with load_trace(name) for name in {baseline_traces}):
{baseline_lines(baseline)}

## Goal
{task.goal}
At the end the harness scores the weights in W on held-out matches you never see (new start positions,
the practice opponent and opponents you have not played), comparing them with the original weights:
score difference, captures, flag carrying, tags, out-of-bounds, and the fraction of weights changed.

## Tools
- python(code): run Python in a persistent kernel (numpy available; the simulator, deep-learning
  frameworks and network access are not). Preloaded:
    W            the weights you are editing: W["agent_0"]["W1"], ... (numpy float32). Edit W in place or reassign it.
    W0           the original weights (read-only)
    forward(net, obs, hidden=False) -> logits, or (logits, h1, h2);  probs(net, obs) -> action probabilities
    OBS_NAMES, ACTIONS   the tables above as lists
    list_traces(), load_trace(name), TRACE_DOC      recorded matches (see TRACE_DOC)
    changed_fraction()   fraction of weights in W that differ from W0
    snapshot(name), restore(name)   keep / go back to a version of W (restore("original") resets)
  Output is truncated to about 20,000 characters. Each run has a {geometry['python_timeout']:.0f} s limit.
  Budget: {python_budget} runs.
- evaluate(episodes, record): play practice matches with the current W against the practice opponent,
  each from a new random start, and get the same statistics as above. With record=true the matches are
  saved as traces you can load. Budget: {episode_budget} matches in total.
- submit(report): finish. The weights in W at that moment are final. In the report, explain the cause you
  found, the edit you made and why it should work, and give changed_fraction().

When you are done, call submit.
"""

