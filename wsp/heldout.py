"""
Held-out scoring: original vs edited weights on seeds and opponents the
editor never saw. Same seeds for both policies (paired), bootstrap CIs.

    python -m wsp.heldout --task carry_home --edited runs/<run>/edited.pt --episodes 40
"""

import argparse
import json
from pathlib import Path

import numpy as np

from wsp.matches import MatchRunner, summarize_metrics
from wsp.sim import MatchConfig, load_actors, weight_diff
from wsp.tasks import TASKS

HELDOUT_SEED = 900_000


def _boot_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(mo: list[dict], me: list[dict]) -> dict:
    sd = np.array([b["blue_score"] - b["red_score"] - (a["blue_score"] - a["red_score"]) for a, b in zip(mo, me)], float)
    cap = np.array([b["blue_score"] - a["blue_score"] for a, b in zip(mo, me)], float)
    pts = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    wr = np.array([pts[b["outcome"]] - pts[a["outcome"]] for a, b in zip(mo, me)], float)
    return {"score_diff_delta": float(sd.mean()), "score_diff_delta_ci95": _boot_ci(sd),
            "captures_delta": float(cap.mean()), "captures_delta_ci95": _boot_ci(cap),
            "points_delta": float(wr.mean()), "points_delta_ci95": _boot_ci(wr)}


def opponent_label(spec: str) -> str:
    if spec.startswith("ckpt:"):
        return f"checkpoint {Path(spec[5:]).stem} (as red)"
    return spec


def compare(task, original, edited, runner: MatchRunner, episodes: int) -> dict:
    seeds = list(range(HELDOUT_SEED, HELDOUT_SEED + episodes))
    out = {}
    for opp in task.heldout_opponents:
        mo = runner.run(original, opp, seeds)
        me = runner.run(edited, opp, seeds)
        out[opp] = {"original": summarize_metrics(mo), "edited": summarize_metrics(me), "paired": paired(mo, me),
                    "episodes_original": mo, "episodes_edited": me}
        print(f"  {opponent_label(opp)}: score diff {out[opp]['original']['mean_score_diff']:+.2f} -> "
              f"{out[opp]['edited']['mean_score_diff']:+.2f}", flush=True)
    return out


ROWS = [  # (label, key, format)
    ("W / D / L", None, None),
    ("mean score (blue : red)", None, None),
    ("score difference", "mean_score_diff", "{:+.2f}"),
    ("flag grabs / match", "flag_grabs", "{:.2f}"),
    ("captures per grab", "captures_per_grab", "{:.2f}"),
    ("carrying steps / match", "carry_steps", "{:.1f}"),
    ("speed toward own half while carrying (m/s)", "carry_homeward_speed_mps", "{:.2f}"),
    ("stop action while carrying", "carry_stop_rate", "{:.0%}"),
    ("times tagged / match", "times_tagged", "{:.2f}"),
    ("out-of-bounds / match", "oob_events", "{:.2f}"),
    ("tags made / match", "tags_made", "{:.2f}"),
    ("time on enemy half", "enemy_side_rate", "{:.0%}"),
    ("loiter rate", "loiter_rate", "{:.1%}"),
    ("spin rate", "spin_rate", "{:.1%}"),
    ("dead-stop rate", "dead_stop_rate", "{:.1%}"),
]


def _cell(s, key, fmt):
    v = s.get(key)
    return "n/a" if v is None else fmt.format(v)


def report_markdown(task, results: dict, diff: dict, run_info: dict | None, episodes: int) -> str:
    L = [f"# Weight-space programming report: `{task.name}`", ""]
    if run_info:
        L += ["## Run", "",
              f"- Editor: `{run_info.get('model')}` (reasoning effort `{run_info.get('effort')}`, protocol `{run_info.get('protocol')}`)",
              f"- Turns: {run_info.get('turns')}, wall time {run_info.get('minutes')} min, end: {run_info.get('end_reason')}",
              f"- Budgets used: {run_info.get('python_runs_used')} python runs, {run_info.get('matches_used')} practice matches",
              f"- Tokens: {run_info.get('usage')}",
              f"- Audit flags (patterns that could mean fitting instead of writing): {run_info.get('audit_flags') or 'none'}",
              ""]
    L += ["## Weights changed", "",
          f"{diff['changed']:,} of {diff['total']:,} weights ({diff['pct']:.3f}%).", "",
          "| tensor | changed | of | max abs delta |", "|---|---:|---:|---:|"]
    for k, v in diff["per_tensor"].items():
        if v["changed"]:
            L.append(f"| {k} | {v['changed']:,} | {v['size']:,} | {v['max_abs_delta']:.4g} |")
    L += ["", f"## Held-out matches ({episodes} per opponent, seeds {HELDOUT_SEED}+, same seeds for both policies)", ""]
    for opp, r in results.items():
        o, e, p = r["original"], r["edited"], r["paired"]
        L += [f"### vs {opponent_label(opp)}", "", "| metric | original | edited |", "|---|---:|---:|",
              f"| W / D / L | {o['wins']} / {o['draws']} / {o['losses']} | {e['wins']} / {e['draws']} / {e['losses']} |",
              f"| mean score (blue : red) | {o['mean_blue_score']:.2f} : {o['mean_red_score']:.2f} | "
              f"{e['mean_blue_score']:.2f} : {e['mean_red_score']:.2f} |"]
        for label, key, fmt in ROWS[2:]:
            L.append(f"| {label} | {_cell(o, key, fmt)} | {_cell(e, key, fmt)} |")
        lo, hi = p["score_diff_delta_ci95"]
        clo, chi = p["captures_delta_ci95"]
        L += ["", f"Paired change in score difference: {p['score_diff_delta']:+.2f} (95% bootstrap CI {lo:+.2f} to {hi:+.2f}); "
                  f"captures per match: {p['captures_delta']:+.2f} ({clo:+.2f} to {chi:+.2f}).", ""]
    if run_info and run_info.get("report"):
        L += ["## Editor's own report", "", run_info["report"], ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="carry_home", choices=sorted(TASKS))
    ap.add_argument("--edited", required=True, help="edited checkpoint (.pt)")
    ap.add_argument("--original", help="defaults to the task's checkpoint")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", help="output directory (default: next to --edited)")
    args = ap.parse_args()

    task = TASKS[args.task]
    original = load_actors(args.original or task.ckpt)
    edited = load_actors(args.edited)
    out = Path(args.out or Path(args.edited).parent)
    out.mkdir(parents=True, exist_ok=True)
    runner = MatchRunner(MatchConfig(team_size=original.team_size), args.workers)
    try:
        results = compare(task, original, edited, runner, args.episodes)
    finally:
        runner.close()
    diff = weight_diff(original, edited)
    run_info = None
    if (out / "run.json").exists():
        run_info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    (out / "heldout.json").write_text(json.dumps({"results": results, "weight_diff": diff}, indent=1, default=str))
    (out / "report.md").write_text(report_markdown(task, results, diff, run_info, args.episodes), encoding="utf-8")
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
