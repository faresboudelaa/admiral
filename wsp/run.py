"""
Weight-space programming with Muse Spark: the editor model reads a trained
policy's weights and edits them directly (no gradients), then the harness
scores the result on held-out matches.

    python -m wsp.run                                  # carry_home task, muse-spark-1.3, effort high
    python -m wsp.run --effort max --match-budget 50
    python -m wsp.run --task from_scratch
    python -m wsp.run --scripted-test --heldout 4      # harness check, no API key needed

Needs MODEL_API_KEY in admiral/.env (or the environment) except with --scripted-test.
"""

import argparse
import json
import os
import time
from pathlib import Path

from wsp.editor import ChatClient, ResponsesClient, ScriptedClient, Session, run_editor
from wsp.heldout import compare, report_markdown
from wsp.matches import MatchRunner, summarize_metrics
from wsp.sandbox import Sandbox, docker_ready
from wsp.sim import ROOT, MatchConfig, action_table, load_actors, make_env, obs_spec, save_actors, weight_diff
from wsp.tasks import TASKS, TRACE_DOC, build_prompt

BASELINE_SEED = 100_000
N_BASELINE = 8

SCRIPTED_TEST = [
    ("python", {"code": "print(AGENTS, {t: x.shape for t, x in W['agent_0'].items()})\n"
                        "tr = load_trace('baseline_00'); print(tr['has_flag'].sum(0), np.bincount(tr['action'][:, 0], minlength=17))"}),
    ("evaluate", {"episodes": 2, "record": True}),
    ("python", {"code": "W['agent_0']['b3'][16] -= 2.0  # make the stop action less likely\nprint(changed_fraction()[0])"}),
    ("evaluate", {"episodes": 2, "record": False}),
    ("submit", {"report": "Scripted harness test: lowered the stop-action bias of agent_0 by 2.0. Not a real edit."}),
]


def load_env_file():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="carry_home", choices=sorted(TASKS))
    ap.add_argument("--model", default="muse-spark-1.3")
    ap.add_argument("--effort", default="high", help="minimal|low|medium|high|xhigh|max")
    ap.add_argument("--protocol", default="responses", choices=["responses", "chat"])
    ap.add_argument("--max-output-tokens", type=int)
    ap.add_argument("--python-budget", type=int, default=100)
    ap.add_argument("--match-budget", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, default=240)
    ap.add_argument("--heldout", type=int, default=40, help="held-out matches per opponent (0 to skip)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--scripted-test", action="store_true", help="replay a fixed tool script instead of calling a model")
    ap.add_argument("--sandbox", default="docker", choices=["docker", "local"],
                    help="where the editor's code runs: an isolated no-network container (default), or a local "
                         "child process (only for --scripted-test; local runs model code unisolated)")
    args = ap.parse_args()

    load_env_file()
    if args.sandbox == "local" and not args.scripted_test:
        raise SystemExit("--sandbox local would run the model's code unisolated on this machine; "
                         "it is only allowed with --scripted-test.")
    if args.sandbox == "docker" and (problem := docker_ready()):
        raise SystemExit(problem)
    task = TASKS[args.task]
    model_tag = "scripted" if args.scripted_test else args.model
    run_dir = ROOT / "runs" / f"{task.name}_{model_tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "sandbox" / "traces").mkdir(parents=True)

    if args.scripted_test:
        client = ScriptedClient(SCRIPTED_TEST)
    else:
        key = os.environ.get("MODEL_API_KEY")
        if not key:
            raise SystemExit("MODEL_API_KEY is not set. Put MODEL_API_KEY=... in admiral/.env.")
        base = os.environ.get("MODEL_API_BASE", "https://api.meta.ai/v1")
        cls = ResponsesClient if args.protocol == "responses" else ChatClient
        client = cls(args.model, args.effort, base, key, args.max_output_tokens)

    original = load_actors(task.ckpt)
    cfg = MatchConfig(team_size=original.team_size)
    env = make_env(cfg)
    spec = obs_spec(env)
    geometry = dict(env_size=[float(x) for x in env.env_size],
                    scrimmage_x=float(env.scrimmage_coords[0][0]),
                    blue_home=tuple(float(x) for x in env.state["flag_home"][0]),
                    red_home=tuple(float(x) for x in env.state["flag_home"][1]),
                    step_seconds=cfg.sim_speedup * 0.1, max_time=cfg.max_time,
                    max_steps=int(cfg.max_time / (cfg.sim_speedup * 0.1)), max_score=cfg.max_score,
                    python_timeout=180.0)
    env.close()

    print(f"Run directory: {run_dir}")
    print(f"Task {task.name}: {task.ckpt.name}, {original.n_weights():,} weights, team size {original.team_size}")
    runner = MatchRunner(cfg, args.workers)
    sandbox = None
    try:
        names = [f"baseline_{i:02d}" for i in range(N_BASELINE)]
        print(f"Recording {N_BASELINE} baseline matches ...", flush=True)
        base_ms = runner.run(original, task.practice_opponent, list(range(BASELINE_SEED, BASELINE_SEED + N_BASELINE)),
                             [run_dir / "sandbox" / "traces" / f"{n}.npz" for n in names])
        baseline = summarize_metrics(base_ms)

        sandbox = Sandbox(run_dir / "sandbox", original,
                          {"obs_names": [n for n, _, _ in spec], "actions": action_table(), "trace_doc": TRACE_DOC},
                          timeout=geometry["python_timeout"], isolation=args.sandbox)
        prompt = build_prompt(task, actors=original, spec=spec, geometry=geometry, baseline=baseline,
                              baseline_traces=names, python_budget=args.python_budget,
                              episode_budget=args.match_budget)
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        session = Session(run_dir, task, original, sandbox, runner, args.python_budget, args.match_budget,
                          practice_seed=BASELINE_SEED + N_BASELINE)
        print(f"Editing with {model_tag} ...", flush=True)
        info = run_editor(client, session, prompt, run_dir, args.max_turns, args.max_minutes)
        info.update(sandbox=args.sandbox, model=model_tag, effort=None if args.scripted_test else args.effort,
                    protocol="scripted" if args.scripted_test else args.protocol, task=task.name,
                    checkpoint=str(task.ckpt), baseline=baseline,
                    budgets={"python": args.python_budget, "matches": args.match_budget})
        edited = session.final
        save_actors(edited, task.ckpt, run_dir / "edited.pt", note=f"WSP edit by {model_tag}: {info['report']}")
        diff = weight_diff(original, edited)
        info["weight_diff"] = {k: v for k, v in diff.items() if k != "per_tensor"}
        (run_dir / "run.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        print(f"Editor finished ({info['end_reason']}): {info['turns']} turns, {info['minutes']} min, "
              f"{diff['pct']:.3f}% of weights changed", flush=True)

        if args.heldout > 0:
            print(f"Held-out scoring, {args.heldout} matches per opponent ...", flush=True)
            results = compare(task, original, edited, runner, args.heldout)
            (run_dir / "heldout.json").write_text(json.dumps({"results": results, "weight_diff": diff},
                                                             indent=1, default=str))
            (run_dir / "report.md").write_text(report_markdown(task, results, diff, info, args.heldout),
                                               encoding="utf-8")
            print(f"Report: {run_dir / 'report.md'}")
    finally:
        if sandbox:
            sandbox.close()
        runner.close()
        for f in ("_eval.npz", "_final.npz"):
            (run_dir / "sandbox" / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
