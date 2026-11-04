"""
The weight editor loop: a model (Muse Spark through Meta's Model API, or a
scripted stand-in for testing) edits policy weights through three tools,
python / evaluate / submit, under fixed budgets. Everything is logged to the
run directory.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from wsp.matches import MatchRunner, summarize_metrics
from wsp.sandbox import Sandbox, save_npz
from wsp.tasks import Task, baseline_lines

INSTRUCTIONS = "Answer the user's request."   # one neutral line, as in the WSP harness

MAX_TOOL_OUTPUT = 20_000
MAX_EPISODES_PER_CALL = 20

TOOLS = [
    {"type": "function", "name": "python",
     "description": "Run Python code in the persistent weight-editing kernel (numpy). Returns printed "
                    "output, any error, and a warning if W no longer matches the architecture.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}},
                    "required": ["code"], "additionalProperties": False}},
    {"type": "function", "name": "evaluate",
     "description": "Play practice matches with the current W against the practice opponent and return "
                    "statistics. Costs `episodes` from the match budget.",
     "parameters": {"type": "object", "properties": {
         "episodes": {"type": "integer", "minimum": 1, "maximum": MAX_EPISODES_PER_CALL},
         "record": {"type": "boolean", "description": "save the matches as traces for load_trace()"}},
         "required": ["episodes", "record"], "additionalProperties": False}},
    {"type": "function", "name": "submit",
     "description": "Finish. The weights in W at this moment are final.",
     "parameters": {"type": "object", "properties": {"report": {"type": "string"}},
                    "required": ["report"], "additionalProperties": False}},
]

# Code patterns that would mean weights were fitted rather than written. Flagged for review, not blocked.
AUDIT = re.compile(r"lstsq|pinv|\.backward\(|autograd|optimi[sz]er|\bgrad\(|\.fit\(|minimize\(|curve_fit")


def _truncate(s: str, n: int = MAX_TOOL_OUTPUT) -> str:
    if len(s) <= n:
        return s
    head, tail = int(n * 0.65), int(n * 0.3)
    return s[:head] + f"\n... [{len(s) - head - tail} characters truncated] ...\n" + s[-tail:]


# ─────────────────────────────────────────────────────────────────────────────
#  Model clients
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Call:
    id: str
    name: str
    arguments: str


@dataclass
class Turn:
    calls: list
    text: str
    usage: dict
    raw: dict = field(default_factory=dict)


def _openai_client(base_url, api_key, timeout):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=4)


class ResponsesClient:
    """Responses API with server-side history (previous_response_id), as the Model API docs recommend
    for agentic work: reasoning is carried across tool turns."""

    def __init__(self, model, effort, base_url, api_key, max_output_tokens=None, timeout=1800):
        self.client = _openai_client(base_url, api_key, timeout)
        self.model, self.effort, self.max_out = model, effort, max_output_tokens
        self.prev = None

    def _create(self, input_):
        kw = dict(model=self.model, input=input_, tools=TOOLS)
        if self.effort:
            kw["reasoning"] = {"effort": self.effort}
        if self.max_out:
            kw["max_output_tokens"] = self.max_out
        if self.prev:
            kw["previous_response_id"] = self.prev
        r = self.client.responses.create(**kw)
        self.prev = r.id
        d = r.model_dump()
        calls, texts = [], []
        for item in d.get("output") or []:
            if item.get("type") == "function_call":
                calls.append(Call(item["call_id"], item["name"], item.get("arguments") or "{}"))
            elif item.get("type") == "message":
                texts += [c.get("text", "") for c in item.get("content") or [] if c.get("text")]
        u = d.get("usage") or {}
        usage = {"input": u.get("input_tokens"), "output": u.get("output_tokens"),
                 "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens")}
        return Turn(calls, "\n".join(texts), usage, d)

    def first(self, prompt):
        return self._create([{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}])

    def next(self, results, user_text=None):
        items = [{"type": "function_call_output", "call_id": cid, "output": out} for cid, out in results]
        if user_text:
            items.append({"role": "user", "content": user_text})
        return self._create(items)


class ChatClient:
    """OpenAI-compatible Chat Completions fallback (history kept client-side; reasoning is not carried
    across turns on this protocol)."""

    def __init__(self, model, effort, base_url, api_key, max_output_tokens=None, timeout=1800):
        self.client = _openai_client(base_url, api_key, timeout)
        self.model, self.effort, self.max_out = model, effort, max_output_tokens
        self.messages = [{"role": "system", "content": INSTRUCTIONS}]
        self.tools = [{"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} for t in TOOLS]

    def _create(self):
        kw = dict(model=self.model, messages=self.messages, tools=self.tools)
        if self.effort:
            kw["reasoning_effort"] = self.effort
        if self.max_out:
            kw["max_tokens"] = self.max_out
        r = self.client.chat.completions.create(**kw)
        d = r.model_dump()
        msg = d["choices"][0]["message"]
        self.messages.append({k: v for k, v in msg.items() if v is not None and k in ("role", "content", "tool_calls")})
        calls = [Call(tc["id"], tc["function"]["name"], tc["function"]["arguments"] or "{}")
                 for tc in msg.get("tool_calls") or []]
        u = d.get("usage") or {}
        usage = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
                 "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        return Turn(calls, msg.get("content") or "", usage, d)

    def first(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        return self._create()

    def next(self, results, user_text=None):
        for cid, out in results:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": out})
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        return self._create()


class ScriptedClient:
    """Replays a fixed list of tool calls. Used to test the harness without an API key."""

    def __init__(self, steps: list[tuple[str, dict]]):
        self.steps = list(steps)
        self.i = 0

    def _turn(self):
        if self.i >= len(self.steps):
            return Turn([], "(script finished)", {})
        name, args = self.steps[self.i]
        self.i += 1
        return Turn([Call(f"call_{self.i}", name, json.dumps(args))], f"[scripted step {self.i}]", {})

    def first(self, prompt):
        return self._turn()

    def next(self, results, user_text=None):
        return self._turn()


# ─────────────────────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, run_dir: Path, task: Task, original, sandbox: Sandbox, runner: MatchRunner,
                 python_budget: int, episode_budget: int, practice_seed: int):
        self.dir, self.task, self.original = run_dir, task, original
        self.sb, self.runner = sandbox, runner
        self.python_left, self.episodes_left = python_budget, episode_budget
        self.python_budget, self.episode_budget = python_budget, episode_budget
        self.next_seed = practice_seed
        self.n_recorded = 0
        self.done = False
        self.report = None
        self.audit = []
        self.log = open(run_dir / "tool_calls.jsonl", "a", encoding="utf-8")

    def _write(self, rec):
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()

    def call(self, turn: int, name: str, arguments: str) -> str:
        t0 = time.time()
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            out = f"Error: arguments are not valid JSON ({e})."
            self._write(dict(turn=turn, name=name, arguments=arguments, output=out, seconds=0))
            return out
        if name == "python":
            out = self._python(turn, str(args.get("code", "")))
        elif name == "evaluate":
            out = self._evaluate(int(args.get("episodes", 1)), bool(args.get("record", False)))
        elif name == "submit":
            out = self._submit(str(args.get("report", "")))
        else:
            out = f"Error: unknown tool {name!r}. Tools: python, evaluate, submit."
        self._write(dict(turn=turn, name=name, arguments=args, output=out, seconds=round(time.time() - t0, 2),
                         python_left=self.python_left, episodes_left=self.episodes_left))
        return out

    def _python(self, turn, code):
        if self.python_left <= 0:
            return "Python budget exhausted. You can still call evaluate (if matches remain) or submit."
        self.python_left -= 1
        for m in AUDIT.finditer(code):
            self.audit.append({"turn": turn, "match": m.group(0)})
        r = self.sb.run(code)
        parts = []
        if r.get("output"):
            parts.append(r["output"])
        if r.get("error"):
            parts.append(r["error"])
        if r.get("weights_problem"):
            parts.append(f"WARNING: {r['weights_problem']}. evaluate and submit need valid weights.")
        parts.append(f"[python runs left: {self.python_left}]")
        return _truncate("\n".join(parts))

    def _evaluate(self, episodes, record):
        if self.episodes_left <= 0:
            return "Match budget exhausted. Use python to keep analysing, or submit."
        n = max(1, min(episodes, MAX_EPISODES_PER_CALL, self.episodes_left))
        actors, problem = self.sb.current_weights("_eval.npz")
        if problem:
            return f"Cannot evaluate: {problem}"
        self.episodes_left -= n
        seeds = list(range(self.next_seed, self.next_seed + n))
        self.next_seed += n
        names = None
        paths = None
        if record:
            names = [f"practice_{self.n_recorded + i:03d}" for i in range(n)]
            self.n_recorded += n
            paths = [self.dir / "sandbox" / "traces" / f"{nm}.npz" for nm in names]
        ms = self.runner.run(actors, self.task.practice_opponent, seeds, paths)
        s = summarize_metrics(ms)
        lines = [f"{n} practice matches against {self.task.practice_opponent} (current W):", baseline_lines(s), "Per match:"]
        for i, m in enumerate(ms):
            t = m["team"]
            lines.append(f"  #{i}: score {m['blue_score']}-{m['red_score']}, grabs {t['flag_grabs']}, "
                         f"carry steps {t['carry_steps']}, carry progress {t['carry_progress_m']:.0f} m, "
                         f"tagged {t['times_tagged']}, oob {t['oob_events']}"
                         + (f"  -> trace '{names[i]}'" if names else ""))
        lines.append(f"[matches left: {self.episodes_left}]")
        return "\n".join(lines)

    def _submit(self, report):
        actors, problem = self.sb.current_weights("_final.npz")
        if problem:
            return f"Submit rejected: {problem}. Fix W, then submit again."
        self.final = actors
        save_npz(actors, self.dir / "final_weights.npz")
        self.report = report
        self.done = True
        return "Submitted. The harness will now score these weights on held-out matches."

    def fallback_final(self, reason: str):
        """End without a valid submit: use the last valid W the sandbox saved."""
        actors, problem = self.sb.current_weights("_final.npz")
        if problem:
            actors = self.sb.autosaved_weights() or self.original.copy()
        self.final = actors
        save_npz(actors, self.dir / "final_weights.npz")
        self.report = f"(no submit: {reason})"
        self.done = True


# ─────────────────────────────────────────────────────────────────────────────
#  Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_editor(client, session: Session, prompt: str, run_dir: Path, max_turns: int = 200,
               max_minutes: float = 240.0) -> dict:
    events = open(run_dir / "events.jsonl", "a", encoding="utf-8")
    transcript = open(run_dir / "transcript.md", "a", encoding="utf-8")
    usage_tot = {"input": 0, "output": 0, "reasoning": 0}
    t_start = time.time()
    nudges = 0
    turns = 0
    end_reason = "submitted"

    def record(turn: Turn):
        events.write(json.dumps({"turn": turns, "t": round(time.time() - t_start, 1), "usage": turn.usage,
                                 "raw": turn.raw}, default=str) + "\n")
        events.flush()
        for k in usage_tot:
            usage_tot[k] += turn.usage.get(k) or 0
        if turn.text:
            transcript.write(f"\n### Model (turn {turns})\n\n{turn.text}\n")
        transcript.flush()

    turn = client.first(prompt)
    while True:
        turns += 1
        record(turn)
        if not turn.calls:
            if nudges >= 3:
                end_reason = "model stopped without submitting"
                break
            nudges += 1
            turn = client.next([], user_text="You have not called submit yet. Continue the task, or call "
                                             "submit if you are finished.")
            continue
        results = []
        for c in turn.calls:
            out = session.call(turns, c.name, c.arguments)
            transcript.write(f"\n**{c.name}** `{_truncate(c.arguments, 4000)}`\n\n```\n{_truncate(out, 4000)}\n```\n")
            results.append((c.id, out))
            if session.done:
                break
        transcript.flush()
        if session.done:
            break
        if turns >= max_turns or (time.time() - t_start) / 60 > max_minutes:
            end_reason = "turn or time limit reached"
            break
        turn = client.next(results)

    if not session.done:
        session.fallback_final(end_reason)
    events.close()
    transcript.close()
    return {"turns": turns, "minutes": round((time.time() - t_start) / 60, 1), "end_reason": end_reason,
            "usage": usage_tot, "report": session.report, "audit_flags": session.audit,
            "python_runs_used": session.python_budget - session.python_left,
            "matches_used": session.episode_budget - session.episodes_left}
