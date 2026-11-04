"""
Persistent Python kernel for the weight editor (runs as a child process).

The editor model's code runs here, in one namespace that survives between
calls, with the policy weights loaded as numpy arrays in `W`. It cannot
import the simulator, a deep-learning framework or a network client: the
only way to see the policy play is the harness's `evaluate` tool, which is
budgeted.

This is a rule-enforcement layer, not a security boundary: the code runs as
the local user. The harness strips API keys from this process's environment.

Protocol: one JSON object per line on stdin; one JSON reply per line on a
private copy of the original stdout. Commands: exec, dump, ping.
"""

import contextlib
import importlib.abc
import io
import json
import os
import sys
import traceback

import numpy as np

BLOCKED = {
    # the simulator: rollouts must go through the budgeted evaluate tool
    "pyquaticus", "pygame", "scenario_init", "wsp", "gymnasium", "pettingzoo",
    # training frameworks: edits are written, not optimised
    "torch", "tensorflow", "jax", "sklearn", "keras",
    # network clients
    "openai", "anthropic", "httpx", "requests", "urllib3", "aiohttp", "socket", "http", "ssl",
}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED or name == "urllib.request":
            raise ImportError(f"import of '{name}' is not allowed in the weight-editing sandbox")
        return None


PRELUDE = r'''
import numpy as np

def _load_npz(path):
    out = {}
    with np.load(path) as z:
        for k in z.files:
            agent, tensor = k.split("/")
            out.setdefault(agent, {})[tensor] = z[k].astype(np.float32)
    return out

W0 = _load_npz("weights_original.npz")
for _net in W0.values():
    for _a in _net.values():
        _a.setflags(write=False)
W = _load_npz("_autosave.npz") if __import__("os").path.exists("_autosave.npz") else \
    {a: {t: x.copy() for t, x in net.items()} for a, net in W0.items()}

import json as _json
_ctx = _json.load(open("context.json"))
OBS_NAMES = _ctx["obs_names"]
ACTIONS = _ctx["actions"]
AGENTS = sorted(W0)

def forward(net, obs, hidden=False):
    """Actor forward pass. net = W["agent_0"] (or any dict with W1..b3); obs: (B, D) or (D,).
    Returns logits, or (logits, h1, h2) with hidden=True."""
    x = np.asarray(obs, dtype=np.float32)
    h1 = np.tanh(x @ net["W1"].T + net["b1"])
    h2 = np.tanh(h1 @ net["W2"].T + net["b2"])
    logits = h2 @ net["W3"].T + net["b3"]
    return (logits, h1, h2) if hidden else logits

def probs(net, obs):
    """Action probabilities (the policy samples from these during evaluation)."""
    lg = forward(net, obs)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def list_traces():
    import os
    return sorted(f[:-4] for f in os.listdir("traces") if f.endswith(".npz"))

def load_trace(name):
    """Recorded episode as a dict of arrays, indexed [step, blue_agent, ...]. See TRACE_DOC."""
    with np.load(f"traces/{name}.npz") as z:
        return {k: z[k] for k in z.files}

TRACE_DOC = _ctx["trace_doc"]

def changed_fraction(tol=0.0):
    """Fraction of weights in W that differ from W0 (overall, and per tensor)."""
    per, n, c = {}, 0, 0
    for a in W0:
        for t in W0[a]:
            d = np.abs(W[a][t] - W0[a][t]) > tol
            per[f"{a}.{t}"] = int(d.sum())
            n += d.size
            c += int(d.sum())
    return c / n, per

_SNAPSHOTS = {}
def snapshot(name):
    """Keep an in-memory copy of the current W under `name`."""
    _SNAPSHOTS[name] = {a: {t: x.copy() for t, x in net.items()} for a, net in W.items()}

def restore(name):
    """Set W back to a snapshot (or to the original with name='original')."""
    global W
    src = W0 if name == "original" else _SNAPSHOTS[name]
    W = {a: {t: x.copy() for t, x in net.items()} for a, net in src.items()}
'''


def _check(ns) -> str | None:
    """Return an error message if W no longer matches the original architecture."""
    W, W0 = ns.get("W"), ns["W0"]
    if not isinstance(W, dict) or set(W) != set(W0):
        return f"W must be a dict with keys {sorted(W0)}"
    for a in W0:
        if not isinstance(W[a], dict) or set(W[a]) != set(W0[a]):
            return f"W['{a}'] must have keys {sorted(W0[a])}"
        for t, x0 in W0[a].items():
            x = np.asarray(W[a][t])
            if x.shape != x0.shape:
                return f"W['{a}']['{t}'] has shape {x.shape}, the architecture needs {x0.shape}"
            if not np.all(np.isfinite(x)):
                return f"W['{a}']['{t}'] contains NaN or inf"
    return None


def _save(ns, path):
    flat = {f"{a}/{t}": np.asarray(x, dtype=np.float32) for a, net in ns["W"].items() for t, x in net.items()}
    np.savez(path, **flat)


def main():
    workdir = sys.argv[1]
    os.chdir(workdir)

    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    ns = {"__name__": "__sandbox__"}
    exec(PRELUDE, ns)
    sys.meta_path.insert(0, _Blocker())

    def reply(**kw):
        proto.write(json.dumps(kw) + "\n")
        proto.flush()

    reply(ok=True, ready=True)
    for line in sys.stdin:
        req = json.loads(line)
        cmd = req.get("cmd")
        if cmd == "ping":
            reply(ok=True)
        elif cmd == "exec":
            buf = io.StringIO()
            err = None
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    exec(compile(req["code"], "<editor>", "exec"), ns)
                except SystemExit:
                    err = "SystemExit is ignored in the sandbox"
                except BaseException:
                    err = traceback.format_exc(limit=8)
            problem = _check(ns)
            if problem is None:
                _save(ns, "_autosave.npz")
            reply(ok=err is None, output=buf.getvalue(), error=err, weights_problem=problem)
        elif cmd == "dump":
            problem = _check(ns)
            if problem is None:
                _save(ns, req["path"])
            reply(ok=problem is None, weights_problem=problem)
        else:
            reply(ok=False, error=f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
