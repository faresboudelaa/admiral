"""
Parent side of the editor's Python sandbox (see sandbox_worker.py).

isolation="docker" (default) runs the worker in a container with no network,
a read-only root filesystem, no Linux capabilities, CPU / memory / process
limits, and only this run's sandbox folder mounted. That container is the
security boundary for the model's code. isolation="local" runs the worker as a
plain child process and is only meant for harness tests that run no model code.

Nothing from the sandbox is trusted on the way back: weights are re-read from
.npz files (no pickle) and checked against the original architecture here.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np

from wsp.sim import ActorSet

WORKER = Path(__file__).with_name("sandbox_worker.py")
DOCKERFILE_DIR = Path(__file__).with_name("docker")
IMAGE = "admiral-wsp-sandbox:1"
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
MAX_REPLY_CHARS = 4_000_000


def save_npz(actors: ActorSet, path: Path):
    np.savez(path, **{f"{a}/{t}": x for a, net in actors.nets.items() for t, x in net.items()})


def load_npz(path: Path, like: ActorSet) -> ActorSet:
    """Read weights written by the sandbox and check them against `like` (keys, shapes, finite values)."""
    nets = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            a, _, t = k.partition("/")
            if a not in like.nets or t not in like.nets[a]:
                raise ValueError(f"unexpected tensor {k!r}")
            x = np.asarray(z[k], dtype=np.float32)
            if x.shape != like.nets[a][t].shape:
                raise ValueError(f"{a}.{t} has shape {x.shape}, expected {like.nets[a][t].shape}")
            if not np.all(np.isfinite(x)):
                raise ValueError(f"{a}.{t} contains NaN or inf")
            nets.setdefault(a, {})[t] = x
    missing = [f"{a}.{t}" for a, net in like.nets.items() for t in net if t not in nets.get(a, {})]
    if missing:
        raise ValueError(f"missing tensors {missing}")
    return ActorSet(nets, like.shared, like.obs_dim, str(path))


def docker_ready() -> str | None:
    """None if Docker is up and the sandbox image exists, else a message saying what is missing."""
    try:
        r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"Docker is not available ({e}). Start Docker Desktop."
    if r.returncode != 0:
        if "daemon" in r.stderr or "pipe" in r.stderr:
            return "The Docker daemon is not running. Start Docker Desktop."
        return f"Sandbox image {IMAGE} is missing. Build it with: docker build -t {IMAGE} {DOCKERFILE_DIR}"
    return None


class Sandbox:
    def __init__(self, workdir: Path, original: ActorSet, context: dict, timeout: float = 180.0,
                 isolation: str = "docker"):
        self.dir = Path(workdir).resolve()
        (self.dir / "traces").mkdir(parents=True, exist_ok=True)
        self.original = original
        self.timeout = timeout
        self.isolation = isolation
        save_npz(original, self.dir / "weights_original.npz")
        (self.dir / "context.json").write_text(json.dumps(context), encoding="utf-8")
        self.proc = None
        self.container = None
        self._start()

    def _command(self) -> tuple[list[str], dict | None]:
        if self.isolation == "local":
            env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
            env["PYTHONIOENCODING"] = "utf-8"
            return [sys.executable, "-I", str(WORKER), str(self.dir)], env
        self.container = f"wsp-sandbox-{uuid.uuid4().hex[:12]}"
        return ["docker", "run", "--rm", "-i", "--name", self.container,
                "--network", "none",
                "--read-only", "--tmpfs", "/tmp:rw,size=64m",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--memory", "4g", "--memory-swap", "4g", "--cpus", "2", "--pids-limit", "128",
                "--mount", f"type=bind,src={self.dir},dst=/work",
                "--mount", f"type=bind,src={WORKER},dst=/app/sandbox_worker.py,readonly",
                "-e", "PYTHONIOENCODING=utf-8", "-e", "PYTHONDONTWRITEBYTECODE=1",
                IMAGE, "python", "-I", "/app/sandbox_worker.py", "/work"], None

    def _start(self):
        cmd, env = self._command()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=str(self.dir), env=env, text=True, encoding="utf-8", bufsize=1)
        self._q = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._q), daemon=True).start()
        try:
            hello = self._q.get(timeout=120)
        except queue.Empty:
            hello = None
        if not hello or not json.loads(hello).get("ready"):
            self._stop()
            raise RuntimeError(f"sandbox worker failed to start ({self.isolation})")

    @staticmethod
    def _pump(proc, q):
        while True:
            line = proc.stdout.readline(MAX_REPLY_CHARS)
            if not line:
                break
            q.put(line)
        q.put(None)

    def _stop(self):
        if self.container:
            subprocess.run(["docker", "kill", self.container], capture_output=True, timeout=60)
            self.container = None
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def _restart(self):
        self._stop()
        self._start()

    def _call(self, req: dict, timeout: float) -> dict:
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            self._restart()
            return {"ok": False, "timeout": True,
                    "error": f"TimeoutError: the run took longer than {timeout:.0f} s. The sandbox was "
                             "restarted: W was restored to its state after the last successful run, "
                             "and all other variables were lost."}
        except OSError:
            line = None
        if line is None:
            self._restart()
            return {"ok": False, "error": "The sandbox process crashed and was restarted "
                                          "(W restored from the last successful run; other variables lost)."}
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            self._restart()
            return {"ok": False, "error": "The sandbox reply was malformed or too long, so the sandbox was "
                                          "restarted (W restored from the last successful run)."}
        return reply if isinstance(reply, dict) else {"ok": False, "error": "malformed sandbox reply"}

    def run(self, code: str) -> dict:
        return self._call({"cmd": "exec", "code": code}, self.timeout)

    def current_weights(self, name: str) -> tuple[ActorSet | None, str | None]:
        """Have the sandbox write W to <sandbox dir>/<name>, then read and validate it here."""
        r = self._call({"cmd": "dump", "path": name}, 60)
        if not r.get("ok"):
            return None, str(r.get("weights_problem") or r.get("error"))
        try:
            return load_npz(self.dir / name, self.original), None
        except Exception as e:  # anything the sandbox wrote is untrusted
            return None, f"invalid weights: {e}"

    def autosaved_weights(self) -> ActorSet | None:
        try:
            return load_npz(self.dir / "_autosave.npz", self.original)
        except Exception:
            return None

    def close(self):
        self._stop()
