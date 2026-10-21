"""Client for the Node engine bridge (rl/bridge/bridge.ts) — Method A.

The Python side never re-implements game rules for training: every environment step runs the same
TypeScript engine, sensors, obs-v1 encoder and reward-v1 code as the browser. This module only frames
messages over the child process's pipes.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_JS = REPO_ROOT / "rl" / "bridge" / "dist" / "bridge.mjs"

OBS_VERSION = "obs-v1"
ACTION_VERSION = "act-v1"
RULES_VERSION = "rules-v1"
REWARD_VERSION = "reward-v1"
OBS_SIZE = 84
NUM_ACTIONS = 18


class BridgeError(RuntimeError):
    pass


class Bridge:
    """One Node process hosting `num_envs` single-learner environments."""

    def __init__(self, node: str | None = None, bridge_js: Path | str | None = None):
        path = Path(bridge_js) if bridge_js else BRIDGE_JS
        if not path.exists():
            raise BridgeError(f"{path} not found - run `npm ci && npm run build:node` first")
        node = node or os.environ.get("SERPENT_NODE") or shutil.which("node")
        if not node:
            raise BridgeError("Node.js not found on PATH (set SERPENT_NODE)")
        self.proc = subprocess.Popen(
            [node, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, cwd=str(REPO_ROOT), bufsize=0
        )
        self.num_envs = 0
        self.versions: dict[str, Any] = {}
        #: act-v1 action each env actually executed in the last step.
        self.last_taken: list[int] = []
        #: expert labels for the observations returned by the last reset/step (-1 = none).
        self.last_experts: list[int] = []
        #: boost availability for the observations returned by the last reset/step.
        self.last_can_boost: list[bool] = []

    # ------------------------------------------------------------- framing
    def _read_exact(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        chunks, got = [], 0
        while got < n:
            b = self.proc.stdout.read(n - got)
            if not b:
                raise BridgeError(f"bridge process exited (code {self.proc.poll()})")
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    def send(self, cmd: str, **payload: Any) -> None:
        """Writes a request without waiting, so several bridge processes can work in parallel."""
        assert self.proc.stdin is not None
        data = json.dumps({"cmd": cmd, **payload}).encode("utf-8")
        self.proc.stdin.write(struct.pack("<I", len(data)) + data)
        self.proc.stdin.flush()

    def recv(self) -> tuple[dict[str, Any], np.ndarray | None]:
        (hlen,) = struct.unpack("<I", self._read_exact(4))
        header = json.loads(self._read_exact(hlen).decode("utf-8"))
        arr = None
        if header.get("bin"):
            (blen,) = struct.unpack("<I", self._read_exact(4))
            arr = np.frombuffer(self._read_exact(blen), dtype="<f4").copy()
        if not header.get("ok"):
            raise BridgeError(header.get("error", "unknown bridge error"))
        return header, arr

    def call(self, cmd: str, **payload: Any) -> tuple[dict[str, Any], np.ndarray | None]:
        self.send(cmd, **payload)
        return self.recv()

    # ------------------------------------------------------------- commands
    def init(self, num_envs: int, env: dict[str, Any], seed: int) -> dict[str, Any]:
        header, _ = self.call("init", numEnvs=num_envs, env=env, seed=seed)
        self.num_envs = num_envs
        self.versions = {k: header[k] for k in ("rules", "obs", "action", "reward")}
        if header["obsSize"] != OBS_SIZE or header["numActions"] != NUM_ACTIONS:
            raise BridgeError(f"contract mismatch: bridge obs {header['obsSize']} / actions {header['numActions']}")
        if self.versions["obs"] != OBS_VERSION or self.versions["action"] != ACTION_VERSION:
            raise BridgeError(f"contract mismatch: {self.versions}")
        return header

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        header, arr = self.call("reset", seeds=seeds)
        assert arr is not None
        self.last_experts = header["experts"]
        self.last_can_boost = header["canBoost"]
        return arr.reshape(-1, OBS_SIZE)

    def step_send(self, actions: np.ndarray | list[int]) -> None:
        self.send("step", actions=[int(a) for a in actions])

    def step_recv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
        header, arr = self.recv()
        assert arr is not None
        rows = arr.reshape(-1, OBS_SIZE)
        n = len(header["rewards"])
        self.last_taken = header["taken"]
        self.last_experts = header["experts"]
        self.last_can_boost = header["canBoost"]
        return (
            rows[:n],
            np.asarray(header["rewards"], dtype=np.float32),
            np.asarray(header["dones"], dtype=bool),
            np.asarray(header["truncated"], dtype=bool),
            header["infos"],
            rows[n:],
        )

    def step(self, actions):
        self.step_send(actions)
        return self.step_recv()

    def set_env(self, env: dict[str, Any]) -> None:
        self.call("setEnv", env=env)

    def add_model(self, model: dict[str, Any]) -> None:
        self.call("addModel", model=model)

    def evaluate(self, policy: str, env: dict[str, Any], seeds: list[int], stochastic: bool = False) -> list[dict[str, Any]]:
        header, _ = self.call("eval", policy=policy, env=env, seeds=seeds, stochastic=stochastic)
        return header["episodes"]

    def obs_fixture(self, index: int = 0) -> dict[str, Any]:
        header, _ = self.call("obs", index=index)
        return header

    def snapshot(self, index: int = 0) -> dict[str, Any]:
        header, _ = self.call("snapshot", index=index)
        return header["match"]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call("close")
            except BridgeError:
                pass
            self.proc.wait(timeout=5)

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def masks_from(can_boost: list[bool]) -> np.ndarray:
    """act-v1 masks: boost actions (odd indices) are invalid when boost is unavailable."""
    m = np.ones((len(can_boost), NUM_ACTIONS), dtype=bool)
    for i, cb in enumerate(can_boost):
        if not cb:
            m[i, 1::2] = False
    return m
