"""NumPy inference of an exported serpent-mlp policy (same arithmetic as src/policy/mlp.ts)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_ACTIONS = 18
TURN_LEVELS = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])


class MlpPolicy:
    def __init__(self, path: str | Path):
        f = json.loads(Path(path).read_text())
        if f.get("format") != "serpent-mlp":
            raise ValueError(f"{path}: not a serpent-mlp model")
        if f["obsVersion"] != "obs-v1" or f["actionVersion"] != "act-v1":
            raise ValueError(f"{path}: contract {f['obsVersion']}/{f['actionVersion']} is not obs-v1/act-v1")
        self.file = f
        self.id = f["id"]
        self.algo = f["algo"]
        self.mean = np.asarray(f["obsMean"], dtype=np.float64) if f["obsMean"] is not None else None
        self.var = np.asarray(f["obsVar"], dtype=np.float64) if f["obsVar"] is not None else None
        self.eps, self.clip = float(f["obsEps"]), float(f["obsClip"])
        self.layers = [(np.asarray(l["w"], dtype=np.float64), np.asarray(l["b"], dtype=np.float64), l["act"]) for l in f["layers"]]

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        if self.mean is not None:
            x = (x - self.mean) / np.sqrt(self.var + self.eps)
        x = np.clip(x, -self.clip, self.clip)
        for w, b, act in self.layers:
            x = w @ x + b
            if act == "tanh":
                x = np.tanh(x)
            elif act == "relu":
                x = np.maximum(x, 0)
        return x

    def act(self, obs: np.ndarray, can_boost: bool) -> tuple[int, np.ndarray]:
        lg = self.logits(obs)
        mask = np.ones(NUM_ACTIONS, bool)
        if not can_boost:
            mask[1::2] = False
        lg = np.where(mask, lg, -np.inf)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        return int(np.argmax(p)), p


def decode(a: int) -> tuple[float, bool]:
    return float(TURN_LEVELS[a >> 1]), bool(a & 1)
