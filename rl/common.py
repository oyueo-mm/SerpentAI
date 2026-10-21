"""Shared pieces of the learning pipeline: network/model construction, export to the browser format,
held-out evaluation through the Node bridge, and small statistics helpers."""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

from rl.bridge_client import ACTION_VERSION, NUM_ACTIONS, OBS_SIZE, OBS_VERSION, RULES_VERSION, Bridge
from rl.envs import EVAL_SEED_BASE

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "rl" / "runs"
MODELS_DIR = REPO / "public" / "models"
NET_ARCH = [256, 256]
GAMMA = 0.99

#: Held-out evaluation seeds (never used for training rollouts: training seeds are < EVAL_SEED_BASE).
EVAL_SEEDS = [EVAL_SEED_BASE + 10_000 + i for i in range(400)]
#: Held-out opponent mixes never used in training (generalisation check).
HELDOUT_OPPONENTS = {
    "mpc-heavy": ["mpc", "mpc", "mpc", "hybrid", "hybrid", "greedy"],
    "swarm": ["greedy"] * 6 + ["random"] * 6,
}


def device() -> str:
    return "cuda" if th.cuda.is_available() else "cpu"


def ppo_kwargs(**over: Any) -> dict[str, Any]:
    kw = dict(
        learning_rate=3e-4, n_steps=256, batch_size=1024, n_epochs=6, gamma=GAMMA, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
        policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=th.nn.Tanh),
        verbose=0, device=device(),
    )
    kw.update(over)
    return kw


def new_model(venv, seed: int, **over: Any) -> MaskablePPO:
    return MaskablePPO("MlpPolicy", venv, seed=seed, **ppo_kwargs(**over))


def policy_layers(policy) -> list[dict[str, Any]]:
    """Actor network as serpent-mlp layers (Linear+Tanh … action_net)."""
    layers: list[dict[str, Any]] = []
    mods = list(policy.mlp_extractor.policy_net)
    i = 0
    while i < len(mods):
        lin = mods[i]
        assert isinstance(lin, th.nn.Linear), f"unexpected module {lin}"
        act = "none"
        if i + 1 < len(mods) and isinstance(mods[i + 1], th.nn.Tanh):
            act, i = "tanh", i + 1
        elif i + 1 < len(mods) and isinstance(mods[i + 1], th.nn.ReLU):
            act, i = "relu", i + 1
        layers.append({"w": lin.weight.detach().cpu().double().tolist(), "b": lin.bias.detach().cpu().double().tolist(), "act": act})
        i += 1
    an = policy.action_net
    layers.append({"w": an.weight.detach().cpu().double().tolist(), "b": an.bias.detach().cpu().double().tolist(), "act": "none"})
    return layers


def export_model(policy, obs_rms, model_id: str, label: str, algo: str, meta: dict[str, Any], out_dir: Path = MODELS_DIR,
                 eps: float = 1e-8, clip: float = 10.0) -> Path:
    """Writes a serpent-mlp JSON the browser and the Node bridge can run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = {
        "format": "serpent-mlp", "id": model_id, "label": label, "algo": algo,
        "obsVersion": OBS_VERSION, "actionVersion": ACTION_VERSION, "rulesVersion": RULES_VERSION,
        "obsMean": obs_rms.mean.astype(float).tolist() if obs_rms is not None else None,
        "obsVar": obs_rms.var.astype(float).tolist() if obs_rms is not None else None,
        "obsEps": eps, "obsClip": clip, "layers": policy_layers(policy),
        "meta": {**meta, "exportedAt": _dt.datetime.now().isoformat(timespec="seconds")},
    }
    path = out_dir / f"{model_id}.json"
    path.write_text(json.dumps(f, separators=(",", ":")))
    return path


def logits_np(policy, obs_norm: np.ndarray) -> np.ndarray:
    """Actor logits for already-normalised observations."""
    with th.no_grad():
        x = th.as_tensor(obs_norm, dtype=th.float32, device=policy.device)
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(x, policy.pi_features_extractor))
        return policy.action_net(latent).cpu().numpy()


def normalize(obs: np.ndarray, rms, eps: float = 1e-8, clip: float = 10.0) -> np.ndarray:
    if rms is None:
        return obs.astype(np.float32)
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)


class RunningStats:
    """Minimal stand-in for VecNormalize.obs_rms when stats come from a dataset."""

    def __init__(self, mean: np.ndarray, var: np.ndarray):
        self.mean, self.var = mean.astype(np.float64), var.astype(np.float64)


def evaluate_in_node(model_file: Path | str | None, policy_id: str, env: dict[str, Any], seeds: list[int], workers: int = 6) -> list[dict]:
    """Runs full episodes in Node, split over `workers` bridge processes (the exported model goes through
    the TypeScript inference path). Episodes come back in seed order."""
    workers = max(1, min(workers, len(seeds)))
    chunks = [seeds[i::workers] for i in range(workers)]
    model = json.loads(Path(model_file).read_text()) if model_file is not None else None
    bridges = [Bridge() for _ in range(workers)]
    try:
        for b in bridges:
            b.init(1, {"scenario": "A-food"}, 0)
            if model is not None:
                b.add_model(model)
        for b, ch in zip(bridges, chunks):
            b.send("eval", policy=policy_id, env=env, seeds=ch, stochastic=False)
        results = [b.recv()[0]["episodes"] for b in bridges]
    finally:
        for b in bridges:
            b.close()
    by_seed = {e["seed"]: e for r in results for e in r}
    return [by_seed[s] for s in seeds]


def summarize_episodes(eps: list[dict]) -> dict[str, Any]:
    def ci(xs: list[float], floor: float = -math.inf) -> dict[str, float]:
        a = np.asarray(xs, dtype=float)
        if len(a) == 0:
            return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
        m = float(a.mean())
        h = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
        return {"mean": m, "lo": max(floor, m - h), "hi": m + h, "n": len(a)}

    def wilson(k: int, n: int) -> dict[str, float]:
        if n == 0:
            return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
        z, p = 1.96, k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return {"mean": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "n": n}

    n = len(eps)
    deaths = [e for e in eps if e["death"]]
    acts = np.sum([e["actions"] for e in eps], axis=0) if eps and "actions" in eps[0] else np.zeros(NUM_ACTIONS)
    tot = max(1, acts.sum())
    return {
        "episodes": n,
        "return": ci([e["r"] for e in eps]),
        "survivalS": ci([e["survivalS"] for e in eps], 0.0),
        "finalLength": ci([e["finalLength"] for e in eps], 0.0),
        "maxLength": ci([e["maxLength"] for e in eps], 0.0),
        "eaten": ci([e["eaten"] for e in eps], 0.0),
        "kills": ci([e["kills"] for e in eps], 0.0),
        "deathRate": wilson(len(deaths), n),
        "wallDeathRate": wilson(sum(1 for e in deaths if e["death"] == "wall"), n),
        "collisionDeathRate": wilson(sum(1 for e in deaths if e["death"] != "wall"), n),
        "boostShare": float(acts[1::2].sum() / tot),
        "actionDist": (acts / tot).round(4).tolist(),
    }


def obs_size_check() -> None:
    assert OBS_SIZE == 84 and NUM_ACTIONS == 18


def finite(obj: Any) -> Any:
    """Replaces NaN/±Infinity with None: browsers' JSON.parse rejects Python's NaN tokens."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return finite(obj.item())
    return obj


def save_json(path: Path, obj: Any) -> None:
    """Strict JSON (no NaN/Infinity) so the web pages can read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(obj), indent=1, default=float, allow_nan=False))


def vecnormalize_for(venv, obs_rms=None, training: bool = True) -> VecNormalize:
    # Rewards are left unnormalised (returns are O(1–20)) so BC value warm-starts use the same scale as PPO.
    vn = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=GAMMA)
    if obs_rms is not None:
        vn.obs_rms.mean = obs_rms.mean.copy()
        vn.obs_rms.var = obs_rms.var.copy()
        vn.obs_rms.count = 1e6
    vn.training = training
    return vn
