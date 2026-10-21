"""Imitation learning: expert dataset collection, Behavior Cloning and DAgger.

    python -m rl.imitate all --expert mpc --steps 200000 --rounds 4 --run il-v1

Artifacts (rl/runs/<run>/): datasets (*.npz), checkpoints (bc-*.zip), log.jsonl, summary.json.
Exported browser models go to public/models/ and a compact report to public/data/rl/<run>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

from rl.bridge_client import NUM_ACTIONS, OBS_SIZE, OBS_VERSION, ACTION_VERSION, RULES_VERSION, REWARD_VERSION
from rl.common import (EVAL_SEEDS, GAMMA, HELDOUT_OPPONENTS, REPO, RUNS, RunningStats, export_model, logits_np, new_model,
                       normalize, save_json, summarize_episodes, evaluate_in_node)
from rl.envs import SerpentVecEnv, action_space, observation_space

TURNS = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
SCENARIO_MIX = {"C-moving": 0.25, "D-mixed": 0.45, "E-attack": 0.30}


# ------------------------------------------------------------------ dataset
@dataclass
class Dataset:
    obs: np.ndarray        # (N, 84) float32
    label: np.ndarray      # (N,) expert act-v1 label
    taken: np.ndarray      # (N,) action actually executed in that state
    mask: np.ndarray       # (N, 18) bool
    episode: np.ndarray    # (N,) global episode id
    seed: np.ndarray       # (N,) episode seed
    tick: np.ndarray       # (N,) decision index within the episode
    reward: np.ndarray     # (N,) reward-v1 received after the step
    done: np.ndarray       # (N,) episode ended after this step
    source: np.ndarray     # (N,) 0 = expert(+noise) rollout, k = DAgger round k
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.label)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, obs=self.obs, label=self.label, taken=self.taken, mask=self.mask, episode=self.episode, seed=self.seed,
                            tick=self.tick, reward=self.reward, done=self.done, source=self.source, meta=json.dumps(self.meta))

    @staticmethod
    def load(path: Path) -> "Dataset":
        z = np.load(path, allow_pickle=False)
        return Dataset(**{k: z[k] for k in ["obs", "label", "taken", "mask", "episode", "seed", "tick", "reward", "done", "source"]}, meta=json.loads(str(z["meta"])))

    @staticmethod
    def concat(parts: list["Dataset"]) -> "Dataset":
        fields = ["obs", "label", "taken", "mask", "episode", "seed", "tick", "reward", "done", "source"]
        meta = {**parts[0].meta, "parts": [p.meta for p in parts]}
        return Dataset(**{f: np.concatenate([getattr(p, f) for p in parts]) for f in fields}, meta=meta)


def state_categories(obs: np.ndarray, label: np.ndarray) -> dict[str, np.ndarray]:
    """Critical-state tags from obs-v1 features (used for the data-quality report and oversampling)."""
    body = obs[:, 6:38:2]
    wall = obs[:, 7:38:2]
    fwd = np.minimum(np.minimum(body[:, [15, 0, 1]], wall[:, [15, 0, 1]]), 1).min(axis=1)
    food = obs[:, 38:54:2].sum(axis=1)
    opp_d = np.hypot(obs[:, 55], obs[:, 56])
    turn = TURNS[label // 2]
    return {
        "danger_ahead": fwd < 0.3,
        "near_body": body.min(axis=1) < 0.15,
        "near_wall": wall.min(axis=1) < 0.3,
        "encircled": (body < 0.5).sum(axis=1) >= 8,
        "sharp_turn": np.abs(turn) >= 0.75,
        "boost": (label % 2) == 1,
        "food_dense": food > 5.45,  # ≈ top 10 % of the expert data (8 log-scaled sector sums)
        "opponent_close": (obs[:, 54] > 0.5) & (opp_d < 0.4),
    }


def sample_weights(ds: Dataset) -> np.ndarray:
    c = state_categories(ds.obs, ds.label)
    w = 1.0 + 2.0 * c["danger_ahead"] + 1.0 * c["near_body"] + 1.0 * c["near_wall"] + 1.0 * c["encircled"] + 1.0 * c["opponent_close"]
    return w / w.mean()


def quality_report(ds: Dataset) -> dict[str, Any]:
    c = state_categories(ds.obs, ds.label)
    hist = np.bincount(ds.label, minlength=NUM_ACTIONS)
    return {
        "samples": int(len(ds)), "episodes": int(len(np.unique(ds.episode))),
        "categories": {k: float(v.mean()) for k, v in c.items()},
        "labelDist": (hist / hist.sum()).round(4).tolist(),
        "labelDiffersFromTaken": float((ds.label != ds.taken).mean()),
    }


class ActingPolicy:
    """A BC/DAgger policy acting inside data collection (masked arg-max or sampling)."""

    def __init__(self, model: MaskablePPO, rms: RunningStats, stochastic: bool = True, seed: int = 0):
        self.model, self.rms, self.stochastic = model, rms, stochastic
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        lg = logits_np(self.model.policy, normalize(obs, self.rms))
        lg = np.where(masks, lg, -1e9)
        if not self.stochastic:
            return lg.argmax(axis=1)
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return np.array([self.rng.choice(NUM_ACTIONS, p=pi) for pi in p])


def collect(expert: str, steps: int, seed: int, *, epsilon: float = 0.0, policy: ActingPolicy | None = None, beta: float = 0.0,
            source: int = 0, episode_base: int = 0, bridges: int = 6, per_bridge: int = 8, mix: dict[str, float] = SCENARIO_MIX) -> Dataset:
    """Rolls out (expert + ε-noise) or (policy, executing the expert's action with prob. β) and labels every state with the expert."""
    rng = np.random.default_rng(seed)
    parts: list[Dataset] = []
    ep_counter = episode_base
    for k, (scenario, frac) in enumerate(mix.items()):
        n_target = int(steps * frac)
        env = SerpentVecEnv({"scenario": scenario, "expert": expert, "maxSteps": 1200}, num_bridges=bridges, envs_per_bridge=per_bridge, seed=seed + 7919 * k)
        try:
            obs = env.reset()
            n = env.num_envs
            ep_id = np.arange(ep_counter, ep_counter + n)
            ep_counter += n
            tick = np.zeros(n, dtype=np.int64)
            buf: dict[str, list] = {f: [] for f in ["obs", "label", "taken", "mask", "episode", "seed", "tick", "reward", "done"]}
            while len(buf["label"]) * n < n_target:
                labels, masks, seeds = env.experts.copy(), env.action_masks().copy(), env.seeds.copy()
                if policy is None:
                    acts = labels.copy()
                    noisy = rng.random(n) < epsilon
                else:
                    acts = policy(obs, masks)
                    noisy = np.zeros(n, dtype=bool)
                    use_expert = rng.random(n) < beta
                    acts[use_expert] = labels[use_expert]
                for i in np.flatnonzero(noisy):
                    acts[i] = rng.choice(np.flatnonzero(masks[i]))
                obs2, rew, done, infos = env.step(acts)
                buf["obs"].append(obs); buf["label"].append(labels); buf["taken"].append(env.taken.copy()); buf["mask"].append(masks)
                buf["episode"].append(ep_id.copy()); buf["seed"].append(seeds); buf["tick"].append(tick.copy())
                buf["reward"].append(rew); buf["done"].append(done)
                tick += 1
                for i in np.flatnonzero(done):
                    ep_id[i] = ep_counter; ep_counter += 1; tick[i] = 0
                obs = obs2
            # (T, n, …) → episode-major flat arrays
            def flat(name):
                a = np.stack(buf[name], axis=1)
                return a.reshape(-1, *a.shape[2:])
            parts.append(Dataset(obs=flat("obs").astype(np.float32), label=flat("label").astype(np.int64), taken=flat("taken").astype(np.int64),
                                 mask=flat("mask").astype(bool), episode=flat("episode").astype(np.int64), seed=flat("seed").astype(np.int64),
                                 tick=flat("tick").astype(np.int64), reward=flat("reward").astype(np.float32), done=flat("done").astype(bool),
                                 source=np.full(len(flat("label")), source, dtype=np.int64), meta={}))
        finally:
            env.close()
    ds = Dataset.concat(parts)
    ds.meta = {"expert": expert, "epsilon": epsilon, "beta": beta, "source": source, "seed": seed, "mix": mix,
               "versions": {"rules": RULES_VERSION, "obs": OBS_VERSION, "action": ACTION_VERSION, "reward": REWARD_VERSION},
               "acting": "expert+noise" if policy is None else "policy"}
    return ds


def episode_split(ds: Dataset, val_frac: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    eps = np.unique(ds.episode)
    rng = np.random.default_rng(seed)
    val_eps = set(rng.choice(eps, size=max(1, int(len(eps) * val_frac)), replace=False).tolist())
    is_val = np.array([e in val_eps for e in ds.episode])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def discounted_returns(ds: Dataset, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo returns per episode; the last 200 steps of unfinished episodes are marked invalid."""
    ret = np.zeros(len(ds), dtype=np.float32)
    valid = np.ones(len(ds), dtype=bool)
    order = np.lexsort((ds.tick, ds.episode))
    i = len(order) - 1
    while i >= 0:
        e = ds.episode[order[i]]
        j = i
        while j >= 0 and ds.episode[order[j]] == e:
            j -= 1
        idx = order[j + 1:i + 1]
        g = 0.0
        for k in idx[::-1]:
            g = ds.reward[k] + gamma * g * (0.0 if ds.done[k] else 1.0)
            ret[k] = g
        if not ds.done[idx[-1]]:
            valid[idx[-200:]] = False
        i = j
    return ret, valid


# ------------------------------------------------------------------ BC
class _SpaceEnv(gym.Env):
    observation_space = observation_space()
    action_space = action_space()

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS_SIZE, np.float32), {}

    def step(self, a):
        return np.zeros(OBS_SIZE, np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(NUM_ACTIONS, bool)


def blank_model(seed: int) -> MaskablePPO:
    from stable_baselines3.common.vec_env import DummyVecEnv
    return new_model(DummyVecEnv([_SpaceEnv]), seed)


def dataset_stats(ds: Dataset) -> RunningStats:
    return RunningStats(ds.obs.mean(axis=0), ds.obs.var(axis=0))


def train_bc(ds: Dataset, rms: RunningStats, *, model: MaskablePPO | None = None, epochs: int = 8, lr: float = 1e-3, batch: int = 1024,
             seed: int = 0, value_epochs: int = 3, log=None) -> tuple[MaskablePPO, dict[str, Any]]:
    """Cross-entropy on masked logits of the PPO actor (weighted towards critical states) + value warm-start."""
    th.manual_seed(seed)
    model = model or blank_model(seed)
    pol = model.policy
    tr, va = episode_split(ds, seed=seed)
    X = th.as_tensor(normalize(ds.obs, rms))
    Y = th.as_tensor(ds.label)
    M = th.as_tensor(ds.mask)
    W = th.as_tensor(sample_weights(ds), dtype=th.float32)
    actor_params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = th.optim.Adam(actor_params, lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(tr)
        tot, n = 0.0, 0
        for s in range(0, len(perm), batch):
            idx = th.as_tensor(perm[s:s + batch])
            lat = pol.mlp_extractor.forward_actor(X[idx])
            logits = pol.action_net(lat).masked_fill(~M[idx], -1e9)
            loss = (th.nn.functional.cross_entropy(logits, Y[idx], reduction="none") * W[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        acc = bc_metrics(model, ds, rms, va)
        hist.append({"epoch": ep, "trainLoss": tot / n, **acc})
        if log:
            log({"phase": "bc", **hist[-1]})
    # Value warm-start on Monte-Carlo returns so PPO does not start from a random critic.
    ret, valid = discounted_returns(ds)
    vidx = np.intersect1d(tr, np.flatnonzero(valid))
    vparams = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
    vopt = th.optim.Adam(vparams, lr=1e-3)
    R = th.as_tensor(ret)
    vloss = float("nan")
    for _ in range(value_epochs):
        perm = rng.permutation(vidx)
        for s in range(0, len(perm), batch):
            idx = th.as_tensor(perm[s:s + batch])
            v = pol.value_net(pol.mlp_extractor.forward_critic(X[idx])).squeeze(-1)
            l = th.nn.functional.mse_loss(v, R[idx])
            vopt.zero_grad(); l.backward(); vopt.step()
            vloss = float(l)
    return model, {"history": hist, "valueLoss": vloss, "trainSamples": int(len(tr)), "valSamples": int(len(va))}


def bc_metrics(model: MaskablePPO, ds: Dataset, rms: RunningStats, idx: np.ndarray) -> dict[str, float]:
    lg = logits_np(model.policy, normalize(ds.obs[idx], rms))
    lg = np.where(ds.mask[idx], lg, -1e9)
    pred = lg.argmax(axis=1)
    y = ds.label[idx]
    tp, ty = TURNS[pred // 2], TURNS[y // 2]
    c = state_categories(ds.obs[idx], y)
    crit = c["danger_ahead"] | c["near_body"]
    return {
        "valAccuracy": float((pred == y).mean()),
        "valTurnWithinOne": float((np.abs(pred // 2 - y // 2) <= 1).mean()),
        "valTurnSignAgree": float((np.sign(tp) == np.sign(ty)).mean()),
        "valBoostAgree": float(((pred % 2) == (y % 2)).mean()),
        "valCriticalAccuracy": float((pred[crit] == y[crit]).mean()) if crit.any() else float("nan"),
    }


# ------------------------------------------------------------------ evaluation
def evaluate_policy_file(model_file: Path | None, policy_id: str, episodes: int, max_steps: int = 1200) -> dict[str, Any]:
    seeds = EVAL_SEEDS[:episodes]
    res: dict[str, Any] = {}
    res["D-mixed"] = summarize_episodes(evaluate_in_node(model_file, policy_id, {"scenario": "D-mixed", "maxSteps": max_steps}, seeds))
    for name, opps in HELDOUT_OPPONENTS.items():
        res[f"heldout:{name}"] = summarize_episodes(evaluate_in_node(model_file, policy_id, {"scenario": "D-mixed", "opponents": opps, "maxSteps": max_steps}, seeds[: max(10, episodes // 2)]))
    return res


def export_bc(model: MaskablePPO, rms: RunningStats, model_id: str, label: str, algo: str, meta: dict[str, Any], out_dir: Path) -> Path:
    return export_model(model.policy, rms, model_id, label, algo, meta, out_dir=out_dir)


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--expert", default="mpc")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--dagger-steps", type=int, default=60_000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--run", default="il-v1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bridges", type=int, default=6)
    args = ap.parse_args()

    run = RUNS / args.run
    run.mkdir(parents=True, exist_ok=True)
    logf = (run / "log.jsonl").open("a", encoding="utf-8")

    def log(rec: dict[str, Any]) -> None:
        rec = {"t": time.time(), **rec}
        logf.write(json.dumps(rec, default=float) + "\n"); logf.flush()
        print(json.dumps(rec, default=float))

    t0 = time.time()
    d0_path = run / "dataset-d0.npz"
    if d0_path.exists():
        d0 = Dataset.load(d0_path)
    else:
        d0 = collect(args.expert, args.steps, args.seed * 1000, epsilon=args.epsilon, bridges=args.bridges)
        d0.save(d0_path)
    log({"phase": "collect", "round": 0, "seconds": time.time() - t0, **quality_report(d0)})
    rms = dataset_stats(d0)
    np.savez(run / "obs_stats.npz", mean=rms.mean, var=rms.var)

    tmp_models = run / "models"
    summary: dict[str, Any] = {"expert": args.expert, "run": args.run, "versions": d0.meta["versions"], "data": {"d0": quality_report(d0)}, "rounds": []}

    # Expert reference on the same evaluation seeds.
    summary["expertEval"] = evaluate_policy_file(None, args.expert, args.eval_episodes)
    log({"phase": "eval", "policy": args.expert, **{k: v["survivalS"]["mean"] for k, v in summary["expertEval"].items()}})

    model, info = train_bc(d0, rms, epochs=args.epochs, seed=args.seed, log=log)
    model.save(run / "bc-r0.zip")
    f = export_bc(model, rms, f"{args.run}-bc", "BC (MPC 모방)", "bc", {"run": args.run, "round": 0, "samples": len(d0), "expert": args.expert}, tmp_models)
    ev = evaluate_policy_file(f, f"{args.run}-bc", args.eval_episodes)
    va = episode_split(d0, seed=args.seed)[1]
    r0 = {"round": 0, "samples": len(d0), "bc": info, "val": bc_metrics(model, d0, rms, va), "eval": ev, "seconds": time.time() - t0}
    summary["rounds"].append(r0)
    log({"phase": "round", "round": 0, "survival": ev["D-mixed"]["survivalS"]["mean"], "val": r0["val"]})
    best = (ev["D-mixed"]["survivalS"]["mean"], 0)
    data = d0
    for k in range(1, args.rounds + 1):
        acting = ActingPolicy(model, rms, stochastic=True, seed=args.seed + k)
        beta = 0.5 ** k
        dk = collect(args.expert, args.dagger_steps, args.seed * 1000 + 100 * k, policy=acting, beta=beta, source=k,
                     episode_base=int(data.episode.max()) + 1, bridges=args.bridges)
        dk.save(run / f"dataset-d{k}.npz")
        log({"phase": "collect", "round": k, "beta": beta, **quality_report(dk)})
        data = Dataset.concat([data, dk])
        model, info = train_bc(data, rms, model=model, epochs=max(3, args.epochs // 2), seed=args.seed + k, log=log)
        model.save(run / f"bc-r{k}.zip")
        f = export_bc(model, rms, f"{args.run}-dagger-r{k}", f"BC+DAgger r{k}", "dagger", {"run": args.run, "round": k, "samples": len(data), "expert": args.expert}, tmp_models)
        ev = evaluate_policy_file(f, f"{args.run}-dagger-r{k}", args.eval_episodes)
        va_k = episode_split(data, seed=args.seed + k)[1]
        rk = {"round": k, "beta": beta, "samples": len(data), "newSamples": len(dk), "bc": info, "val": bc_metrics(model, data, rms, va_k),
              "valOnD0": bc_metrics(model, d0, rms, va), "eval": ev, "seconds": time.time() - t0}
        summary["rounds"].append(rk)
        log({"phase": "round", "round": k, "survival": ev["D-mixed"]["survivalS"]["mean"], "val": rk["val"]})
        if ev["D-mixed"]["survivalS"]["mean"] > best[0]:
            best = (ev["D-mixed"]["survivalS"]["mean"], k)
    summary["bestRound"] = best[1]
    summary["seconds"] = time.time() - t0
    save_json(run / "summary.json", summary)
    save_json(REPO / "public" / "data" / "rl" / f"{args.run}.json", summary)
    print(f"done in {summary['seconds']:.0f}s; best DAgger round {best[1]} (survival {best[0]:.1f}s)")


if __name__ == "__main__":
    main()
