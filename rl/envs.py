"""Gymnasium / Stable-Baselines3 environments backed by the Node engine bridge.

- SerpentEnv: a single gymnasium.Env (one bridge process, one env) — used by checkers and tests.
- SerpentVecEnv: an SB3 VecEnv over several bridge processes stepped in parallel; exposes
  `action_masks()` through env_method so sb3-contrib MaskablePPO can mask unavailable boosts.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl.bridge_client import NUM_ACTIONS, OBS_SIZE, Bridge, masks_from

OBS_LOW, OBS_HIGH = -1.0, 2.0

#: Seeds ≥ EVAL_SEED_BASE are reserved for evaluation and never used for training rollouts.
EVAL_SEED_BASE = 900_000
TRAIN_SEED_LIMIT = EVAL_SEED_BASE


def observation_space() -> spaces.Box:
    return spaces.Box(low=OBS_LOW, high=OBS_HIGH, shape=(OBS_SIZE,), dtype=np.float32)


def action_space() -> spaces.Discrete:
    return spaces.Discrete(NUM_ACTIONS)


class SerpentEnv(gym.Env):
    """Single-learner environment (obs-v1 → act-v1, reward-v1)."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any] | None = None, seed: int = 0):
        super().__init__()
        self.env_config = env_config or {"scenario": "D-mixed"}
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.bridge = Bridge()
        self.bridge.init(1, self.env_config, seed)
        self._next_seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._next_seed = int(seed)
        obs = self.bridge.reset([self._next_seed])
        self._next_seed += 1
        return obs[0].astype(np.float32), {"expert": self.bridge.last_experts[0]}

    def step(self, action):
        obs, rew, done, trunc, infos, terminals = self.bridge.step([int(action)])
        info: dict[str, Any] = {"taken": self.bridge.last_taken[0], "expert": self.bridge.last_experts[0]}
        if done[0]:
            ep = infos[0]
            info["episode_info"] = ep
            # The bridge auto-resets; the terminal observation is what this step produced.
            return terminals[ep["terminal_index"]].astype(np.float32), float(rew[0]), not bool(trunc[0]), bool(trunc[0]), info
        return obs[0].astype(np.float32), float(rew[0]), False, False, info

    def action_masks(self) -> np.ndarray:
        return masks_from(self.bridge.last_can_boost)[0]

    def close(self):
        self.bridge.close()


class SerpentVecEnv(VecEnv):
    """`num_bridges` Node processes × `envs_per_bridge` envs, stepped in parallel."""

    def __init__(self, env_config: dict[str, Any], num_bridges: int = 4, envs_per_bridge: int = 8, seed: int = 0):
        self.bridges = [Bridge() for _ in range(num_bridges)]
        self.envs_per_bridge = envs_per_bridge
        for i, b in enumerate(self.bridges):
            # Disjoint training seed ranges per bridge, all below EVAL_SEED_BASE.
            b.init(envs_per_bridge, env_config, (seed + i * 100_003) % TRAIN_SEED_LIMIT)
        self.env_config = env_config
        super().__init__(num_bridges * envs_per_bridge, observation_space(), action_space())
        self._actions: np.ndarray | None = None
        self._masks = np.ones((self.num_envs, NUM_ACTIONS), dtype=bool)
        self._experts = np.full(self.num_envs, -1, dtype=np.int64)
        self._taken = np.zeros(self.num_envs, dtype=np.int64)

    # ------------------------------------------------------------------ VecEnv API
    def reset(self):
        obs = [b.reset() for b in self.bridges]
        self._refresh()
        return np.concatenate(obs).astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(-1)
        k = self.envs_per_bridge
        for i, b in enumerate(self.bridges):
            b.step_send(self._actions[i * k:(i + 1) * k])

    def step_wait(self):
        all_obs, all_rew, all_done, infos = [], [], [], []
        for b in self.bridges:
            obs, rew, done, trunc, binfos, terminals = b.step_recv()
            all_obs.append(obs)
            all_rew.append(rew)
            all_done.append(done)
            for j, inf in enumerate(binfos):
                info: dict[str, Any] = {}
                if done[j]:
                    info["terminal_observation"] = terminals[inf["terminal_index"]].astype(np.float32)
                    info["TimeLimit.truncated"] = bool(trunc[j])
                    info["episode"] = {"r": inf["r"], "l": inf["l"], "t": inf["survivalS"]}
                    info["episode_info"] = inf
                infos.append(info)
        self._refresh()
        return np.concatenate(all_obs).astype(np.float32), np.concatenate(all_rew), np.concatenate(all_done), infos

    def _refresh(self) -> None:
        cb = [c for b in self.bridges for c in b.last_can_boost]
        self._masks = masks_from(cb)
        self._experts = np.asarray([e for b in self.bridges for e in b.last_experts], dtype=np.int64)
        self._taken = np.asarray([t for b in self.bridges for t in (b.last_taken or [0] * self.envs_per_bridge)], dtype=np.int64)

    def close(self) -> None:
        for b in self.bridges:
            b.close()

    def set_env_config(self, env_config: dict[str, Any]) -> None:
        self.env_config = env_config
        for b in self.bridges:
            b.set_env(env_config)

    def add_model(self, model: dict[str, Any]) -> None:
        for b in self.bridges:
            b.add_model(model)

    @property
    def experts(self) -> np.ndarray:
        """Expert labels for the current observations (requires env_config['expert'])."""
        return self._experts

    @property
    def taken(self) -> np.ndarray:
        return self._taken

    def action_masks(self) -> np.ndarray:
        return self._masks

    # Attribute/method plumbing required by VecEnv (and used by MaskablePPO's get_action_masks).
    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        idx = self._get_indices(indices)
        if attr_name == "action_masks":
            return [True for _ in idx]
        return [getattr(self, attr_name) for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        idx = list(self._get_indices(indices))
        if method_name == "action_masks":
            return [self._masks[i] for i in idx]
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def has_attr(self, attr_name: str) -> bool:
        return attr_name == "action_masks" or hasattr(self, attr_name)

    def _get_indices(self, indices) -> Iterable[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def seed(self, seed: int | None = None) -> Sequence[None]:
        return [None] * self.num_envs
