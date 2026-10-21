"""Environment contract, checker, determinism and cross-language rule-equivalence tests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from rl.bridge_client import NUM_ACTIONS, OBS_SIZE, Bridge, masks_from
from rl.envs import SerpentEnv, SerpentVecEnv
from rl.rules_reference import World as RefWorld

FIXTURES = Path(__file__).parent / "fixtures"


def test_contract_versions():
    with Bridge() as b:
        h = b.init(2, {"scenario": "A-food"}, 1)
        assert (h["rules"], h["obs"], h["action"], h["reward"]) == ("rules-v1", "obs-v1", "act-v1", "reward-v1")
        assert h["obsSize"] == OBS_SIZE == 84 and h["numActions"] == NUM_ACTIONS == 18


def test_gymnasium_and_sb3_checkers():
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check

    env = SerpentEnv({"scenario": "C-moving", "maxSteps": 50}, seed=3)
    try:
        gym_check(env, skip_render_check=True)
        sb3_check(env, warn=True)
    finally:
        env.close()


def test_observations_in_declared_space_and_masks():
    env = SerpentVecEnv({"scenario": "D-mixed", "maxSteps": 40}, num_bridges=2, envs_per_bridge=3, seed=11)
    try:
        obs = env.reset()
        assert obs.shape == (6, OBS_SIZE)
        rng = np.random.default_rng(0)
        finished = 0
        for _ in range(120):
            masks = env.action_masks()
            assert masks.shape == (6, NUM_ACTIONS) and masks[:, 0::2].all()
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            obs, rew, done, infos = env.step(acts)
            assert np.all(obs >= -1.0) and np.all(obs <= 2.0) and np.isfinite(obs).all()
            assert np.isfinite(rew).all()
            finished += int(done.sum())
            for inf in infos:
                if "episode_info" in inf:
                    assert inf["terminal_observation"].shape == (OBS_SIZE,)
        assert finished > 0
    finally:
        env.close()


def test_boost_mask_matches_engine_rule():
    # A learner at boostMinLength cannot boost: mask hides odd actions and a boost request runs as no-boost.
    assert masks_from([False])[0].tolist() == [i % 2 == 0 for i in range(NUM_ACTIONS)]
    with Bridge() as b:
        b.init(1, {"scenario": "A-food", "foodTarget": 0}, 5)
        b.reset([5])
        assert b.last_can_boost == [True]  # start length 240 > 160
        for _ in range(30):  # circle (turn 0.25) with boost for 3 s: 80 u of length burns in ~1.8 s
            _, _, done, _, _, _ = b.step([11])
            assert not done[0]
        assert b.last_can_boost == [False]
        b.step([11])
        assert b.last_taken == [10]


def test_same_seed_same_rollout():
    def rollout():
        with Bridge() as b:
            b.init(2, {"scenario": "D-mixed"}, 0)
            out = [b.reset([123, 456])]
            for t in range(60):
                out.append(b.step([(t * 5) % 18, (t * 7) % 18])[0])
            return np.stack(out)

    assert np.array_equal(rollout(), rollout())


def test_expert_labels_are_valid_and_do_not_change_the_episode():
    with Bridge() as a, Bridge() as b:
        a.init(1, {"scenario": "D-mixed"}, 0)
        b.init(1, {"scenario": "D-mixed", "expert": "mpc"}, 0)
        oa, ob = a.reset([77]), b.reset([77])
        for t in range(40):
            assert 0 <= b.last_experts[0] < NUM_ACTIONS
            oa = a.step([t % 18])[0]
            ob = b.step([t % 18])[0]
            assert np.array_equal(oa, ob)


def test_in_node_evaluation_runs_builtin_policies():
    with Bridge() as b:
        b.init(1, {"scenario": "A-food"}, 0)
        eps = b.evaluate("hybrid", {"scenario": "C-moving", "maxSteps": 100}, [900_001, 900_002])
        assert len(eps) == 2
        for e in eps:
            assert e["l"] <= 100 and sum(e["actions"]) == e["l"]


# ---------------------------------------------------------------- Method B: Python reference engine
POS_TOL = 1e-6


@pytest.mark.parametrize("name", ["kinematics", "food", "brawl"])
def test_python_reference_engine_matches_typescript(name):
    """Same initial state + same inputs → same heads, headings, bodies, lengths, food, deaths and rewards."""
    g = json.loads((FIXTURES / f"golden-{name}.json").read_text())
    sc = g["scenario"]
    w = RefWorld(sc["rules"], sc["seed"], fill_food=sc["fillFood"])
    assert [[fid, *p[:3]] for fid, p in w.food.items()] == [[f[0], pytest.approx(f[1], abs=1e-9), pytest.approx(f[2], abs=1e-9), f[3]] for f in g["initialFood"]]
    for sp in sc["spawns"]:
        w.add_snake(sp["x"], sp["y"], sp["heading"], sp.get("length"))
    R = g["reward"]
    for t, (controls, st) in enumerate(zip(g["controls"], g["states"])):
        before = [s.length if s.alive else 0 for s in w.snakes]
        w.step(controls)
        assert w.tick == st["tick"]
        for s, e, b0 in zip(w.snakes, st["snakes"], before):
            tag = f"{name} t={t} snake={s.id}"
            assert s.alive == e["alive"], tag
            assert s.death_cause == e["cause"] and s.killer == e["killer"] and s.kills == e["kills"], tag
            assert abs(s.length - e["length"]) < POS_TOL, tag
            if b0 > 0:
                died = next((d for d in w.deaths if d["id"] == s.id), None)
                dl = (died["length"] if died else s.length) - b0
                reward = R["length"] * dl + (R["death"] if died else R["alive"])
                assert abs(reward - e["reward"]) < 1e-9, tag
            if s.alive:
                assert abs(s.x - e["x"]) < POS_TOL and abs(s.y - e["y"]) < POS_TOL, tag
                assert abs(((s.heading - e["heading"] + np.pi) % (2 * np.pi)) - np.pi) < 1e-9, tag
                assert s.body.n == e["vertices"], tag
                assert abs(s.body.pts[0] - e["tailX"]) < POS_TOL and abs(s.body.pts[1] - e["tailY"]) < POS_TOL, tag
        assert len(w.food) == st["food"] and w.next_food_id == st["nextFoodId"], f"{name} t={t}"
        assert abs(sum(p[2] for p in w.food.values()) - st["foodValue"]) < 1e-6


def test_reference_engine_throughput_is_recorded():
    """Measures Method B speed (ticks/s) so the Method A/B comparison in docs is reproducible."""
    w = RefWorld({"foodTarget": 1200}, 1)
    for i in range(10):  # tight circles (radius ≈ 47 u) 600 u apart: nobody collides
        w.add_snake(500 + 600 * (i % 4), 600 + 700 * (i // 4), 0.0)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 1.0:
        w.step([{"turn": 1.0}] * 10)
        n += 1
    rate = n / (time.perf_counter() - t0)
    assert all(s.alive for s in w.snakes)
    print(f"\n[method B] python reference engine, 10 live snakes, 1200 pellets: {rate:.0f} ticks/s")
