"""Learning-pipeline tests: supervisor logic, BC/DAgger updates, PPO updates, checkpoints and the
Python↔TypeScript inference parity of exported models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch as th

from rl.bridge_client import NUM_ACTIONS, OBS_SIZE, Bridge
from rl.common import RunningStats, export_model, logits_np, new_model, normalize, vecnormalize_for
from rl.envs import SerpentVecEnv
from rl.imitate import ActingPolicy, Dataset, blank_model, bc_metrics, collect, dataset_stats, episode_split, train_bc
from rl.supervisor import Supervisor, SupervisorCallback, SupervisorConfig


def ep(**kw):
    base = dict(r=5.0, l=600, survivalS=60.0, finalLength=500, maxLength=600, eaten=300.0, kills=0, death="body", boostS=3.0,
                seed=1, scenario="D-mixed", pathLen=8000.0, netDisp=900.0, cells=60, wallNearFrac=0.05, spinFrac=0.05, killsBy={"body": 0, "headon": 0})
    base.update(kw)
    return base


# ---------------------------------------------------------------- supervisor
def test_supervisor_flags_behavioural_anomalies():
    s = Supervisor()
    assert s.check_episodes(1, [ep() for _ in range(20)]) == []
    assert "spinning_in_place" in s.check_episodes(2, [ep(spinFrac=0.9, cells=3) for _ in range(20)])
    assert "wall_hugging" in s.check_episodes(3, [ep(wallNearFrac=0.6) for _ in range(20)])
    assert "pointless_boost" in s.check_episodes(4, [ep(boostS=50.0, eaten=10.0) for _ in range(20)])
    assert "passive_ignores_food" in s.check_episodes(5, [ep(survivalS=120.0, eaten=5.0, death=None) for _ in range(20)])
    assert "implausible_reward_rate" in s.check_episodes(6, [ep(r=500.0, l=100)])
    assert "headon_rule_farming" in s.check_episodes(7, [ep(killsBy={"body": 0, "headon": 2}) for _ in range(6)])
    counts = np.zeros(NUM_ACTIONS); counts[8] = 100
    assert "action_collapse" in s.check_episodes(8, [ep()], counts)
    assert s.check_generalisation(9, 10.0, 4.0) == ["opponent_specific"]


def test_supervisor_intervention_sequence(tmp_path: Path):
    s = Supervisor(SupervisorConfig(tolerance=0.05, patience_lr=2, patience_rollback=2), tmp_path / "sup.jsonl")
    kinds = []
    lr, kl = 3e-4, 0.03
    for step, score in enumerate([1.0, 2.0, 3.0, 2.9, 2.0, 1.9, 1.8, 1.7]):
        ivs = s.observe_validation(step, score, lr, kl)
        for iv in ivs:
            if iv.kind == "reduce_lr":
                lr, kl = iv.data["lr"], iv.data["target_kl"]
        kinds.append([iv.kind for iv in ivs])
    assert kinds[2] == ["new_best"]
    assert kinds[3] == []  # 2.9 is within 5 % of 3.0
    assert kinds[4] == ["warning", "preserve"]
    assert kinds[5] == ["warning", "preserve", "reduce_lr"]
    assert lr == pytest.approx(1.5e-4) and kl == pytest.approx(0.015)
    assert kinds[7] == ["warning", "preserve", "rollback"]
    logged = [json.loads(l) for l in (tmp_path / "sup.jsonl").read_text().splitlines()]
    assert any(r.get("action") == "rollback" and r["to_step"] == 2 for r in logged)


def test_supervisor_callback_really_changes_lr_and_rolls_back(tmp_path: Path):
    """End-to-end on a tiny PPO: a degrading validator must cut the LR and restore the best weights."""
    venv = SerpentVecEnv({"scenario": "A-food", "maxSteps": 30}, num_bridges=1, envs_per_bridge=2, seed=5)
    try:
        model = new_model(vecnormalize_for(venv, None, training=False), 0, n_steps=32, batch_size=64, n_epochs=1, learning_rate=1e-3)
        scores = iter([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1])
        sup = Supervisor(SupervisorConfig(tolerance=0.01, patience_lr=2, patience_rollback=2), tmp_path / "s.jsonl")
        snapshots: list[dict] = []
        cb = SupervisorCallback.make(sup, lambda m, step: (next(scores), {}), eval_every=64, snapshot_dir=tmp_path / "pres",
                                     on_new_best=lambda m, step: snapshots.append({k: v.clone() for k, v in m.policy.state_dict().items()}))
        model.learn(64 * 6, callback=cb)
        acts = [e["action"] for e in sup.events if e["kind"] == "intervention"]
        assert "reduce_lr" in acts and "rollback" in acts
        assert model.learning_rate == pytest.approx(5e-4)
        assert list((tmp_path / "pres").glob("preserved-*.zip"))
        # After the rollback the weights equal the best snapshot (later updates may follow, so compare at rollback time).
        rb = [e for e in sup.events if e.get("action") == "rollback"][0]
        assert rb["to_step"] == sup.best_step
        assert len(snapshots) == 1
    finally:
        venv.close()


# ---------------------------------------------------------------- imitation
@pytest.fixture(scope="module")
def small_dataset() -> Dataset:
    return collect("hybrid", 2400, 4, epsilon=0.1, bridges=2, per_bridge=4, mix={"C-moving": 0.5, "D-mixed": 0.5})


def test_dataset_records_required_fields(small_dataset: Dataset):
    d = small_dataset
    assert d.obs.shape[1] == OBS_SIZE and len(d) >= 2000
    assert set(d.meta["versions"]) == {"rules", "obs", "action", "reward"} and d.meta["expert"] == "hybrid"
    assert (d.label >= 0).all() and (d.label < NUM_ACTIONS).all()
    assert d.mask.shape == (len(d), NUM_ACTIONS)
    # ε-noise makes the executed action differ from the label on some states; labels stay the expert's.
    assert 0.0 < (d.label != d.taken).mean() < 0.3
    # Every episode id maps to exactly one seed.
    for e in np.unique(d.episode)[:20]:
        assert len(np.unique(d.seed[d.episode == e])) == 1
    tr, va = episode_split(d)
    assert not set(d.episode[tr]) & set(d.episode[va])


def test_bc_update_learns_the_expert(small_dataset: Dataset):
    rms = dataset_stats(small_dataset)
    _, va = episode_split(small_dataset)
    model0 = blank_model(0)
    before = bc_metrics(model0, small_dataset, rms, va)["valAccuracy"]
    model, info = train_bc(small_dataset, rms, model=model0, epochs=6, seed=0)
    after = info["history"][-1]["valAccuracy"]
    assert info["history"][-1]["trainLoss"] < info["history"][0]["trainLoss"]
    assert after > before + 0.1 and after > 2.0 / NUM_ACTIONS


def test_dagger_collection_labels_policy_visited_states(small_dataset: Dataset):
    rms = dataset_stats(small_dataset)
    model, _ = train_bc(small_dataset, rms, epochs=2, seed=0)
    d1 = collect("hybrid", 800, 9, policy=ActingPolicy(model, rms, seed=1), beta=0.0, source=1, episode_base=10_000, bridges=1, per_bridge=4, mix={"D-mixed": 1.0})
    assert (d1.source == 1).all() and d1.episode.min() >= 10_000
    # The learner acted (taken) while the expert labelled (label): they disagree on a share of states.
    assert (d1.label != d1.taken).mean() > 0.05


# ---------------------------------------------------------------- PPO, checkpoints, export parity
def test_ppo_update_changes_parameters_and_checkpoint_roundtrip(tmp_path: Path):
    venv = SerpentVecEnv({"scenario": "C-moving", "maxSteps": 40}, num_bridges=1, envs_per_bridge=4, seed=2)
    try:
        model = new_model(vecnormalize_for(venv, None, training=False), 0, n_steps=32, batch_size=64, n_epochs=2)
        p0 = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(128)
        changed = sum(not th.equal(a, b) for a, b in zip(p0, model.policy.parameters()))
        assert changed > 0
        model.save(tmp_path / "m.zip")
        from sb3_contrib import MaskablePPO
        m2 = MaskablePPO.load(tmp_path / "m.zip", device="cpu")
        x = np.random.default_rng(0).normal(size=(16, OBS_SIZE)).astype(np.float32)
        assert np.allclose(logits_np(model.policy, x), logits_np(m2.policy, x))
    finally:
        venv.close()


def test_exported_model_matches_typescript_inference(tmp_path: Path):
    rng = np.random.default_rng(1)
    model = blank_model(3)
    for p in model.policy.parameters():  # non-trivial weights
        p.data.normal_(0, 0.3)
    rms = RunningStats(rng.normal(0.3, 0.2, OBS_SIZE), rng.uniform(0.01, 0.2, OBS_SIZE))
    f = export_model(model.policy, rms, "parity-test", "parity", "ppo", {}, out_dir=tmp_path)
    obs = rng.uniform(-1, 2, size=(32, OBS_SIZE)).astype(np.float32)
    py = logits_np(model.policy, normalize(obs, rms))
    with Bridge() as b:
        b.init(1, {"scenario": "A-food"}, 0)
        b.add_model(json.loads(f.read_text()))
        ts = np.asarray(b.call("logits", model="parity-test", obs=obs.astype(float).tolist())[0]["logits"])
    assert np.abs(py - ts).max() < 1e-4
    assert (py.argmax(1) == ts.argmax(1)).all()


def test_rebuilt_supervisor_callback_keeps_the_validation_spacing(tmp_path: Path):
    """Regression: a callback rebuilt for a new curriculum stage must not fire a burst of validations."""
    venv = SerpentVecEnv({"scenario": "A-food", "maxSteps": 30}, num_bridges=1, envs_per_bridge=2, seed=6)
    try:
        model = new_model(vecnormalize_for(venv, None, training=False), 0, n_steps=32, batch_size=64, n_epochs=1)
        model.learn(64 * 5)  # stage 1 without validation: num_timesteps = 320
        calls: list[int] = []
        cb = SupervisorCallback.make(Supervisor(), lambda m, step: (calls.append(step), (1.0, {}))[1], eval_every=128, snapshot_dir=tmp_path)
        model.learn(64 * 4, callback=cb, reset_num_timesteps=False)  # 320 → 576
        assert calls == [384, 512]
    finally:
        venv.close()
