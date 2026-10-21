"""Vision pipeline and Vision Bridge tests (no ground truth is ever passed into the pipeline)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision.bridge.policy import MlpPolicy
from vision.bridge.runner import ESTOP_FILE, GuardedPage, GuardViolation, RunnerConfig, SafetyStop, Status, VisionRunner, check_target
from vision.detection.classical import detect, read_hud, length_from_gauge
from vision.evaluate import angle_err, to_screen
from vision.observation.encode import OBS_SIZE, Sensors, encode_obs
from vision.observation.from_vision import VisionObserver, ray_distances, wall_distances
from vision.tracking.tracker import Odometry, Tracker

FIX = Path(__file__).parent / "fixtures"
FRAMES = sorted(p for p in FIX.glob("frame-*.json"))


def load(p: Path):
    meta = json.loads(p.read_text())
    img = np.asarray(Image.open(p.with_suffix(".png")).convert("RGB"))
    return img, meta["gt"], meta["draw"]


# ---------------------------------------------------------------- observation contract
def test_python_encoder_matches_typescript_encoder():
    from rl.bridge_client import Bridge
    with Bridge() as b:
        b.init(2, {"scenario": "D-mixed"}, 0)
        b.reset([4242, 4343])
        for _ in range(30):
            b.step([8, 10])
        for i in range(2):
            fx = b.obs_fixture(i)
            py = encode_obs(Sensors.from_ts(fx["sensors"]), 0.0, False)
            assert np.abs(py - np.asarray(fx["obs"], dtype=np.float32)).max() < 1e-6


def test_ray_and_wall_geometry():
    m = np.zeros((400, 400), bool)
    m[:, 300:] = True                       # obstacle 100 px to the right of the centre (200, 200)
    d = ray_distances(m, 200, 200, 0.0, zoom=0.5)
    assert d[0] == pytest.approx(200, abs=2.5)   # 100 px / 0.5 px per u
    assert d[8] == 400                            # behind: nothing within range
    w = wall_distances((-math.inf, -math.inf, 350, math.inf), 200, 200, 0.0, zoom=0.5)
    assert w[0] == pytest.approx(300) and w[4] == 400  # forward hits x = 350; sideways never does


# ---------------------------------------------------------------- tracking & odometry
def test_tracker_estimates_speed_and_survives_short_occlusion():
    tr = Tracker()
    t = 0.0
    for k in range(12):
        t = k * 0.1
        meas = [] if k in (6, 7) else [(100 + 300 * t, 50.0, 30.0, 10.0, 0.0)]  # 300 u/s, occluded for 2 frames
        tr.update(t, meas)
    assert len(tr.tracks) == 1
    assert tr.tracks[0].speed == pytest.approx(300, rel=0.1)
    for k in range(12, 20):
        tr.update(k * 0.1, [])
    assert tr.tracks == []                  # lost after max_misses


def test_odometry_unwraps_grid_phase_with_motion_prior():
    od = Odometry()
    zoom = 0.8
    od.update((10.0, 20.0), zoom, np.zeros(2))
    # Camera moves +40 u in x: grid lines shift by −32 px (mod 80 px period).
    cam = od.update(((10.0 - 32.0) % 80, 20.0), zoom, np.array([38.0, 0.0]))
    assert cam[0] == pytest.approx(40.0, abs=1e-6) and cam[1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- detection on real game frames
@pytest.mark.skipif(not FRAMES, reason="no fixture frames")
@pytest.mark.parametrize("path", FRAMES, ids=[p.stem for p in FRAMES])
def test_detection_on_game_frames(path: Path):
    img, gt, draw = load(path)
    cam = draw["cam"]
    h, w = img.shape[:2]
    det = detect(img)
    assert det.ok, det.reason
    assert abs(det.zoom - cam["zoom"]) / cam["zoom"] < 0.02
    assert math.degrees(angle_err(det.self_heading, gt["self"]["heading"])) < 6
    assert abs(length_from_gauge(det.hud_fraction) - gt["self"]["length"]) / gt["self"]["length"] < 0.05
    # Every opponent head well inside the frame is found within one body radius.
    for o in gt["others"]:
        sx, sy = to_screen(cam, w, h, o["x"], o["y"])
        if 30 < sx < w - 30 and 30 < sy < h - 60:
            near = [hd for hd in det.heads if math.hypot(hd.x - sx, hd.y - sy) < max(4, o["radius"] * cam["zoom"])]
            assert near, f"head of snake {o['id']} at ({sx:.0f},{sy:.0f}) not detected"


@pytest.mark.skipif(not FRAMES, reason="no fixture frames")
def test_occluded_head_is_not_hallucinated_and_confidence_drops_without_eyes():
    def visible(p: Path):
        img, gt, draw = load(p)
        h, w = img.shape[:2]
        return [o for o in gt["others"] if 30 < to_screen(draw["cam"], w, h, o["x"], o["y"])[0] < w - 30 and 30 < to_screen(draw["cam"], w, h, o["x"], o["y"])[1] < h - 60]

    path = next((p for p in FRAMES if visible(p)), None)
    if path is None:
        pytest.skip("no fixture frame with an opponent head on screen")
    img, gt, draw = load(path)
    cam = draw["cam"]
    h, w = img.shape[:2]
    o = visible(path)[0]
    sx, sy = to_screen(cam, w, h, o["x"], o["y"])
    occ = img.copy()
    r = int(3 * o["radius"] * cam["zoom"])
    occ[int(sy) - r:int(sy) + r, int(sx) - r:int(sx) + r] = (128, 128, 128)  # grey patch over the head
    det = detect(occ)
    assert not [hd for hd in det.heads if math.hypot(hd.x - sx, hd.y - sy) < r]
    # Cover the own eyes: heading falls back to the body (neck) and the observation confidence drops.
    occ2 = img.copy()
    occ2[h // 2 - 6:h // 2 + 7, w // 2 - 6:w // 2 + 7] = 0
    d2 = detect(occ2)
    s_full, _ = VisionObserver().observe(detect(img), 0.0, 0.0, False)
    s_occ, _ = VisionObserver().observe(d2, 0.0, 0.0, False)
    if s_occ is not None:
        assert s_occ.confidence < s_full.confidence


def test_hud_reader_rejects_missing_hud():
    blank = np.full((720, 960, 3), 250, np.uint8)
    assert read_hud(blank) == (None, None)
    assert detect(blank).ok is False


# ---------------------------------------------------------------- Vision Bridge safety
def test_target_allowlist():
    check_target("http://127.0.0.1:4173/SerpentAI/?view=target")
    check_target("https://oyueo-mm.github.io/SerpentAI/?view=target")
    for bad in ["https://slither.io/", "http://example.com/SerpentAI/", "http://127.0.0.1:4173/other/", "file:///C:/SerpentAI/index.html"]:
        with pytest.raises(SafetyStop):
            check_target(bad)


class FakePage:
    """Records every call; serves fixture frames like a browser canvas would."""

    def __init__(self, frames: list[np.ndarray], url: str = "http://127.0.0.1:4173/SerpentAI/?view=target"):
        self.frames, self.url = frames, url
        self.calls: list[tuple] = []
        self.i = 0
        outer = self

        class _Mouse:
            def move(self, x, y):
                outer.calls.append(("move", x, y))

        class _Kb:
            def down(self, k):
                outer.calls.append(("down", k))

            def up(self, k):
                outer.calls.append(("up", k))

            def press(self, k):
                outer.calls.append(("press", k))

        class _Loc:
            def bounding_box(self):
                h, w = outer.frames[0].shape[:2]
                return {"x": 0, "y": 0, "width": w, "height": h}

        self.mouse, self.keyboard = _Mouse(), _Kb()
        self._loc = _Loc()

    def locator(self, sel):
        return self._loc

    def screenshot(self, **kw):
        import io
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        b = io.BytesIO()
        Image.fromarray(f).save(b, format="PNG")
        return b.getvalue()

    def evaluate(self, *a, **k):  # must never be reached through GuardedPage
        self.calls.append(("evaluate",))
        return None


@pytest.fixture
def policy(tmp_path: Path) -> MlpPolicy:
    rng = np.random.default_rng(0)
    layers = [{"w": rng.normal(0, 0.1, (32, OBS_SIZE)).tolist(), "b": [0.0] * 32, "act": "tanh"},
              {"w": rng.normal(0, 0.1, (18, 32)).tolist(), "b": [0.0] * 18, "act": "none"}]
    f = {"format": "serpent-mlp", "id": "t", "label": "t", "algo": "ppo", "obsVersion": "obs-v1", "actionVersion": "act-v1", "rulesVersion": "rules-v1",
         "obsMean": None, "obsVar": None, "obsEps": 1e-8, "obsClip": 10, "layers": layers, "meta": {}}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(f))
    return MlpPolicy(p)


def test_guarded_page_blocks_engine_access():
    g = GuardedPage(FakePage([np.zeros((10, 10, 3), np.uint8)]))
    with pytest.raises(GuardViolation):
        g.evaluate("() => window.__serpentGT")
    with pytest.raises(GuardViolation):
        _ = g.context


@pytest.mark.skipif(not FRAMES, reason="no fixture frames")
def test_runner_plays_from_pixels_only_and_releases_keys(policy: MlpPolicy):
    img, _, _ = load(FRAMES[0])
    page = FakePage([img] * 6)
    r = VisionRunner(GuardedPage(page), policy, RunnerConfig(max_seconds=0.6))
    s = r.run()
    assert s["steps"] >= 3
    assert not any(c[0] == "evaluate" for c in page.calls)
    assert any(c[0] == "move" for c in page.calls)
    downs = sum(1 for c in page.calls if c == ("down", "Space"))
    ups = sum(1 for c in page.calls if c == ("up", "Space"))
    assert downs == ups  # boost never left pressed


def test_runner_stops_when_the_game_cannot_be_seen(policy: MlpPolicy):
    blank = np.full((720, 960, 3), 245, np.uint8)
    r = VisionRunner(GuardedPage(FakePage([blank] * 20)), policy, RunnerConfig(max_seconds=10, max_detect_failures=4))
    s = r.run()
    assert "not detected" in s["endReason"]


@pytest.mark.skipif(not FRAMES, reason="no fixture frames")
def test_runner_stops_on_navigation_estop_and_control_endpoint(policy: MlpPolicy, tmp_path: Path):
    img, _, _ = load(FRAMES[0])
    page = FakePage([img] * 50)
    page.url = "https://example.com/"
    assert "not allowed" in VisionRunner(GuardedPage(page), policy, RunnerConfig(max_seconds=5)).run()["endReason"]
    ESTOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    ESTOP_FILE.write_text("stop")
    try:
        s = VisionRunner(GuardedPage(FakePage([img] * 50)), policy, RunnerConfig(max_seconds=5)).run()
        assert "emergency" in s["endReason"]
    finally:
        ESTOP_FILE.unlink()
    st = Status(command="estop")
    s = VisionRunner(GuardedPage(FakePage([img] * 50)), policy, RunnerConfig(max_seconds=5), status=st).run()
    assert st.state == "estopped" and "estop" in s["endReason"]
