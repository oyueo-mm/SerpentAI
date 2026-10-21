"""Vision Runner: plays the SerpentAI web game in a separate browser using only screen pixels.

    python -m vision.bridge.runner --model public/models/ppo-dagger-v1.json --seed 900123 --mode realtime --record vision/sessions/demo

Path of information (and nothing else):
    browser canvas pixels → classical CV → VisionObserver → obs-v1 → MLP → mouse heading / Space key → game

Safety (docs/VISION.md §Vision Bridge): the target URL must be the SerpentAI game (allowlist); the loop
stops on navigation away, repeated detection failure, a confidence collapse, frame timeouts, slow
inference, game over, Ctrl+C, the emergency-stop file or the local control endpoint.

The page handed to the control loop is a GuardedPage: it exposes screenshots, the canvas rectangle and
official inputs only. Reading engine state (page.evaluate, the GT log) raises GuardViolation. The
optional calibration recorder reads the ground-truth log once, after the loop has ended.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from vision.bridge.policy import MlpPolicy, decode
from vision.detection.classical import detect
from vision.observation.encode import RULES, encode_obs
from vision.observation.from_vision import VisionObserver

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
ALLOWED_REMOTE = ("https://oyueo-mm.github.io/SerpentAI/",)
ESTOP_FILE = Path("vision/EMERGENCY_STOP")
DECISION_S = 0.1   # act-v1 decision period (3 ticks at 30 Hz)


class GuardViolation(RuntimeError):
    pass


class SafetyStop(RuntimeError):
    pass


def check_target(url: str) -> None:
    """Only the user's own SerpentAI game may be controlled."""
    u = urlparse(url)
    if url.startswith(ALLOWED_REMOTE):
        return
    if u.scheme in ("http", "https") and u.hostname in ALLOWED_HOSTS and "/SerpentAI/" in u.path:
        return
    raise SafetyStop(f"target not allowed: {url} (only the local SerpentAI build or {ALLOWED_REMOTE[0]})")


class GuardedPage:
    """The only view of the browser the control loop gets: pixels in, official inputs out."""

    _ALLOWED = {"capture", "move_pointer", "key_down", "key_up", "press", "url", "canvas_box"}

    def __init__(self, page):
        object.__setattr__(self, "_page", page)
        object.__setattr__(self, "_box", None)

    def __getattr__(self, name: str):
        raise GuardViolation(f"control loop tried to access page.{name}: only {sorted(self._ALLOWED)} are allowed")

    def url(self) -> str:
        return self._page.url

    def canvas_box(self) -> dict[str, float]:
        if self._box is None:
            box = self._page.locator("canvas").bounding_box()
            if not box:
                raise SafetyStop("game canvas not found")
            object.__setattr__(self, "_box", box)
        return self._box

    def capture(self) -> np.ndarray:
        b = self.canvas_box()
        png = self._page.screenshot(clip={"x": b["x"], "y": b["y"], "width": b["width"], "height": b["height"]}, type="png", timeout=2000)
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def move_pointer(self, x: float, y: float) -> None:
        b = self.canvas_box()
        self._page.mouse.move(b["x"] + x, b["y"] + y)

    def key_down(self, key: str) -> None:
        self._page.keyboard.down(key)

    def key_up(self, key: str) -> None:
        self._page.keyboard.up(key)

    def press(self, key: str) -> None:
        self._page.keyboard.press(key)


@dataclass
class Status:
    state: str = "idle"          # idle | running | stopped | estopped
    reason: str = ""
    step: int = 0
    last: dict[str, Any] = field(default_factory=dict)
    command: str | None = None   # set by the control endpoint: start | stop | estop


def serve_control(status: Status, port: int) -> ThreadingHTTPServer:
    """Local control endpoint (127.0.0.1 only): GET /status, POST /start, /stop, /estop."""

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"state": status.state, "reason": status.reason, "step": status.step, "last": status.last})

        def do_POST(self):
            cmd = self.path.strip("/")
            if cmd in ("start", "stop", "estop"):
                status.command = cmd
                self._send(200, {"ok": True, "command": cmd})
            else:
                self._send(404, {"ok": False})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@dataclass
class RunnerConfig:
    mode: str = "realtime"            # realtime | lockstep
    max_seconds: float = 130.0
    max_steps: int = 0                # stop after this many control steps (0 = no limit); lockstep: 1200 = 120 s game time
    max_detect_failures: int = 8
    min_confidence: float = 0.3
    low_conf_frames: int = 10
    max_step_ms: float = 400.0
    max_slow_steps: int = 5
    aim_px: float = 160.0             # pointer distance from the head
    record_every: int = 1             # save every k-th frame (0 = no frames)
    jpeg_quality: int = 80


class VisionRunner:
    def __init__(self, page: GuardedPage, policy: MlpPolicy, cfg: RunnerConfig, record: Path | None = None, status: Status | None = None):
        self.page, self.policy, self.cfg = page, policy, cfg
        self.record = record
        self.status = status or Status()
        self.obs = VisionObserver()
        self.boost_held = False
        self.prev_turn, self.prev_boost = 0.0, False
        self.log: list[dict[str, Any]] = []

    def release(self) -> None:
        """Safe state: release boost, stop steering (pointer straight ahead of the last heading)."""
        try:
            if self.boost_held:
                self.page.key_up("Space")
                self.boost_held = False
        except Exception:
            pass

    def apply(self, heading: float, turn: float, boost: bool, w: int, h: int) -> float:
        # act-v1 turn level u = fraction of maxTurnRate held for one decision: aim at θ + u·ω_max·Δt.
        target = heading + turn * RULES["maxTurnRate"] * DECISION_S
        t0 = time.time()
        self.page.move_pointer(w / 2 + math.cos(target) * self.cfg.aim_px, h / 2 + math.sin(target) * self.cfg.aim_px)
        if boost != self.boost_held:
            (self.page.key_down if boost else self.page.key_up)("Space")
            self.boost_held = boost
        return t0

    def check_stop(self) -> None:
        if ESTOP_FILE.exists():
            self.status.state = "estopped"
            raise SafetyStop("emergency stop file present")
        if self.status.command in ("stop", "estop"):
            self.status.state = "estopped" if self.status.command == "estop" else "stopped"
            raise SafetyStop(f"{self.status.command} requested via control endpoint")
        check_target(self.page.url())

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        if self.record:
            (self.record / "frames").mkdir(parents=True, exist_ok=True)
        fails = low_conf = slow = 0
        t_start = time.time()
        self.status.state, self.status.reason = "running", ""
        end_reason = "max_seconds"
        try:
            check_target(self.page.url())
            while time.time() - t_start < cfg.max_seconds:
                if cfg.max_steps and self.status.step >= cfg.max_steps:
                    end_reason = "max_steps"
                    break
                self.check_stop()
                step_t0 = time.time()
                t_cap0 = time.time()
                try:
                    frame = self.page.capture()
                except Exception as err:
                    raise SafetyStop(f"frame capture failed: {err}")
                t_cap1 = time.time()
                if t_cap1 - t_cap0 > 1.5:
                    raise SafetyStop(f"frame capture took {t_cap1 - t_cap0:.2f}s")
                h, w = frame.shape[:2]
                det = detect(frame)
                t_det = time.time()
                # Game over: the GAME OVER overlay darkens the whole canvas.
                if float(frame.mean()) < 90 and not det.ok:
                    end_reason = "game_over"
                    break
                sensors, info = self.obs.observe(det, t_cap0, self.prev_turn, self.prev_boost, frame_age=0.0)
                rec: dict[str, Any] = {"step": self.status.step, "tCap": t_cap0 * 1000, "captureMs": (t_cap1 - t_cap0) * 1000, "detectMs": (t_det - t_cap1) * 1000,
                                       "ok": det.ok, "reason": det.reason, "zoom": det.zoom, "zoomConf": det.zoom_conf, "headingSrc": det.self_heading_src,
                                       "heading": det.self_heading, "heads": len(det.heads), "pellets": len(det.pellets), "hud": det.hud_fraction,
                                       "headList": [[round(hd.x, 1), round(hd.y, 1), round(hd.heading, 4), round(hd.radius_px, 2)] for hd in det.heads],
                                       "w": w, "h": h, "cam": info.get("cam"), "obsZoom": info.get("zoom")}
                if sensors is None:
                    fails += 1
                    self.release()
                    rec["action"] = None
                    self.log.append(rec)
                    if fails >= cfg.max_detect_failures:
                        raise SafetyStop(f"game area / own snake not detected for {fails} frames ({det.reason})")
                    self._pace(step_t0)
                    continue
                fails = 0
                low_conf = low_conf + 1 if sensors.confidence < cfg.min_confidence else 0
                if low_conf >= cfg.low_conf_frames:
                    raise SafetyStop(f"observation confidence below {cfg.min_confidence} for {low_conf} frames")
                sensors.frame_age = time.time() - t_cap0
                obs = encode_obs(sensors, self.prev_turn, self.prev_boost)
                can_boost = det.can_boost if det.can_boost is not None else sensors.length > RULES["boostMinLength"]
                a, probs = self.policy.act(obs, can_boost)
                t_pol = time.time()
                turn, boost = decode(a)
                t_send = self.apply(float(det.self_heading), turn, boost, w, h)
                t_done = time.time()
                step_ms = (t_done - step_t0) * 1000
                slow = slow + 1 if step_ms > cfg.max_step_ms else 0
                if slow >= cfg.max_slow_steps:
                    raise SafetyStop(f"control step slower than {cfg.max_step_ms} ms for {slow} steps")
                self.prev_turn, self.prev_boost = turn, boost
                rec.update({"policyMs": (t_pol - t_det) * 1000, "inputMs": (t_done - t_send) * 1000, "stepMs": step_ms, "tSend": t_send * 1000,
                            "action": a, "turn": turn, "boost": boost, "probs": np.round(probs, 4).tolist(), "obs": np.round(obs, 4).tolist(),
                            "conf": sensors.confidence, "length": sensors.length})
                self.log.append(rec)
                self.status.step += 1
                self.status.last = {k: rec[k] for k in ("step", "captureMs", "detectMs", "policyMs", "stepMs", "action", "conf", "length", "headingSrc")}
                if self.record and cfg.record_every and self.status.step % cfg.record_every == 0:
                    Image.fromarray(frame).save(self.record / "frames" / f"{self.status.step:05d}.jpg", quality=cfg.jpeg_quality)
                if cfg.mode == "lockstep":
                    self.page.press("]")   # official "advance one decision" key while paused
                    time.sleep(0.03)
                else:
                    self._pace(step_t0)
            self.status.state = "stopped"
        except SafetyStop as err:
            end_reason = str(err)
            if self.status.state == "running":
                self.status.state = "stopped"
        except KeyboardInterrupt:
            end_reason = "keyboard interrupt"
            self.status.state = "stopped"
        finally:
            self.release()
        self.status.reason = end_reason
        summary = {"steps": self.status.step, "seconds": time.time() - t_start, "endReason": end_reason, "mode": cfg.mode, "model": self.policy.id,
                   "tracker": self.obs.tracker.stats}
        if self.record:
            with (self.record / "log.jsonl").open("w", encoding="utf-8") as f:
                for r in self.log:
                    f.write(json.dumps(r, default=float) + "\n")
            (self.record / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        return summary

    def _pace(self, step_t0: float) -> None:
        left = DECISION_S - (time.time() - step_t0)
        if left > 0:
            time.sleep(left)


def target_url(base: str, seed: int, bots: str, theme: str, mode: str, gt: bool) -> str:
    q = f"view=target&seed={seed}&bots={bots}&theme={theme}&selfRespawn=0&autostart={'0' if mode == 'lockstep' else '1'}"
    return f"{base}?{q}{'&gt=1' if gt else ''}"


DEFAULT_BOTS = "greedy:2,survival:2,aggressive:2,potential:2,hybrid:2,mpc-lite:1"


def run_session(model: str, seed: int, mode: str = "realtime", theme: str = "default", bots: str = DEFAULT_BOTS, record: Path | None = None,
                gt_record: bool = False, base: str = "http://127.0.0.1:4173/SerpentAI/", headless: bool = True, max_seconds: float = 130.0,
                size: tuple[int, int] = (960, 720), control_port: int | None = None, wait_start: bool = False) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    url = target_url(base, seed, bots, theme, mode, gt_record)
    check_target(url)
    policy = MlpPolicy(model)
    status = Status()
    srv = serve_control(status, control_port) if control_port else None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1)
        page = ctx.new_page()
        page.goto(url)
        page.wait_for_selector("canvas")
        page.wait_for_timeout(800)
        guarded = GuardedPage(page)
        if mode == "lockstep":
            guarded.press("]")
            time.sleep(0.2)
        if wait_start:
            status.state = "armed"
            print("armed: waiting for Start (POST http://127.0.0.1:%d/start) ..." % control_port, flush=True)
            while status.command not in ("start", "stop", "estop"):
                time.sleep(0.1)
            if status.command != "start":
                browser.close()
                return {"steps": 0, "endReason": f"{status.command} before start"}
            status.command = None
        runner = VisionRunner(guarded, policy, RunnerConfig(mode=mode, max_seconds=max_seconds, max_steps=1200 if mode == "lockstep" else 0), record, status)
        summary = runner.run()
        # Calibration only: the ground-truth log is read once the control loop has finished.
        if gt_record and record:
            gt = page.evaluate("() => window.__serpentGT")
            (record / "groundtruth.json").write_text(json.dumps(gt))
            summary["groundTruthFrames"] = len(gt["frames"])
        summary.update({"url": url, "seed": seed, "theme": theme, "bots": bots, "size": list(size)})
        if record:
            (record / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        browser.close()
    if srv:
        srv.shutdown()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=900_123)
    ap.add_argument("--mode", choices=["realtime", "lockstep"], default="realtime")
    ap.add_argument("--theme", default="default")
    ap.add_argument("--bots", default=DEFAULT_BOTS)
    ap.add_argument("--base", default="http://127.0.0.1:4173/SerpentAI/")
    ap.add_argument("--record", default=None)
    ap.add_argument("--gt-record", action="store_true", help="after the run, save the page's ground-truth log for calibration")
    ap.add_argument("--headed", action="store_true", help="show the controlled browser window")
    ap.add_argument("--max-seconds", type=float, default=130.0)
    ap.add_argument("--control-port", type=int, default=8765)
    ap.add_argument("--wait-start", action="store_true", help="arm and wait for Start from the control endpoint / web panel")
    args = ap.parse_args()
    rec = Path(args.record) if args.record else None
    s = run_session(args.model, args.seed, args.mode, args.theme, args.bots, rec, args.gt_record, args.base, not args.headed, args.max_seconds,
                    control_port=args.control_port, wait_start=args.wait_start and bool(args.control_port))
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
