/// <reference lib="webworker" />
import type { ModelStore } from '../ai/registry';
import { MlpModel, type MlpModelFile } from '../policy/mlp';
import { buildFrame } from '../render/frame';
import { Match } from '../sim/match';
import { Pacer } from './pacing';
import type { FromWorker, GroundTruth, LeaderRow, ToWorker } from './protocol';
import { VIEW_RADIUS, senseWorld } from '../observation/sensors';
import { encodeObs } from '../observation/obs';
import { radiusFor } from '../sim/config';

declare const self: DedicatedWorkerGlobalScope;

let match: Match | null = null;
let focusId = 0;
let debug = false;
let running = false;
let awaitingAck = false;
const pacer = new Pacer(1, 30);
const trail: number[] = [];
let lastTickMs = 0;
let groundTruth = false;
let humanSeq = 0;
let gtBuffer: GroundTruth[] = [];

/** Exact state around the focus after a tick (only with ?gt=1). */
function captureGroundTruth(m: Match, appliedSeq: number): GroundTruth {
  const w = m.world, s = w.snakes[focusId];
  const R = 1.5 * VIEW_RADIUS;
  const food: number[] = [];
  w.queryFood(s.x - R, s.y - R, s.x + R, s.y + R, (p) => { food.push(p.x, p.y, p.r, p.value); });
  const sensors = s.alive ? senseWorld(w, s) : null;
  return {
    tick: w.tick, appliedSeq,
    self: { x: s.x, y: s.y, heading: s.heading, length: s.length, radius: radiusFor(w.rules, s.length), alive: s.alive, boosting: s.boosting, speed: s.speed },
    others: w.snakes.filter((o) => o.alive && o.id !== s.id && Math.hypot(o.x - s.x, o.y - s.y) < R)
      .map((o) => ({ id: o.id, x: o.x, y: o.y, heading: o.heading, speed: o.speed, length: o.length, radius: radiusFor(w.rules, o.length) })),
    food, sensors,
    obs: sensors ? Array.from(encodeObs(sensors, { turn: s.omega / w.rules.maxTurnRate, boost: s.boosting }, w.rules)) : null,
  };
}

function post(msg: FromWorker, transfer: Transferable[] = []): void {
  self.postMessage(msg, transfer);
}

function modelsFrom(files: MlpModelFile[]): ModelStore {
  const store: ModelStore = new Map();
  for (const f of files) store.set(f.id, new MlpModel(f));
  return store;
}

function sendFrame(): void {
  if (!match || awaitingAck) return;
  const m = match;
  const frame = buildFrame(m, focusId, { debug, trail });
  const leaders: LeaderRow[] = m.world.snakes
    .map((s) => ({ id: s.id, label: s.label, policy: m.agents[s.id].spec.policy, length: s.length, alive: s.alive, kills: m.agents[s.id].record.kills }))
    .sort((a, b) => Number(b.alive) - Number(a.alive) || b.length - a.length);
  awaitingAck = true;
  const gt = groundTruth ? gtBuffer : undefined;
  gtBuffer = [];
  post(
    { type: 'frame', frame, stats: { ups: running ? pacer.ups : 0, actualSpeed: running ? pacer.actualSpeed : 0, tickMs: lastTickMs, running }, leaders, records: m.agents.map((a) => a.record), gt },
    [frame.food.buffer as ArrayBuffer, ...frame.snakes.map((s) => s.body.buffer as ArrayBuffer)],
  );
}

function tick(n: number): number {
  if (!match) return 0;
  const t0 = performance.now();
  let done = 0;
  for (; done < n && !match.ended; done++) {
    const seq = humanSeq;
    match.step();
    if (groundTruth) gtBuffer.push(captureGroundTruth(match, seq));
    const f = match.world.snakes[focusId];
    if (f?.alive && match.tick % 3 === 0) {
      trail.push(f.x, f.y);
      if (trail.length > 2 * 120) trail.splice(0, 2);
    }
  }
  if (done > 0) lastTickMs = (performance.now() - t0) / done;
  if (match.ended) running = false;
  return done;
}

// Main loop: a MessageChannel ping keeps the worker responsive between slices.
const channel = new MessageChannel();
let lastSlice = performance.now();
channel.port1.onmessage = () => {
  const now = performance.now();
  if (running && match) {
    if (!Number.isFinite(pacer.speed)) {
      // As fast as possible, yielding every ~20 ms for messages and frames.
      const t0 = performance.now();
      let ran = 0;
      while (performance.now() - t0 < 20 && running) ran += tick(1);
      pacer.observe(ran, performance.now() - lastSlice || 1);
    } else {
      const n = pacer.due(now, 240);
      pacer.observe(n > 0 ? tick(n) : 0, now - lastSlice || 1);
    }
  }
  lastSlice = now;
  sendFrame();
  setTimeout(() => channel.port2.postMessage(0), running ? 4 : 30);
};
channel.port2.postMessage(0);

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  try {
    switch (m.type) {
      case 'init':
        match = new Match(m.config, modelsFrom(m.models));
        focusId = m.focusId;
        running = m.running;
        groundTruth = m.groundTruth === true;
        gtBuffer = [];
        trail.length = 0;
        pacer.reset(performance.now());
        awaitingAck = false;
        sendFrame();
        break;
      case 'load':
        match = Match.fromJSON(m.snapshot, modelsFrom(m.models));
        focusId = m.focusId;
        trail.length = 0;
        awaitingAck = false;
        sendFrame();
        break;
      case 'running':
        running = m.running && !!match && !match.ended;
        pacer.reset(performance.now());
        break;
      case 'speed':
        pacer.speed = m.speed;
        pacer.reset(performance.now());
        break;
      case 'step':
        running = false;
        tick(Math.max(1, Math.min(300, m.ticks ?? 1)));
        awaitingAck = false;
        sendFrame();
        break;
      case 'human':
        if (match) match.humanControl = m.control;
        if (m.seq !== undefined) humanSeq = m.seq;
        break;
      case 'focus':
        focusId = m.id;
        trail.length = 0;
        break;
      case 'debug':
        debug = m.enabled;
        break;
      case 'save':
        if (match) post({ type: 'saved', snapshot: match.toJSON() });
        break;
      case 'ack':
        awaitingAck = false;
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
