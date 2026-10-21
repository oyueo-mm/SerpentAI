/// <reference lib="webworker" />
import type { ModelStore } from '../ai/registry';
import { MlpModel, type MlpModelFile } from '../policy/mlp';
import { runRound, summarize, type RoundResult, type TournamentConfig } from '../sim/tournament';

declare const self: DedicatedWorkerGlobalScope;

/** Runs a tournament off the main thread, posting progress after every round. */
self.onmessage = (e: MessageEvent<{ config: TournamentConfig; models: MlpModelFile[] }>) => {
  const { config, models } = e.data;
  const store: ModelStore = new Map(models.map((f) => [f.id, new MlpModel(f)]));
  const rounds: RoundResult[] = [];
  const t0 = performance.now();
  for (let i = 0; i < config.rounds; i++) {
    rounds.push(runRound(config, i, store));
    self.postMessage({ type: 'progress', done: i + 1, total: config.rounds, summary: summarize(config.policies, rounds), wallMs: performance.now() - t0 });
  }
  self.postMessage({ type: 'done' });
};
