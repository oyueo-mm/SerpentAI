import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import type { ModelStore } from '../ai/registry';
import { MlpModel, type MlpModelFile } from '../policy/mlp';

/** Loads every *.json serpent-mlp file in a directory (skips index/manifest files). */
export function loadModelDir(dir: string): ModelStore {
  const store: ModelStore = new Map();
  if (!existsSync(dir)) return store;
  for (const f of readdirSync(dir)) {
    if (!f.endsWith('.json') || f === 'index.json') continue;
    const file = JSON.parse(readFileSync(join(dir, f), 'utf8')) as MlpModelFile;
    if (file.format !== 'serpent-mlp') continue;
    store.set(file.id, new MlpModel(file));
  }
  return store;
}
