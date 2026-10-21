import { MlpModel, checkRulesVersion, type MlpModelFile } from '../policy/mlp';

export interface ModelIndex {
  models: { file: string; id: string }[];
}

export interface LoadedModels {
  files: MlpModelFile[];
  errors: string[];
}

/**
 * Loads exported networks listed in public/models/index.json. Every file is validated (format, obs/action
 * contract, input/output sizes) before it can be used; failures are reported, never replaced.
 */
export async function loadModels(base: string): Promise<LoadedModels> {
  const errors: string[] = [];
  const files: MlpModelFile[] = [];
  let index: ModelIndex;
  try {
    const r = await fetch(`${base}models/index.json`, { cache: 'no-cache' });
    if (!r.ok) return { files, errors };
    index = (await r.json()) as ModelIndex;
  } catch {
    return { files, errors };
  }
  await Promise.all(index.models.map(async (m) => {
    try {
      const r = await fetch(`${base}models/${m.file}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const f = (await r.json()) as MlpModelFile;
      new MlpModel(f);
      const warn = checkRulesVersion(f);
      if (warn) throw new Error(warn);
      files.push(f);
    } catch (err) {
      errors.push(`${m.id}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }));
  const order = index.models.map((m) => m.id);
  files.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  return { files, errors };
}
