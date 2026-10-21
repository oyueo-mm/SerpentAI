import type { MlpModelFile } from '../policy/mlp';
import { t } from '../i18n';

export function LabView({ models }: { models: MlpModelFile[] }) {
  return <div className="p-6 text-sm text-slate-500">{t.common.notMeasured} ({models.length})</div>;
}
