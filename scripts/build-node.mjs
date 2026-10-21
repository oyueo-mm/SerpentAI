// Bundles the TypeScript engine for Node: the RL bridge (driven by Python) and the CLI tools.
import { build } from 'esbuild';

const common = { bundle: true, platform: 'node', format: 'esm', target: 'node20', sourcemap: true, logLevel: 'warning' };
await Promise.all([
  build({ ...common, entryPoints: ['rl/bridge/bridge.ts'], outfile: 'rl/bridge/dist/bridge.mjs' }),
  build({ ...common, entryPoints: ['src/cli/tournament.ts'], outfile: 'dist-node/tournament.mjs' }),
  build({ ...common, entryPoints: ['src/cli/benchmark.ts'], outfile: 'dist-node/benchmark.mjs' }),
]);
console.log('built rl/bridge/dist/bridge.mjs, dist-node/tournament.mjs, dist-node/benchmark.mjs');
