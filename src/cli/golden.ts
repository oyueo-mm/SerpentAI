// Writes golden trajectories of the TypeScript engine for the Python reference-engine equivalence test.
// node dist-node/golden.mjs rl/tests/fixtures
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { DEFAULT_REWARD } from '../../rl/bridge/envs';
import { Rng } from '../sim/rng';
import { World, type Control, type SnakeSpawn } from '../sim/world';

interface Scenario {
  name: string;
  seed: number;
  rules: Record<string, number>;
  fillFood: boolean;
  spawns: SnakeSpawn[];
  ticks: number;
  controlSeed: number;
  boostProb: number;
}

const SCENARIOS: Scenario[] = [
  { name: 'kinematics', seed: 1, rules: { foodTarget: 0 }, fillFood: false, ticks: 300, controlSeed: 5, boostProb: 0.3,
    spawns: [{ x: 1500, y: 1500, heading: 0.3, length: 500 }, { x: 600, y: 600, heading: 1.2 }] },
  { name: 'food', seed: 2, rules: { foodTarget: 900 }, fillFood: true, ticks: 600, controlSeed: 6, boostProb: 0.1,
    spawns: [{ x: 1200, y: 1300, heading: 0 }, { x: 1800, y: 1700, heading: 2.5 }, { x: 1500, y: 900, heading: 1.6 }] },
  { name: 'brawl', seed: 3, rules: { foodTarget: 1200, worldSize: 1400 }, fillFood: true, ticks: 900, controlSeed: 7, boostProb: 0.2,
    spawns: Array.from({ length: 10 }, (_, i) => ({ x: 300 + (i % 5) * 200, y: 400 + Math.floor(i / 5) * 500, heading: (i * 2.39) % 6.28, length: 240 + i * 40 })) },
];

const out = process.argv[2] ?? 'rl/tests/fixtures';
mkdirSync(out, { recursive: true });
for (const sc of SCENARIOS) {
  const w = new World(sc.rules, sc.seed, { fillFood: sc.fillFood });
  for (const sp of sc.spawns) w.addSnake(sp);
  const initialFood = [...w.food.values()].map((p) => [p.id, p.x, p.y, p.value]);
  const rng = new Rng(sc.controlSeed);
  const controls: Control[][] = [];
  const states: unknown[] = [];
  for (let t = 0; t < sc.ticks; t++) {
    const c = w.snakes.map(() => ({ turn: Math.round(rng.range(-1, 1) * 1000) / 1000, boost: rng.next() < sc.boostProb }));
    controls.push(c);
    const before = w.snakes.map((s) => (s.alive ? s.length : 0));
    w.step(c);
    states.push({
      tick: w.tick,
      snakes: w.snakes.map((s, i) => {
        const died = w.deaths.find((d) => d.id === s.id);
        // reward-v1 terms as the learner env computes them, for every snake.
        const dL = before[i] > 0 ? (died ? died.length : s.length) - before[i] : 0;
        const reward = before[i] > 0 ? DEFAULT_REWARD.length * dL + (died ? DEFAULT_REWARD.death : DEFAULT_REWARD.alive) : 0;
        return { alive: s.alive, x: s.x, y: s.y, heading: s.heading, length: s.length, vertices: s.body.vertexCount,
          tailX: s.alive ? s.body.tailX() : null, tailY: s.alive ? s.body.tailY() : null, kills: s.stats.kills, cause: s.deathCause, killer: s.killer, reward };
      }),
      food: w.food.size,
      foodValue: [...w.food.values()].reduce((a, p) => a + p.value, 0),
      nextFoodId: w.nextFoodId,
    });
  }
  writeFileSync(join(out, `golden-${sc.name}.json`), JSON.stringify({ scenario: sc, initialFood, controls, states, reward: DEFAULT_REWARD }));
  console.log(`golden-${sc.name}: ${sc.ticks} ticks, deaths ${w.snakes.filter((s) => !s.alive).length}, food ${w.food.size}`);
}
