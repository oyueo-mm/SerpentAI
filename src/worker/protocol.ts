import type { Frame } from '../render/frame';
import type { AgentRecord, MatchConfig, MatchJSON } from '../sim/match';
import type { MlpModelFile } from '../policy/mlp';
import type { Control } from '../sim/world';
import type { Sensors } from '../observation/sensors';

export type ToWorker =
  | { type: 'init'; config: MatchConfig; models: MlpModelFile[]; focusId: number; running: boolean; groundTruth?: boolean }
  | { type: 'load'; snapshot: MatchJSON; models: MlpModelFile[]; focusId: number }
  | { type: 'running'; running: boolean }
  | { type: 'speed'; speed: number }
  | { type: 'step'; ticks?: number }
  | { type: 'human'; control: Control; seq?: number }
  | { type: 'focus'; id: number }
  | { type: 'debug'; enabled: boolean }
  | { type: 'save' }
  | { type: 'ack' };

export interface LeaderRow {
  id: number;
  label: string;
  policy: string;
  length: number;
  alive: boolean;
  kills: number;
}

export interface SimStats {
  ups: number;
  actualSpeed: number;
  /** Mean wall time of one tick (ms) over the last slice. */
  tickMs: number;
  running: boolean;
}

/**
 * Exact engine state around the focused snake, published only when the page is opened with ?gt=1.
 * It is for evaluation and calibration logs; the Vision policy path never reads it.
 */
export interface GroundTruth {
  tick: number;
  /** Sequence number of the human control that was applied during this tick. */
  appliedSeq: number;
  self: { x: number; y: number; heading: number; length: number; radius: number; alive: boolean; boosting: boolean; speed: number };
  others: { id: number; x: number; y: number; heading: number; speed: number; length: number; radius: number }[];
  /** Pellets within 1.5 × VIEW_RADIUS: [x, y, r, value] × n. */
  food: number[];
  sensors: Sensors | null;
  obs: number[] | null;
}

export type FromWorker =
  | { type: 'frame'; frame: Frame; stats: SimStats; leaders: LeaderRow[]; records: AgentRecord[]; gt?: GroundTruth[] }
  | { type: 'saved'; snapshot: MatchJSON }
  | { type: 'error'; message: string };
