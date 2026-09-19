import type { BrainData } from "./brainPanel";

/** Human-readable names for the display groups (flyseek/connectome/layout.py DISPLAY_GROUPS). */
export const CIRCUIT_LABELS: Record<string, string> = {
  navigation: "Compass & goal · EPG FC2 PFL3",
  steering: "Steering DNs · DNa02 DNa03",
  pursuit: "Pursuit · LC10a",
  looming_escape: "Looming & escape · LC4 LPLC2 DNp01",
  olfaction: "Smell · danger meter / pings",
  locomotion: "Walking DNs",
  vision: "Photoreceptors",
};

/** Display order: the pathways Phase 4–5 showed carry the behaviour come first. */
export const CIRCUIT_ORDER = ["navigation", "steering", "pursuit", "looming_escape", "olfaction", "locomotion", "vision"];

export interface CircuitLevel {
  key: string;
  label: string;
  color: string;
  level: number; // 0..1 display level
  rateHz: number; // mean per-neuron firing rate over the smoothing window
  everActive: boolean;
}

/**
 * Per-fly activity of each named circuit: spikes per neuron per second, smoothed with an
 * exponential window, then squashed to 0..1 for the bar (1 - exp(-rate / refHz)).
 * Rates are real: they come straight from the recorded spikes of that fly.
 */
export class CircuitMeter {
  private groupOf: Uint8Array;
  private sizes: number[];
  private rate: Float64Array; // smoothed Hz per group
  private ever: boolean[];
  private names: string[];
  private colors: string[];

  constructor(data: BrainData, private tauMs = 250, private refHz = 15) {
    const meta = data as unknown as { group_names: string[]; group_colors: string[] };
    this.names = meta.group_names;
    this.colors = meta.group_colors;
    this.groupOf = data.groupIds;
    this.sizes = this.names.map(() => 0);
    for (let i = 0; i < this.groupOf.length; i++) this.sizes[this.groupOf[i]]++;
    this.rate = new Float64Array(this.names.length);
    this.ever = this.names.map(() => false);
  }

  /** Feed one tick of spikes (tickMs of brain time). */
  addTick(spikes: Uint32Array, tickMs: number) {
    const counts = new Float64Array(this.names.length);
    for (let i = 0; i < spikes.length; i++) {
      const g = this.groupOf[spikes[i]];
      if (g !== undefined) counts[g]++;
    }
    const a = 1 - Math.exp(-tickMs / this.tauMs);
    for (let g = 0; g < this.names.length; g++) {
      const hz = this.sizes[g] ? (counts[g] / this.sizes[g]) * (1000 / tickMs) : 0;
      this.rate[g] += a * (hz - this.rate[g]);
      if (counts[g] > 0) this.ever[g] = true;
    }
  }

  reset() {
    this.rate.fill(0);
    this.ever = this.names.map(() => false);
  }

  levels(): CircuitLevel[] {
    return CIRCUIT_ORDER.filter((k) => this.names.includes(k)).map((k) => {
      const g = this.names.indexOf(k);
      const hz = this.rate[g];
      return { key: k, label: CIRCUIT_LABELS[k] ?? k, color: this.colors[g], level: 1 - Math.exp(-hz / this.refHz),
               rateHz: hz, everActive: this.ever[g] };
    });
  }
}
