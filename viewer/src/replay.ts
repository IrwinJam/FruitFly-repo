export interface ReplayMeta {
  name: string;
  roles: string[];
  brain_agents: number[];
  graph: string;
  neurons_simulated: number;
  tick_ms: number;
  n_ticks: number;
  n_agents: number;
  state_fields: string[];
  map: { x0: number; y0: number; res: number; height: number; width: number };
  vents: { x: number; y: number; group: number; room: string }[];
  events: ReplayEvent[];
  timers: { hide_phase_s: number; round_length_s: number; final_hide_s: number; ping_interval_s: number };
  vision_range: { hider_range_units: number; seeker_range_units: number };
  danger_range: number;
  engineered: string[];
  odor_channels_enabled: boolean;
  disclaimer: string;
  seed: number;
  preset: string;
}

export interface ReplayEvent {
  tick: number;
  kind: "phase" | "kill" | "vent_enter" | "vent_exit" | "ping" | "over";
  [k: string]: any;
}

export interface ReplayIndexEntry {
  name: string;
  brain_agents: number[];
  graph: string;
  seconds: number;
  winner: string | null;
}

export class Replay {
  constructor(
    public meta: ReplayMeta,
    private states: Float32Array,
    private spikeIdx: Uint32Array,
    private spikeOff: Uint32Array,
    public walkable: Uint8Array,
  ) {}

  static async load(name: string, base = "/replays"): Promise<Replay> {
    const dir = `${base}/${name}`;
    const get = (f: string) => fetch(`${dir}/${f}`).then((r) => {
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    const [meta, states, idx, off, walk] = await Promise.all([
      fetch(`${dir}/meta.json`).then((r) => r.json()),
      get("states.bin"),
      get("spike_idx.bin"),
      get("spike_off.bin"),
      get("walkable.bin"),
    ]);
    return new Replay(meta, new Float32Array(states), new Uint32Array(idx), new Uint32Array(off), new Uint8Array(walk));
  }

  get nFields() {
    return this.meta.state_fields.length;
  }

  field(tick: number, agent: number, name: string): number {
    const f = this.meta.state_fields.indexOf(name);
    return this.states[(tick * this.meta.n_agents + agent) * this.nFields + f];
  }

  spikes(tick: number, agent: number): Uint32Array {
    const k = tick * this.meta.n_agents + agent;
    return this.spikeIdx.subarray(this.spikeOff[k], this.spikeOff[k + 1]);
  }

  phaseAt(tick: number): string {
    let phase = "hide";
    for (const e of this.meta.events) {
      if (e.tick > tick) break;
      if (e.kind === "phase") phase = e.phase;
      if (e.kind === "over") phase = "over";
    }
    return phase;
  }

  inVent(tick: number, agent: number): boolean {
    let inside = false;
    for (const e of this.meta.events) {
      if (e.tick > tick) break;
      if (e.agent !== agent) continue;
      if (e.kind === "vent_enter") inside = true;
      if (e.kind === "vent_exit") inside = false;
    }
    return inside;
  }
}
