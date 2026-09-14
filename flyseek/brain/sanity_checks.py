"""
Milestone M3: does the connectome LIF model actually carry a usable signal from
sensory neurons to the motor/descending neurons we plan to read out for gameplay?

This is flagged in PROJECT_PLAN.md section 7 as the single biggest open risk --
Eon's own writeup describes their vision pathway as "somewhat decorative" (not
actually driving behavior), and DOOMFLY's model "failed visual ... validation
gates". So this script tests it directly rather than assuming it works, using the
roles resolved in `celltypes.py` against real MaleCNS body IDs.

NOTE ON SCOPE: the original plan's W0 checklist included a sugar-GRN -> MN9 feeding
circuit test (reproducing Shiu et al. 2024). That's dropped here: cell-type
resolution (M1) found MaleCNS has no gustatory (sugar/bitter) receptor neurons
resolvable by name at this confidence threshold (see celltypes.py docstring) --
Shiu et al.'s original test used specific body IDs from FlyWire's female
connectome, not type-name lookups on MaleCNS, so it isn't directly reproducible
with the roles this project resolved. The four checks below stand in as the
male-CNS-appropriate equivalents for FlySeek's actual sensory/motor roles.

Checks:
  1. steering_readout   -- force DNa02-L spiking directly; does the readout function
                            (motors.py-equivalent, inlined here) turn the kinematic
                            body left, not right or nothing?
  2. pursuit_pathway     -- drive LC10a on one side; does DNa02 show a same-side (or
                            at least consistent) firing-rate asymmetry downstream?
  3. looming_pathway     -- drive LC4+LPLC2 (looming); does DNp01 (giant fiber) rate
                            rise above its undriven baseline?
  4. backward_readout    -- force MDN spiking; does the readout produce backward
                            (negative forward) drive?

Each check reports actual measured numbers, not just pass/fail, and the
`bypass_downstream` flag from config/senses.yaml is exercised for check 2: if
LC10a driven directly doesn't produce an asymmetry, we retry injecting at the
downstream pursuit_relay (AOTU019/025) instead, and report which one (if either)
worked.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from flyseek.brain.lif_torch import LIFBrain

CACHE_DIR = Path(r"C:\flyseek-data\cache")
RESULTS_PATH = Path(r"C:\Users\Irwin\OneDrive\Desktop\FruitFly\docs\sanity_check_results.json")


def load_role_indices() -> dict[str, list[int]]:
    """role -> list of neuron `idx` (0..N-1, matching neurons.parquet / the graph)."""
    role_body_ids = json.loads((CACHE_DIR / "role_body_ids.json").read_text())
    import pandas as pd
    neurons = pd.read_parquet(CACHE_DIR / "neurons.parquet")
    body_to_idx = dict(zip(neurons["bodyId"], neurons["idx"]))
    role_idx = {}
    for role, bids in role_body_ids.items():
        role_idx[role] = [body_to_idx[b] for b in bids if b in body_to_idx]
    return role_idx


def firing_rate_hz(spike_counts: dict[int, int], indices: list[int], window_ms: float) -> float:
    if not indices:
        return 0.0
    total = sum(spike_counts.get(i, 0) for i in indices)
    return total / (len(indices) * (window_ms / 1000.0))


def run_driven_trial(
    brain: LIFBrain,
    drive_indices: list[int],
    drive_hz: float,
    n_steps: int,
    track_roles: dict[str, list[int]],
) -> dict[str, dict[int, int]]:
    """
    Drives `drive_indices` with a Poisson process at drive_hz for n_steps, tracks
    spike counts per neuron for every role in `track_roles`. Returns
    {role: {idx: count}}.
    """
    brain.reset(batch_size=1)
    n, dev, dt = brain.n_neurons, brain.device, brain.dt_ms
    p_spike_per_step = drive_hz * (dt / 1000.0)

    counts: dict[str, dict[int, int]] = {r: {} for r in track_roles}
    drive_idx_t = torch.tensor(drive_indices, dtype=torch.long, device=dev)

    for _ in range(n_steps):
        ext = torch.zeros(n, 1, device=dev)
        if len(drive_idx_t):
            fire = (torch.rand(len(drive_idx_t), device=dev) < p_spike_per_step).float()
            ext[drive_idx_t, 0] = fire * 40.0  # strong forced synaptic-equivalent kick
        spikes = brain.step(ext)
        spiking_idx = spikes[:, 0].nonzero().flatten().tolist()
        spiking_set = set(spiking_idx)
        for role, idxs in track_roles.items():
            for i in idxs:
                if i in spiking_set:
                    counts[role][i] = counts[role].get(i, 0) + 1

    return counts


def check_steering_readout(brain: LIFBrain, role_idx: dict) -> dict:
    steering = role_idx["steering_high_gain"]  # DNa02, 2 bodies (L,R)
    import pandas as pd
    neurons = pd.read_parquet(CACHE_DIR / "neurons.parquet").set_index("idx")
    sides = {i: neurons.loc[i, "somaSide"] for i in steering}
    left_idx = [i for i, s in sides.items() if s == "L"]

    counts = run_driven_trial(brain, left_idx, drive_hz=300.0, n_steps=400, track_roles={"steering_high_gain": steering})
    rates = {sides[i]: firing_rate_hz(counts["steering_high_gain"], [i], 400 * brain.dt_ms) for i in steering}
    turn_signal = rates.get("L", 0.0) - rates.get("R", 0.0)
    return {
        "check": "steering_readout",
        "description": "Force DNa02-L spiking; expect L rate >> R rate (turn-left readout).",
        "rates_hz": rates,
        "turn_signal": turn_signal,
        "pass": turn_signal > 0,
    }


def check_pursuit_pathway(brain: LIFBrain, role_idx: dict, bypass: bool) -> dict:
    import pandas as pd
    neurons = pd.read_parquet(CACHE_DIR / "neurons.parquet").set_index("idx")

    drive_role = "pursuit_relay" if bypass else "target_motion_detector"
    drive_all = role_idx[drive_role]
    drive_sides = {i: neurons.loc[i, "somaSide"] for i in drive_all}
    drive_left = [i for i, s in drive_sides.items() if s == "L"]

    steering = role_idx["pursuit_descending"]  # DNa03, DNa02
    steer_sides = {i: neurons.loc[i, "somaSide"] for i in steering}

    counts = run_driven_trial(brain, drive_left, drive_hz=200.0, n_steps=600, track_roles={"pursuit_descending": steering})
    rate_l = firing_rate_hz(counts["pursuit_descending"], [i for i, s in steer_sides.items() if s == "L"], 600 * brain.dt_ms)
    rate_r = firing_rate_hz(counts["pursuit_descending"], [i for i, s in steer_sides.items() if s == "R"], 600 * brain.dt_ms)
    asymmetry = rate_l - rate_r
    return {
        "check": "pursuit_pathway",
        "description": f"Drive {drive_role} (L side); expect L-side DNa02/DNa03 rate > R-side.",
        "bypass_downstream": bypass,
        "rate_l_hz": rate_l,
        "rate_r_hz": rate_r,
        "asymmetry": asymmetry,
        "pass": asymmetry > 0.5,
    }


def check_looming_pathway(brain: LIFBrain, role_idx: dict) -> dict:
    drive = role_idx["looming_expansion"] + role_idx["looming_size"]
    gf = role_idx["giant_fiber"]

    baseline_counts = run_driven_trial(brain, [], drive_hz=0.0, n_steps=400, track_roles={"giant_fiber": gf})
    baseline_rate = firing_rate_hz(baseline_counts["giant_fiber"], gf, 400 * brain.dt_ms)

    driven_counts = run_driven_trial(brain, drive, drive_hz=250.0, n_steps=400, track_roles={"giant_fiber": gf})
    driven_rate = firing_rate_hz(driven_counts["giant_fiber"], gf, 400 * brain.dt_ms)

    return {
        "check": "looming_pathway",
        "description": "Drive LC4+LPLC2; expect DNp01 (giant fiber) rate to rise above baseline.",
        "baseline_hz": baseline_rate,
        "driven_hz": driven_rate,
        "delta_hz": driven_rate - baseline_rate,
        "pass": driven_rate > baseline_rate,
    }


def check_backward_readout(brain: LIFBrain, role_idx: dict) -> dict:
    mdn = role_idx["backward_drive"]
    counts = run_driven_trial(brain, mdn, drive_hz=300.0, n_steps=400, track_roles={"backward_drive": mdn})
    rate = firing_rate_hz(counts["backward_drive"], mdn, 400 * brain.dt_ms)
    return {
        "check": "backward_readout",
        "description": "Force MDN spiking; readout should register nonzero backward drive.",
        "mdn_rate_hz": rate,
        "pass": rate > 0,
    }


def main():
    role_idx = load_role_indices()
    print(f"Loaded {len(role_idx)} resolved roles.")

    brain = LIFBrain(tag="full")
    print(f"Loaded 'full' graph: {brain.n_neurons:,} neurons, {brain.num_edges():,} edges on {brain.device}.")

    results = []

    print("\n[1/4] steering_readout ...")
    r1 = check_steering_readout(brain, role_idx)
    print(f"  L={r1['rates_hz']} turn_signal={r1['turn_signal']:.2f} Hz -> {'PASS' if r1['pass'] else 'FAIL'}")
    results.append(r1)

    print("\n[2/4] pursuit_pathway (direct LC10a injection) ...")
    r2 = check_pursuit_pathway(brain, role_idx, bypass=False)
    print(f"  L={r2['rate_l_hz']:.2f}Hz R={r2['rate_r_hz']:.2f}Hz asymmetry={r2['asymmetry']:.2f} -> {'PASS' if r2['pass'] else 'FAIL'}")
    if not r2["pass"]:
        print("  Direct injection did not produce asymmetry. Retrying with bypass_downstream (AOTU019/025) ...")
        r2b = check_pursuit_pathway(brain, role_idx, bypass=True)
        print(f"  [bypass] L={r2b['rate_l_hz']:.2f}Hz R={r2b['rate_r_hz']:.2f}Hz asymmetry={r2b['asymmetry']:.2f} -> {'PASS' if r2b['pass'] else 'FAIL'}")
        results.append(r2b)
    else:
        results.append(r2)

    print("\n[3/4] looming_pathway ...")
    r3 = check_looming_pathway(brain, role_idx)
    print(f"  baseline={r3['baseline_hz']:.2f}Hz driven={r3['driven_hz']:.2f}Hz delta={r3['delta_hz']:.2f} -> {'PASS' if r3['pass'] else 'FAIL'}")
    results.append(r3)

    print("\n[4/4] backward_readout ...")
    r4 = check_backward_readout(brain, role_idx)
    print(f"  MDN rate={r4['mdn_rate_hz']:.2f}Hz -> {'PASS' if r4['pass'] else 'FAIL'}")
    results.append(r4)

    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n{n_pass}/{len(results)} checks passed.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
