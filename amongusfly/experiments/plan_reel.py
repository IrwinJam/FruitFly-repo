"""
Plan the showcase video from recorded matches: one clip per story beat (hide phase, a chase
and catch, a vent escape, Final Hide pings, a hider win, a seeker win), chosen from each match's
own events and positions. The viewer plays the reel back to back and records it as one WebM.
Every clip is real, unedited game time.

    python -m amongusfly.experiments.plan_reel --prefix showcase_v4
    open http://localhost:5173/?reel=/replays/reel_showcase_v4.json&rec=1&save=server
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPLAYS = Path("viewer/public/replays")


def load(name: str):
    d = REPLAYS / name
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    nf = len(m["state_fields"])
    st = np.fromfile(d / "states.bin", dtype=np.float32).reshape(m["n_ticks"], m["n_agents"], nf)
    g = m["map"]
    walk = np.fromfile(d / "walkable.bin", dtype=np.uint8).reshape(g["height"], g["width"])
    return m, st, walk


def clear_line(walk, g, a, b) -> bool:
    """True if the straight segment a→b stays on walkable cells (row 0 is lowest y)."""
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / (g["res"] * 0.5)))
    for t in np.linspace(0.0, 1.0, n):
        x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        c, r = int(round((x - g["x0"]) / g["res"])), int(round((y - g["y0"]) / g["res"]))
        if not (0 <= r < g["height"] and 0 <= c < g["width"]) or not walk[r, c]:
            return False
    return True


def sightings(m, st, walk):
    """Ticks (every 100 ms) at which the seeker first gets a hider in view, per hider."""
    f = {k: i for i, k in enumerate(m["state_fields"])}
    seek = next((e["tick"] for e in m["events"] if e["kind"] == "phase" and e["phase"] == "seek"), 0)
    rng = m["vision_range"]["seeker_range_units"]
    seeker = m["roles"].index("seeker")
    first = {}
    for t in range(seek, m["n_ticks"], 5):
        s = st[t, seeker]
        for h, role in enumerate(m["roles"]):
            if role != "hider" or h in first or st[t, h, f["alive"]] < 0.5:
                continue
            a, b = (s[f["x"]], s[f["y"]]), (st[t, h, f["x"]], st[t, h, f["y"]])
            if np.hypot(b[0] - a[0], b[1] - a[1]) <= rng and clear_line(walk, m["map"], a, b):
                first[h] = t
    return first


def plan(prefix: str) -> dict:
    names = sorted(p.name for p in REPLAYS.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not names:
        raise SystemExit(f"no replays named {prefix}*")
    games = {n: load(n) for n in names}
    s = lambda m, tick: round(tick * m["tick_ms"] / 1000, 2)
    over = {n: next(e for e in g[0]["events"] if e["kind"] == "over") for n, g in games.items()}
    seeker_wins = [n for n in names if over[n].get("winner") == "seeker"]
    hider_wins = [n for n in names if over[n].get("winner") != "seeker"]
    segs = []

    # 1. the hide phase of a 3-hider seeker win
    first = next((n for n in seeker_wins if "_3h_" in n), names[0])
    m = games[first][0]
    segs.append({"replay": first, "start": 0, "end": m["timers"]["hide_phase_s"] + 4, "speed": 2,
                 "caption": ["Hide phase: the seeker is frozen while the hiders scatter",
                             "Each panel is one fly's simulated brain; glowing points just spiked"]})

    # 2. first sighting → catch: the tightest sighting-to-catch pair across all replays
    best = None
    for n, (m, st, walk) in games.items():
        seen = sightings(m, st, walk)
        for e in m["events"]:
            if e["kind"] == "kill" and e["victim"] in seen:
                lag = e["tick"] - seen[e["victim"]]
                if 40 <= lag <= 500 and (best is None or lag < best[0]):
                    best = (lag, n, seen[e["victim"]], e["tick"])
    if best:
        _, n, t0, t1 = best
        m = games[n][0]
        segs.append({"replay": n, "start": max(0, s(m, t0) - 3), "end": s(m, t1) + 1.5, "speed": 1,
                     "caption": ["The seeker spots a hider and gives chase",
                                 "Watch the pursuit (LC10a) and steering (DNa02/03) bars in its panel"]})

    # 3. a hider escaping through a vent
    for n, (m, _, _) in games.items():
        v = next((e for e in m["events"] if e["kind"] == "vent_enter"), None)
        if v:
            segs.append({"replay": n, "start": max(0, s(m, v["tick"]) - 3), "end": s(m, v["tick"]) + 4, "speed": 1,
                         "caption": ["A hider escapes through a vent",
                                     "Vents sit at their real in-game positions and link as in the game"]})
            break

    # 4. Final Hide pings, in a match that reaches it
    for n in hider_wins + seeker_wins:
        m = games[n][0]
        fh = next((e for e in m["events"] if e["kind"] == "phase" and e["phase"] == "final_hide"), None)
        if fh:
            segs.append({"replay": n, "start": s(m, fh["tick"]) - 2, "end": s(m, fh["tick"]) + 14, "speed": 2,
                         "caption": ["Final Hide: every hider's position is pinged",
                                     f"Every {m['timers']['ping_interval_s']} s the hiders are revealed and the seeker heads for them"]})
            break

    # 5. a hider win, to its end card
    if hider_wins:
        n = hider_wins[0]
        m = games[n][0]
        end = s(m, over[n]["tick"])
        segs.append({"replay": n, "start": end - 8, "speed": 1, "caption": ["Time runs out"]})

    # 6. a seeker win, from the last catch to its end card
    n = seeker_wins[-1] if seeker_wins else names[-1]
    m = games[n][0]
    last = max(e["tick"] for e in m["events"] if e["kind"] == "kill")
    segs.append({"replay": n, "start": max(0, s(m, last) - 6), "speed": 1, "caption": ["The last hider is caught"]})

    return {"name": f"reel_{prefix}", "segments": segs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="showcase_v4")
    a = ap.parse_args()
    reel = plan(a.prefix)
    out = REPLAYS / f"{reel['name']}.json"
    out.write_text(json.dumps(reel, indent=2), encoding="utf-8")
    total = 0.0
    for g in reel["segments"]:
        m = json.loads((REPLAYS / g["replay"] / "meta.json").read_text(encoding="utf-8"))
        end = g.get("end", m["n_ticks"] * m["tick_ms"] / 1000)
        dur = (end - g["start"]) / g["speed"] + (1.5 if "end" not in g else 0.8)
        total += dur
        print(f"  {g['replay']:24s} {g['start']:7.1f}-{end:7.1f}s x{g['speed']}  ~{dur:4.1f}s  {g['caption'][0]}")
    print(f"wrote {out}  (~{total:.0f} s of video)")


if __name__ == "__main__":
    main()
