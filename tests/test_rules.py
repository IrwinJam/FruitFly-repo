"""CPU tests for the Hide n Seek rules engine."""
import numpy as np

from flyseek.world.rules import HideNSeekRules, load_game_config

EXTRAS = {"vents": [{"group": 0, "room": "A", "x": 0.0, "y": 0.0}, {"group": 0, "room": "B", "x": 20.0, "y": 0.0}],
          "spawn": {"xy": [0, 0]}}


def cfg(**timers):
    c = load_game_config()
    c["timers"].update({"hide_phase_s": 1.0, "round_length_s": 10.0, "final_hide_s": 4.0, "ping_interval_s": 1.0, **timers})
    return c


def run(rules, x, y, seconds):
    events = []
    out = None
    for _ in range(int(round(seconds / rules.tick_s))):
        out = rules.step(x, y)
        events += out.events
        for a, (tx, ty) in out.teleports.items():
            x[a], y[a] = tx, ty
    return out, events


def test_seeker_frozen_in_hide_phase_and_boosted_in_final_hide():
    r = HideNSeekRules(["seeker", "hider"], 0.1, cfg(), EXTRAS)
    x, y = np.array([0.0, 50.0]), np.array([0.0, 50.0])
    out, _ = run(r, x, y, 0.5)
    assert out.speed_mult[0] == 0 and out.speed_mult[1] == 1
    out, ev = run(r, x, y, 6.0)  # t = 6.5: final hide starts at 6
    assert r.phase == "final_hide"
    assert out.speed_mult[0] == cfg()["timers"]["final_hide_speed_multiplier"]
    assert [e["phase"] for e in ev if e["kind"] == "phase"] == ["seek", "final_hide"]


def test_no_kill_during_hide_phase_then_kill_and_seeker_wins():
    r = HideNSeekRules(["seeker", "hider"], 0.1, cfg(), EXTRAS)
    x, y = np.array([5.0, 5.2]), np.array([5.0, 5.0])
    run(r, x, y, 0.5)
    assert r.alive[1]
    _, ev = run(r, x, y, 1.0)
    assert not r.alive[1]
    assert r.over and r.winner == "seeker"
    assert any(e["kind"] == "kill" for e in ev)
    r.step(x, y)
    assert r.phase == "over"  # stays over after the seeker wins


def test_hiders_win_on_time_and_pings_fire_in_final_hide():
    r = HideNSeekRules(["seeker", "hider"], 0.1, cfg(), EXTRAS)
    x, y = np.array([0.0, 50.0]), np.array([0.0, 50.0])
    _, ev = run(r, x, y, 11.0)
    assert r.winner == "hiders"
    pings = [e for e in ev if e["kind"] == "ping"]
    assert 3 <= len(pings) <= 5  # final hide 6..10 s, 1 s interval
    assert pings[0]["positions"] == [(50.0, 50.0)]


def test_auto_vent_hides_and_exits_at_linked_vent():
    c = cfg()
    c["vents"].update({"max_time_inside_s": 1.0, "use_radius_units": 0.6, "auto_vent_danger_range_units": 4.0, "max_uses_per_hider": 1})
    r = HideNSeekRules(["seeker", "hider"], 0.1, c, EXTRAS)
    x, y = np.array([3.0, 0.1]), np.array([0.0, 0.0])
    run(r, x, y, 1.05)  # into seek phase
    out, ev = run(r, x, y, 0.2)
    assert any(e["kind"] == "vent_enter" for e in ev)
    assert not out.visible[1] and out.speed_mult[1] == 0
    x[0] = 100.0  # seeker leaves so the hider isn't re-killed on exit
    out, ev = run(r, x, y, 1.2)
    assert any(e["kind"] == "vent_exit" for e in ev)
    assert (x[1], y[1]) == (20.0, 0.0)
    assert r.alive[1]
