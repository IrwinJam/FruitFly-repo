"""CPU tests for the adapter encoding, per-fly decoder parameters and the goal policy."""
import copy

import numpy as np
import pytest

from amongusfly.agents.goal_policy import ExplorationGoalPolicy
from amongusfly.motors.decoders import MotorDecoder, load_motor_config
from amongusfly.train.adapter import EXPLORE_PARAMS, apply_decoder, decode, policy_values, to_unit
from amongusfly.world.grid import OccupancyGrid


def test_adapter_roundtrip_and_bounds():
    u = to_unit(EXPLORE_PARAMS)
    vals = decode(EXPLORE_PARAMS, u)
    for p in EXPLORE_PARAMS:
        assert vals[p.name] == pytest.approx(p.init, rel=1e-6, abs=1e-9)
    extreme = decode(EXPLORE_PARAMS, np.array([[0.0] * len(EXPLORE_PARAMS), [1.0] * len(EXPLORE_PARAMS)]))
    for p in EXPLORE_PARAMS:
        assert extreme[p.name][0] == pytest.approx(p.lo) and extreme[p.name][1] == pytest.approx(p.hi)


def test_decoder_accepts_per_fly_parameters():
    cfg = copy.deepcopy(load_motor_config())
    apply_decoder(cfg, {"decoder:turn.scale_hz": np.array([10.0, 100.0]), "decoder:ema_tau_ms": np.array([20.0, 200.0])})
    d = MotorDecoder(2, cfg)
    hz = {t: {"L": np.zeros(2), "R": np.zeros(2)} for t in d.rates}
    hz["DNa02"]["L"] = np.array([30.0, 30.0])
    d.update_rates(hz, 20.0)
    cmd = d.decode()
    assert cmd.omega[0] > cmd.omega[1] > 0  # small scale + short tau turns harder


def test_goal_policy_prefers_open_unvisited_direction_and_never_picks_a_wall():
    g = OccupancyGrid.arena(10, 10, res=0.05)
    params = {"w_free": 1.0, "w_novel": 2.0, "w_align": 0.0, "w_keep": 0.0, "decide_s": 1.0}
    pol = ExplorationGoalPolicy(1, g, params)
    # fly near the west wall facing west: the chosen goal must point away from the wall (east-ish)
    goal = pol.step(np.array([-4.6]), np.array([0.0]), np.array([np.pi]), 0.02)
    assert np.cos(goal[0]) > 0.3
    # visit a strip east of the fly heavily; novelty should now favor north/south over east
    for x in np.arange(-3.5, 3.5, 0.5):
        pol.visits[0, pol._bins(np.array([x]), np.array([0.0]))[0][0], pol._bins(np.array([x]), np.array([0.0]))[1][0]] += 50
    pol.next_decide[:] = 0
    goal = pol.step(np.array([-4.0]), np.array([0.0]), np.array([0.0]), 0.02)
    assert abs(np.sin(goal[0])) > 0.3
