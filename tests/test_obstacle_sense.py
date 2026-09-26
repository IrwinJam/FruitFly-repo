"""
Obstacle sense: off for adapters without it, and a wall makes the brain steer away.
"""
import json

import numpy as np
import pytest

from amongusfly.paths import RESULTS_DIR
from amongusfly.train.adapter import PARAM_SETS


def test_adapters_without_the_sense_keep_it_off():
    from amongusfly.train.eval_role import load_values
    run = RESULTS_DIR / "train" / "explore_v5"
    if not (run / "best.json").exists():
        pytest.skip("explore_v5 not trained on this machine")
    assert "gain:obstacle" not in json.loads((run / "best.json").read_text())["values"]
    assert "gain:obstacle" not in load_values("init", "explore_v5", PARAM_SETS["seeker"])


torch = pytest.importorskip("torch")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU brain")


@needs_gpu
def test_wall_on_the_left_steers_the_fly_right():
    from amongusfly.agents.fly_agent import FlyPopulation
    from amongusfly.world.grid import OccupancyGrid
    grid = OccupancyGrid.arena(20, 20)
    n = 8
    # facing east (heading 0) with the north wall 0.45 units away on the left; half the flies without the sense
    y_top = grid.y0 + (grid.h - 1) * grid.res
    ys = np.full(n, y_top - 0.45 - grid.res)
    gain = np.array([1.0] * 4 + [0.0] * 4)
    pop = FlyPopulation("navcore", np.zeros(n), ys, np.zeros(n), grid, seed=5,  # arena is centred on 0
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": 1.0, "goal": 1.0, "obstacle": gain})
    pop.brain.keep_mask = None
    rates = pop._obstacle_rates()
    assert np.all(rates["R"][:4] > 10.0) and np.all(rates["L"][:4] < 1.0), rates  # contralateral delivery
    lr, cnt = np.zeros(n), 0
    for t in range(60):
        res, _ = pop.tick([], goal_angle=np.zeros(n), movable=np.zeros(n, bool))
        if t >= 20:
            hz = res.dn_hz
            lr += (hz["DNa02"]["L"] - hz["DNa02"]["R"]) + (hz["DNa03"]["L"] - hz["DNa03"]["R"])
            cnt += 1
    lr /= cnt
    assert lr[:4].mean() < -5.0, lr  # negative = right turn, away from the wall
    assert abs(lr[4:].mean()) < 3.0, lr  # without the sense the wall has no effect
