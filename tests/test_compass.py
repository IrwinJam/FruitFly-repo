"""
The compass pathway must steer toward the goal whichever way the fly faces.

Checks the steering sign at all four cardinal headings, so an error in the left/right bridge
convention (which cancels steering at 90 and 270 degrees) cannot pass unnoticed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs the GPU brain", allow_module_level=True)

from amongusfly.agents.fly_agent import FlyPopulation  # noqa: E402
from amongusfly.world.grid import OccupancyGrid  # noqa: E402


def test_steering_sign_is_correct_at_every_cardinal_heading():
    heads = np.deg2rad(np.repeat([0.0, 90.0, 180.0, 270.0], 2))
    side = np.tile([1.0, -1.0], 4)  # goal 90 deg to the left, then to the right
    goal = heads + side * np.pi / 2
    n = len(heads)
    pop = FlyPopulation("navcore", np.full(n, 10.0), np.full(n, 10.0), heads.copy(), OccupancyGrid.arena(20, 20), seed=3,
                        channel_gain={"target": 0.0, "loom": 0.0, "photo": 1.0, "danger": 0.0, "ping": 0.0,
                                      "compass": 1.0, "goal": 1.0})
    pop.brain.keep_mask = None
    drive, cnt = np.zeros(n), 0
    for t in range(60):  # 1.2 s, bodies held still
        res, _ = pop.tick([], goal_angle=goal, movable=np.zeros(n, bool))
        if t >= 20:
            hz = res.dn_hz
            drive += (hz["DNa02"]["L"] - hz["DNa02"]["R"]) + (hz["DNa03"]["L"] - hz["DNa03"]["R"])
            cnt += 1
    drive /= cnt
    # + = left turn; every heading must turn toward the goal with a clear margin
    assert np.all(side * drive > 5.0), dict(zip([f"{np.rad2deg(h):.0f}{'L' if s > 0 else 'R'}" for h, s in zip(heads, side)],
                                                np.round(drive, 1)))
