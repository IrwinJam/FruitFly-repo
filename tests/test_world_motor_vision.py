"""CPU tests for the grid, kinematic body, motor decoder and vision encoder (no connectome needed)."""
import numpy as np
import pytest

from amongusfly.motors.body_kinematic import KinematicBody
from amongusfly.motors.decoders import MotorDecoder, load_motor_config
from amongusfly.senses.vision import VisionEncoder, VisualObject
from amongusfly.world.grid import OccupancyGrid


def test_arena_raycast_distances():
    g = OccupancyGrid.arena(10, 10, res=0.05)
    x, y = np.array([0.0]), np.array([0.0])
    angles = np.array([[0.0, np.pi / 2, np.pi, -np.pi / 2]])
    d = g.raycast(x, y, angles, max_range=20.0)[0]
    assert np.allclose(d, 5.0, atol=0.1)


def test_body_moves_forward_and_slides_on_wall():
    g = OccupancyGrid.arena(10, 10, res=0.05)
    b = KinematicBody(np.array([0.0, 4.6]), np.array([0.0, 0.0]), np.array([0.0, np.pi / 4]))
    for _ in range(20):
        b.step(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.05, g)
    assert b.x[0] == pytest.approx(1.0, abs=1e-6) and b.y[0] == pytest.approx(0.0, abs=1e-6)
    # second fly hits the +x wall heading diagonally: x blocked near the wall, y keeps sliding
    # (slower than in the open: body_kinematic slows down close to geometry, see CAREFUL_*)
    assert b.x[1] <= 5.0 - b.radius + 0.06
    assert b.y[1] > 0.3
    assert b.wall_contact[1]


def test_careful_approach_slows_near_walls_only():
    g = OccupancyGrid.arena(10, 10, res=0.05)
    b = KinematicBody(np.array([0.0, 4.7]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert g.dist_at(b.x, b.y)[1] == pytest.approx(0.3, abs=0.01)  # one body radius of slack
    b.step(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.05, g)
    assert b.speed[0] == pytest.approx(1.0)  # open floor: full speed
    assert 0.45 <= b.speed[1] < 0.75         # close to the wall: slowed


def test_wedged_body_escapes_instead_of_freezing():
    """A fly teleported closer to a wall than its own radius (a vent exit) must not stay stuck."""
    g = OccupancyGrid.arena(10, 10, res=0.05)
    b = KinematicBody(np.array([4.95]), np.array([0.0]), np.array([0.0]))  # inside the wall margin
    assert g.dist_at(b.x, b.y)[0] < b.radius
    for _ in range(40):
        b.step(np.array([0.0]), np.array([0.0]), 0.05, g)  # not even commanded to move
    assert g.dist_at(b.x, b.y)[0] >= b.radius


def test_left_turn_sign():
    g = OccupancyGrid.arena(10, 10)
    b = KinematicBody(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    b.step(np.array([0.0]), np.array([1.0]), 0.1, g)
    assert b.heading[0] == pytest.approx(0.1)


def test_decoder_left_dn_turns_left_and_giant_fiber_dashes():
    cfg = load_motor_config()
    d = MotorDecoder(2, cfg)
    hz = {t: {"L": np.zeros(2), "R": np.zeros(2)} for t in d.rates}
    hz["DNa02"]["L"] = np.array([40.0, 0.0])
    hz["DNp01"]["L"] = np.array([0.0, 200.0])
    hz["DNp01"]["R"] = np.array([0.0, 200.0])
    for _ in range(50):
        d.update_rates(hz, 20.0)
    cmd = d.decode()
    assert cmd.omega[0] > 0 and cmd.omega[1] == 0
    assert not cmd.dash[0] and cmd.dash[1]
    assert cmd.speed[1] == pytest.approx(cfg["forward"]["base_speed_units_per_s"] * cfg["dash"]["speed_multiplier"])


def test_vision_target_side_and_occlusion():
    g = OccupancyGrid.arena(10, 10)
    enc = VisionEncoder(3)
    x = np.zeros(3); y = np.zeros(3); h = np.zeros(3)
    # fly 0: target to the left; fly 1: to the right; fly 2: target outside the arena behind a wall
    tx = np.array([2.0, 2.0, 8.0]); ty = np.array([2.0, -2.0, 0.0])
    out = enc.encode(x, y, h, [VisualObject(tx, ty, 0.3, "target")], g, 0.02)
    r = out.rates["target"]
    assert r["L"][0] > 0 and r["R"][0] == 0
    assert r["R"][1] > 0 and r["L"][1] == 0
    assert r["L"][2] == 0 and r["R"][2] == 0
    assert r["L"][0] <= enc.cfg.target_max_hz


def test_vision_looming_needs_expansion():
    g = OccupancyGrid.arena(10, 10)
    enc = VisionEncoder(1)
    z = np.zeros(1)
    enc.encode(z, z, z, [VisualObject(np.array([3.0]), np.array([1.0]), 0.3, "threat")], g, 0.02)
    near = enc.encode(z, z, z, [VisualObject(np.array([2.0]), np.array([0.7]), 0.3, "threat")], g, 0.02)
    assert near.rates["loom"]["L"][0] > 0
    same = enc.encode(z, z, z, [VisualObject(np.array([2.0]), np.array([0.7]), 0.3, "threat")], g, 0.02)
    assert same.rates["loom"]["L"][0] == 0
