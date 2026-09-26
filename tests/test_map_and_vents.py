"""The Skeld floor and vent placement must stay usable by a fly that is 0.25 units wide."""
import numpy as np
import pytest

from amongusfly.world.grid import OccupancyGrid
from amongusfly.world.match import room_grid
from amongusfly.world.pathing import BODY_CLEARANCE, GridPaths
from amongusfly.world.rules import load_extras, load_game_config

BODY_RADIUS = 0.25
EXPECTED_ROOM = {
    "AdminVent": ("Admin",), "BigYVent": ("O2", "Hallway", "Cafeteria"), "CafeVent": ("Cafeteria",),
    "ElecVent": ("Electrical",), "LEngineVent": ("Upper Engine",), "SecurityVent": ("Security",),
    "MedVent": ("MedBay",), "WeaponsVent": ("Weapons",), "ReactorVent": ("Reactor",),
    "REngineVent": ("Lower Engine",), "ShieldsVent": ("Shields",), "UpperReactorVent": ("Reactor",),
    "NavVentNorth": ("Navigation",), "NavVentSouth": ("Navigation",),
}


@pytest.fixture(scope="module")
def world():
    g = OccupancyGrid.skeld()
    return g, room_grid(g), load_extras(), GridPaths(g)


def test_body_radius_matches_the_planner(world):
    assert load_game_config()["body_radius_units"] == BODY_RADIUS <= BODY_CLEARANCE


def test_all_fourteen_vents_are_on_their_real_coordinates(world):
    g, rooms, ex, _ = world
    assert len(ex["vents"]) == 14 and len({v["group"] for v in ex["vents"]}) == 6
    for v in ex["vents"]:
        assert v["snap_distance"] <= g.res, f"{v['game_name']} moved {v['snap_distance']} from its real position"
        assert v["room"] in EXPECTED_ROOM[v["game_name"]], f"{v['game_name']} landed in {v['room']}"


def test_vents_are_standable_and_routable(world):
    g, _, ex, paths = world
    for v in ex["vents"]:
        assert g.dist_at(v["x"], v["y"]) >= BODY_RADIUS, f"a fly cannot stand on {v['game_name']}"
        assert paths.free[g._cell(v["x"], v["y"])], f"the planner cannot route to {v['game_name']}"


def test_vent_links_are_symmetric(world):
    _, _, ex, _ = world
    by_name = {v["game_name"]: v for v in ex["vents"]}
    for v in ex["vents"]:
        for other in v["links_to"]:
            assert v["game_name"] in by_name[other]["links_to"]
            assert by_name[other]["group"] == v["group"]


def test_every_room_is_reachable_by_a_full_size_body(world):
    """Corridors must be wide enough for the body, or flies bounce at doorways."""
    g, rooms, ex, paths = world
    field = paths.distance_field([tuple(ex["spawn"]["xy"])])
    reachable = np.isfinite(field) & paths.free
    assert reachable.sum() == paths.free.sum(), "some walkable cells are unreachable at body clearance"
    for room in ("Weapons", "Navigation", "Reactor", "Electrical", "Admin", "Shields", "O2", "Comms"):
        assert reachable[rooms == room].any(), f"{room} is unreachable"


def test_cafeteria_tables_still_exist(world):
    """The floor reconstruction must not swallow the five Cafeteria tables."""
    from scipy import ndimage
    g, rooms, _, _ = world
    ys, xs = np.nonzero(rooms == "Cafeteria")
    sub = g.walkable[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    holes = ndimage.binary_fill_holes(sub) & ~sub
    _, n = ndimage.label(holes)
    assert n == 5, f"expected 5 tables in Cafeteria, found {n}"
