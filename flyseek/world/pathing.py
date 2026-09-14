"""
Grid pathfinding for scripted agents: geodesic (walkable) distance fields computed
with Dijkstra on the occupancy grid (8-connected, C speed via scipy.sparse.csgraph).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from flyseek.world.grid import OccupancyGrid

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class GridPaths:
    def __init__(self, grid: OccupancyGrid, clearance: float = 0.2):
        self.grid = grid
        free = grid.walkable & (grid.dist >= clearance)
        self.free = free
        h, w = free.shape
        node_of = -np.ones((h, w), dtype=np.int64)
        cells = np.argwhere(free)
        node_of[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
        self.node_of, self.cells = node_of, cells

        rows, cols, wts = [], [], []
        for dy, dx in _NEIGH:
            r2, c2 = cells[:, 0] + dy, cells[:, 1] + dx
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            ok[ok] = free[r2[ok], c2[ok]]
            rows.append(np.arange(len(cells))[ok])
            cols.append(node_of[r2[ok], c2[ok]])
            wts.append(np.full(int(ok.sum()), np.hypot(dx, dy) * grid.res))
        n = len(cells)
        self.graph = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    def node_at(self, x: float, y: float) -> int:
        cy, cx = self.grid._cell(x, y)
        node = self.node_of[cy, cx]
        if node >= 0:
            return int(node)
        # snap to the nearest free cell
        d = (self.cells[:, 0] - cy) ** 2 + (self.cells[:, 1] - cx) ** 2
        return int(np.argmin(d))

    def distance_field(self, sources_xy: list[tuple[float, float]]) -> np.ndarray:
        """Geodesic distance from the nearest source to every free cell, as an [H, W] array (inf = unreachable)."""
        src = sorted({self.node_at(x, y) for x, y in sources_xy})
        d = dijkstra(self.graph, directed=False, indices=src, min_only=True)
        field = np.full(self.free.shape, np.inf)
        field[self.cells[:, 0], self.cells[:, 1]] = d
        return field

    def field_at(self, field: np.ndarray, x, y) -> np.ndarray:
        cy, cx = self.grid._cell(x, y)
        return field[cy, cx]

    def descend(self, field: np.ndarray, x: float, y: float, ascend: bool = False) -> float | None:
        """Heading (radians) toward the neighboring cell with the lowest (or highest) field value."""
        cy, cx = self.grid._cell(x, y)
        best, best_dir = field[cy, cx], None
        for dy, dx in _NEIGH:
            r, c = cy + dy, cx + dx
            if 0 <= r < field.shape[0] and 0 <= c < field.shape[1] and self.free[r, c]:
                v = field[r, c]
                if np.isfinite(v) and ((v > best) if ascend else (v < best)):
                    best, best_dir = v, (dx, dy)
        return None if best_dir is None else float(np.arctan2(best_dir[1], best_dir[0]))
