"""Smoothing (D6): OneEuroFilter on hand points, for the cursor and landmarks layers.

Gestures are not smoothed here; their stability comes from the stabilize stage.
"""

from OneEuroFilter import OneEuroFilter

from mc_gesture.config import Config

PALM = (0, 5, 9, 13, 17)  # wrist + finger bases: stays put whatever the fingers do


class HandSmoother:
    """One filter pair (x, y) per point per hand; a hand's filters reset when it disappears."""

    def __init__(self, config: Config, freq: float = 30.0):
        self.config, self.freq = config, freq
        self._filters: dict[tuple, list[tuple[OneEuroFilter, OneEuroFilter]]] = {}

    def _new(self, n: int):
        c = self.config
        return [(OneEuroFilter(self.freq, c.min_cutoff, c.beta), OneEuroFilter(self.freq, c.min_cutoff, c.beta))
                for _ in range(n)]

    def smooth(self, key: tuple, points: list[tuple[float, float]], t: float) -> list[tuple[float, float]]:
        filters = self._filters.setdefault(key, self._new(len(points) + 1))
        return [(fx(x, t), fy(y, t)) for (fx, fy), (x, y) in zip(filters, points)]

    def cursor(self, key: tuple, points: list[tuple[float, float]], t: float) -> tuple[float, float]:
        """Smoothed palm center. Uses its own filter pair (the extra last one)."""
        filters = self._filters.setdefault(key, self._new(len(points) + 1))
        fx, fy = filters[-1]
        x = sum(points[i][0] for i in PALM) / len(PALM)
        y = sum(points[i][1] for i in PALM) / len(PALM)
        return fx(x, t), fy(y, t)

    def keep_only(self, keys: set) -> None:
        for key in [k for k in self._filters if k not in keys]:
            del self._filters[key]
