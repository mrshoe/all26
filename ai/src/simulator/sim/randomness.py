"""Domain randomization: `Range` values, drawn from the episode's seed.

A `Range` can stand in for any number, color or vector an object takes, for
example `Light(intensity=Range(500, 2000))`, or one coordinate of a position,
`(Range(-2, 2), 3, 0.5)`. The framework draws every Range once, after the
Scene's and Scenario's `setup`, in the order the objects were added. So a
Scenario that replaces a Scene's Range replaces it before anything is drawn.
"""

from __future__ import annotations

import colorsys
from typing import Any, Sequence

import numpy as np

# sRGB, 0 to 1.
COLORS = {
    "yellow": (0.98, 0.85, 0.05),
    "orange": (0.95, 0.45, 0.05),
    "red": (0.8, 0.08, 0.08),
    "green": (0.1, 0.6, 0.15),
    "blue": (0.08, 0.25, 0.8),
    "white": (0.92, 0.92, 0.9),
    "gray": (0.5, 0.5, 0.5),
    "black": (0.05, 0.05, 0.05),
    "tan": (0.8, 0.7, 0.5),
    "brown": (0.4, 0.25, 0.12),
}


class Range:
    """A value drawn at random for each episode.

        Range(1, 3)                     a number from 1 to 3
        Range.choice(["wood", "tile"])  one of these
        Range.color("yellow")           a yellow, varied a little
        Range.color("white", "tan")     a white or a tan, varied a little
        Range.sphere()                  a direction (unit vector), any way
    """

    def __init__(self, low: float, high: float):
        if high < low:
            raise ValueError(f"Range({low}, {high}): the high end is below the low end")
        self.low, self.high = low, high

    def draw(self, rng: np.random.Generator) -> Any:
        return float(rng.uniform(self.low, self.high))

    def __repr__(self) -> str:
        return f"Range({self.low}, {self.high})"

    @staticmethod
    def choice(options: Sequence[Any]) -> Range:
        return _Choice(list(options))

    @staticmethod
    def color(*names: str, hue: float = 0.02, saturation: float = 0.15, value: float = 0.2) -> Range:
        """One of the named colors (see COLORS), with its hue, saturation
        and brightness each moved by up to the given amount."""
        for name in names:
            if name not in COLORS:
                raise ValueError(f"Unknown color {name!r}. Known colors: {', '.join(COLORS)}")
        return _Color(names or ("gray",), hue, saturation, value)

    @staticmethod
    def sphere() -> Range:
        return _Sphere()


class _Choice(Range):
    def __init__(self, options: list):
        if not options:
            raise ValueError("Range.choice needs at least one option")
        self.options = options

    def draw(self, rng):
        return self.options[int(rng.integers(len(self.options)))]

    def __repr__(self):
        return f"Range.choice({self.options!r})"


class _Color(Range):
    def __init__(self, names, hue, saturation, value):
        self.names, self.hue, self.saturation, self.value = names, hue, saturation, value

    def draw(self, rng):
        name = self.names[int(rng.integers(len(self.names)))]
        h, s, v = colorsys.rgb_to_hsv(*COLORS[name])
        h = (h + rng.uniform(-self.hue, self.hue)) % 1.0
        s = float(np.clip(s + rng.uniform(-self.saturation, self.saturation), 0, 1))
        v = float(np.clip(v + rng.uniform(-self.value, self.value), 0, 1))
        return tuple(round(c, 4) for c in colorsys.hsv_to_rgb(h, s, v))

    def __repr__(self):
        return f"Range.color({', '.join(map(repr, self.names))})"


class _Sphere(Range):
    def __init__(self):
        pass

    def draw(self, rng):
        v = rng.normal(size=3)
        return tuple(float(x) for x in v / np.linalg.norm(v))

    def __repr__(self):
        return "Range.sphere()"


def resolve(value: Any, rng: np.random.Generator) -> Any:
    """`value` with every Range in it drawn, looking inside tuples, lists and
    dicts' values."""
    if isinstance(value, Range):
        return value.draw(rng)
    if isinstance(value, (tuple, list)):
        return type(value)(resolve(v, rng) for v in value)
    if isinstance(value, dict):
        return {k: resolve(v, rng) for k, v in value.items()}
    return value
