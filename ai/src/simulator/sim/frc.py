"""FIRST's official FRC field for a season, as a Group: add it to any Scene
or Scenario.

    from sim.frc import FrcField

    self.field = self.add(FrcField(year=2026, rotation=(0, 0, Range(-2, 2))))
    self.field.hub          # [hub_blue, hub_red]: Props
    self.field.fuel         # the game pieces at their starting spots
    self.remove(self.field.outpost[0])

Its models come from FIRST's CAD (sim/assets/frc). Each element is a static
Prop named after what it is (`field.hub.0`), labeled if the season's config
gives it a label. Game pieces are GamePieces, which physics moves, except
those starting on something other than the carpet (2025's ALGAE on the REEF
or on CORAL), which stay put, as squeezed real ones do. The
field's center is the Group's position; its carpet is at z = 0, so a Scene
with a floor keeps its floor under it.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from sim import assets
from sim.objects import GamePiece, Prop
from sim.world import Group


@dataclass(kw_only=True)
class FrcField(Group):
    # The season: there's a catalog for each one imported (sim/assets/frc).
    year: int = 2026
    # Put the game pieces where the CAD has them at the start of a match.
    starting_pieces: bool = True
    # Which elements to include, by name (e.g. ("hub", "carpet")); all if None.
    elements: Any = None

    def build(self, rng):
        library = assets.frc_library(int(self.year))
        catalog = library.catalog
        keep = None if self.elements is None else set(self.elements)
        if keep is not None:
            known = {library.info(e["asset"])["element"] for e in catalog["layout"]}
            if keep - known:
                raise ValueError(f"No {', '.join(sorted(keep - known))} on the {self.year} field. "
                                 f"Its elements: {', '.join(sorted(known))}")
        reserved = {f.name for f in dataclasses.fields(self)} | set(dir(type(self)))
        found: dict[str, list] = defaultdict(list)
        for entry in catalog["layout"]:
            model = library.info(entry["asset"])
            if keep is not None and model["element"] not in keep:
                continue
            found[model["element"]].append(Prop(
                asset=f"frc/{self.year}/{entry['asset']}", label=model.get("label"),
                position=tuple(entry["position"]),
            ))
        if self.starting_pieces:
            for name, piece in catalog["pieces"].items():
                model = library.info(piece["asset"])
                # Pieces held by what they're on (2025's ALGAE on the REEF)
                # stay put: nothing here knocks them loose.
                held = piece.get("held") or [False] * len(piece["start"])
                found[name] += [GamePiece(asset=f"frc/{self.year}/{piece['asset']}", label=model.get("label"),
                                          position=tuple(p), static=h) for p, h in zip(piece["start"], held)]
        for name, items in found.items():
            setattr(self, f"{name}_" if name in reserved else name, self.add(items))

    def to_wpilib(self, position, yaw: float = 0.0) -> tuple[float, float, float, float]:
        """A point in the world (and a heading about z, in degrees) in
        WPILib's field coordinates: x, y and z in meters from the blue
        alliance's corner, and the heading. For comparing with what robot
        code sees, e.g. a logged pose."""
        fit = assets.frc_library(int(self.year)).catalog["field"]["wpilib"]
        turn = Rotation.from_euler("xyz", self.rotation, degrees=True)
        local = turn.inv().apply(np.asarray(position, float) - np.asarray(self.position, float))
        a = math.radians(fit["yaw"])
        x = math.cos(a) * local[0] - math.sin(a) * local[1] + fit["offset"][0]
        y = math.sin(a) * local[0] + math.cos(a) * local[1] + fit["offset"][1]
        heading = yaw - float(turn.as_euler("xyz", degrees=True)[2]) + fit["yaw"]
        return (x, y, float(local[2]), (heading + 180) % 360 - 180)
