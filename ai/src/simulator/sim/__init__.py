"""Team 100's simulator framework. See ../README.md."""

import sys
from pathlib import Path

# ai/src, so `shared` (code the simulator and the robot share) imports
# wherever sim runs from.
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.append(_SRC)

from sim.objects import Background, Box, Camera, Floor, Light, Sphere, Wall
from sim.randomness import Range
from sim.world import Scenario, Scene

__all__ = [
    "Background", "Box", "Camera", "Floor", "Light", "Range", "Scenario", "Scene", "Sphere", "Wall",
]
