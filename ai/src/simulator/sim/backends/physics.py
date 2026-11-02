"""The `physics` backend: MuJoCo, and no pictures.

MuJoCo runs anywhere Python does (pip install mujoco), so students can check
a Scenario on their own laptops. The `blender` backend uses this for its
physics too.
"""

from __future__ import annotations

import math
import sys

import mujoco
import numpy as np

from sim import assets
from sim.backends import Pose
from sim.geometry import quat_from_euler
from sim.objects import Box, Floor, GamePiece, Prop, Solid, Sphere, Wall
from sim.world import World

# How long a bounce's contact lasts, in seconds: about 5 physics steps at
# 240 Hz. Shorter sinks things into the floor less, but needs more steps.
CONTACT_TIME = 0.02


def damping_ratio(bounciness: float) -> float:
    """The contact damping ratio that bounces back with this coefficient of
    restitution, for a damped spring: e = exp(-zπ / sqrt(1 - z²))."""
    # Above about 0.95, MuJoCo starts adding energy.
    e = min(max(bounciness, 0.0), 0.95)
    if e <= 0:
        return 1.0
    log_e = math.log(e)
    return -log_e / math.sqrt(math.pi**2 + log_e**2)


def contact_spring(bounciness: float) -> list[float]:
    """MuJoCo's solref for a contact that bounces like this, given directly
    as (-stiffness, -damping). MuJoCo's usual (time, damping ratio) form
    gains energy and flies off when the damping ratio is small."""
    omega = math.pi / CONTACT_TIME
    return [-(omega**2), -2 * damping_ratio(bounciness) * omega]


class Physics:
    name = "physics"

    def build(self, world: World) -> None:
        self.world = world
        spec = mujoco.MjSpec()
        spec.option.timestep = 1 / world.scenario.physics_hz
        self.bodies: dict[int, str] = {}
        self.stand_ins: list[str] = []
        for item in world.solids:
            self._add(spec, item)
        if self.stand_ins:
            print(f"physics: {len(self.stand_ins)} props' files aren't on this computer, so each is "
                  "its box from the catalog. Cloud jobs use their real shapes.", file=sys.stderr)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # Where each moving object's free joint is in qpos and qvel.
        self.qpos, self.qvel = {}, {}
        for item_id, name in self.bodies.items():
            joint = self.model.body_jntadr[self.model.body(name).id]
            self.qpos[item_id] = self.model.jnt_qposadr[joint]
            self.qvel[item_id] = self.model.jnt_dofadr[joint]
        for item in world.moving():
            self.set_velocity(item, item.velocity)
        mujoco.mj_forward(self.model, self.data)

    def _add(self, spec, item: Solid) -> None:
        position, quat = list(map(float, item.position)), list(quat_from_euler(item.rotation))
        if isinstance(item, Prop):
            self._prop(spec, item, position, quat)
            return
        if isinstance(item, Floor):
            sx, sy = item.size
            kind, size, position = mujoco.mjtGeom.mjGEOM_PLANE, [sx / 2, sy / 2, 0.1], [0, 0, 0]
        elif isinstance(item, Wall):
            center, full, yaw = item.box()
            kind, size = mujoco.mjtGeom.mjGEOM_BOX, list(full / 2)
            position, quat = list(center), list(quat_from_euler((0, 0, yaw)))
        elif isinstance(item, Box):
            kind, size = mujoco.mjtGeom.mjGEOM_BOX, [s / 2 for s in item.size]
        elif isinstance(item, Sphere):
            kind, size = mujoco.mjtGeom.mjGEOM_SPHERE, [item.radius, 0, 0]
        else:
            raise TypeError(f"The physics backend doesn't know {type(item).__name__}")

        if item.static:
            geom = spec.worldbody.add_geom(type=kind, size=size, pos=position, quat=quat)
        else:
            name = f"item{item.id}"
            body = spec.worldbody.add_body(name=name, pos=position, quat=quat)
            body.add_freejoint()
            geom = body.add_geom(type=kind, size=size, mass=float(item.mass))
            # Moving things' contact settings win over static things'.
            geom.priority = 1
            self.bodies[item.id] = name
        self._surface(geom, item)

    def _prop(self, spec, item: Prop, position, quat) -> None:
        """One convex mesh geom per collision part: in the world body if it's
        static, else in a body of its own (a GamePiece). MuJoCo collides
        with each part's convex hull."""
        try:
            parts = [p * item.scale for p in assets.collision(item.asset)]
        except assets.MissingAsset:
            if assets.cloud_bucket():
                raise
            self.stand_ins.append(item.name)
            size = item.size()
            corners = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0, 1)])
            parts = [corners * size]
        if item.static:
            for i, vertices in enumerate(parts):
                mesh = spec.add_mesh(name=f"item{item.id}.{i}", uservert=np.asarray(vertices, float).ravel().tolist())
                geom = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                                               pos=position, quat=quat)
                self._surface(geom, item)
            return
        name = f"item{item.id}"
        body = spec.worldbody.add_body(name=name, pos=position, quat=quat)
        body.add_freejoint()
        for i, vertices in enumerate(parts):
            mesh = spec.add_mesh(name=f"item{item.id}.{i}", uservert=np.asarray(vertices, float).ravel().tolist())
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                                 mass=float(item.mass) / len(parts))
            geom.priority = 1
            self._surface(geom, item)
        self.bodies[item.id] = name

    def _surface(self, geom, item: Solid) -> None:
        geom.solref = contact_spring(item.bounciness)
        # Sliding, twisting and rolling friction; rolling friction lets a
        # ball roll to a stop.
        geom.friction = [float(item.friction), 0.005, 0.0001]
        # 6 adds rolling friction, so balls roll to a stop.
        geom.condim = 6 if isinstance(item, (Sphere, GamePiece)) else 3

    def step(self, dt: float) -> None:
        for _ in range(max(1, round(dt / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def poses(self) -> dict[int, Pose]:
        out = {}
        for item_id, name in self.bodies.items():
            body = self.model.body(name).id
            v = self.qvel[item_id]
            out[item_id] = Pose(
                position=self.data.xpos[body].copy(),
                quat=self.data.xquat[body].copy(),
                velocity=self.data.qvel[v : v + 3].copy(),
                # A free joint's angular velocity is in the body's own frame.
                spin=self.data.xmat[body].reshape(3, 3) @ self.data.qvel[v + 3 : v + 6],
            )
        return out

    def _moving(self, item: Solid) -> None:
        if item.id not in self.bodies:
            raise RuntimeError(f"{item.name} is static, so it can't move. Make it static=False.")

    def set_pose(self, item: Solid, position, quat) -> None:
        self._moving(item)
        a = self.qpos[item.id]
        self.data.qpos[a : a + 3] = position
        self.data.qpos[a + 3 : a + 7] = np.asarray(quat) / np.linalg.norm(quat)
        mujoco.mj_forward(self.model, self.data)

    def set_velocity(self, item: Solid, velocity) -> None:
        self._moving(item)
        v = self.qvel[item.id]
        self.data.qvel[v : v + 3] = velocity

    def push(self, item: Solid, impulse) -> None:
        self._moving(item)
        v = self.qvel[item.id]
        mass = self.model.body_mass[self.model.body(self.bodies[item.id]).id]
        self.data.qvel[v : v + 3] += np.asarray(impulse, dtype=float) / mass

    def render(self, camera, position, quat):
        return None

    def close(self) -> None:
        pass
