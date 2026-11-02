"""The `blender` backend: MuJoCo for physics, Blender's Cycles for pictures.

Cycles renders on the CPU, so this runs on Cloud Run CPU jobs and on any
laptop with bpy installed (pip install bpy==4.5.4, Python 3.11). Each frame,
moving objects are put where physics says, and each camera renders a color
image and an object-id mask. The mask comes from Cycles' object index pass,
which isn't blended at edges, so every pixel is exactly one object's id.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import bpy
import numpy as np
from PIL import Image

from sim import assets
from sim.backends import Picture
from sim.backends.physics import Physics
from sim.geometry import quat_from_euler, quat_looking_at
from sim.objects import Box, Camera, Floor, Light, Prop, Solid, Sphere, Wall
from sim.world import World

# With the denoiser, 16 looks the same as 32 (mean pixel difference under
# 0.5/255) in about 55% of the time. SIM_SAMPLES overrides it.
DEFAULT_SAMPLES = 16
INDOOR_BACKGROUND = ((0.05, 0.05, 0.05), 1.0)


def linear(color) -> tuple:
    """sRGB (what people write) to linear (what Blender's shaders take)."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in color[:3]) + (1.0,)


class Blender(Physics):
    name = "blender"

    def __init__(self, samples: int | None = None):
        self.samples = samples or int(os.environ.get("SIM_SAMPLES", DEFAULT_SAMPLES))
        self.tmp = Path(tempfile.mkdtemp(prefix="sim-blender-"))

    def build(self, world: World) -> None:
        import _cycles  # Cycles' own module, which exists once bpy is loaded

        super().build(world)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        self.scene = scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = self.samples
        scene.cycles.max_bounces = 4
        # OpenImageDenoise cleans up the noise from using few samples.
        scene.cycles.use_denoising = _cycles.with_openimagedenoise
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.seed = world.seed % (2**31)
        # Keeps the scene's geometry between renders, which saves time when
        # only a few things move.
        scene.render.use_persistent_data = True
        scene.render.resolution_percentage = 100
        # Colors come out as written, rather than through a film-like curve.
        scene.view_settings.view_transform = "Standard"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"

        self._background(world)
        self.models: dict[str, bpy.types.Object] = {}  # asset id -> its first Prop's object
        self.objects: dict[int, bpy.types.Object] = {}
        for item in world.solids:
            self.objects[item.id] = self._solid(item)
        for light in world.lights:
            self._light(light)
        self.camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
        scene.collection.objects.link(self.camera)
        scene.camera = self.camera
        self.camera.rotation_mode = "QUATERNION"
        self.camera.data.sensor_fit = "HORIZONTAL"
        self.camera.data.clip_start, self.camera.data.clip_end = 0.01, 1000
        self._mask_output()

    # --- Building --------------------------------------------------------

    def _background(self, world: World) -> None:
        bg = world.background
        color, strength = (bg.color, bg.strength) if bg else INDOOR_BACKGROUND
        sky = bpy.data.worlds.new("background")
        sky.use_nodes = True
        node = sky.node_tree.nodes["Background"]
        node.inputs["Color"].default_value = linear(color)
        node.inputs["Strength"].default_value = float(strength)
        self.scene.world = sky

    def _solid(self, item: Solid) -> bpy.types.Object:
        if isinstance(item, Prop):
            return self._model(item)
        if isinstance(item, Sphere):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=item.radius, segments=48, ring_count=24)
            bpy.ops.object.shade_smooth()
            obj = bpy.context.object
            obj.location, rotation = item.position, item.rotation
        elif isinstance(item, Floor):
            bpy.ops.mesh.primitive_plane_add(size=1)
            obj = bpy.context.object
            obj.scale = (item.size[0], item.size[1], 1)
            apply_scale()
            obj.location, rotation = (0, 0, 0), (0, 0, 0)
        elif isinstance(item, Wall):
            center, size, yaw = item.box()
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.object
            obj.scale = tuple(size)
            apply_scale()
            obj.location, rotation = tuple(center), (0, 0, yaw)
        elif isinstance(item, Box):
            bpy.ops.mesh.primitive_cube_add(size=1)
            obj = bpy.context.object
            obj.scale = tuple(item.size)
            apply_scale()
            obj.location, rotation = item.position, item.rotation
        else:
            raise TypeError(f"The blender backend doesn't know {type(item).__name__}")
        obj.name = item.name
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = tuple(quat_from_euler(rotation))
        obj.pass_index = item.id
        obj.data.materials.append(self._material(item))
        return obj

    def _model(self, item: Prop) -> bpy.types.Object:
        """The model's mesh and textures. Props of the same model share
        them, so ten chairs cost about what one does."""
        first = self.models.get(item.asset)
        if first is None:
            path = assets.fetch(item.asset) / "model.glb"
            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.import_scene.gltf(filepath=str(path))
            meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
            if len(meshes) != 1:
                raise RuntimeError(f"{item.asset}'s model.glb should hold one mesh, not {len(meshes)}")
            obj = self.models[item.asset] = meshes[0]
        else:
            obj = first.copy()  # shares the mesh and materials
            self.scene.collection.objects.link(obj)
        obj.name = item.name
        obj.scale = (float(item.scale),) * 3
        obj.location = tuple(map(float, item.position))
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = tuple(quat_from_euler(item.rotation))
        obj.pass_index = item.id
        if tuple(item.tint[:3]) != (1.0, 1.0, 1.0):
            for slot in obj.material_slots:
                if slot.material is not None:
                    material = slot.material
                    slot.link = "OBJECT"  # this object's own copy, leaving the others
                    slot.material = tinted(material, item.tint, f"{item.name}.{material.name}")
        return obj

    def _material(self, item: Solid) -> bpy.types.Material:
        mat = bpy.data.materials.new(item.name)
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = float(item.roughness)
        bsdf.inputs["Base Color"].default_value = linear(item.color)
        texture = getattr(item, "texture", "plain")
        if texture != "plain":
            color = floor_texture(nodes, links, texture, item.color)
            links.new(color, bsdf.inputs["Base Color"])
        return mat

    def _light(self, light: Light) -> None:
        kind = {"sun": "SUN", "area": "AREA", "point": "POINT", "spot": "SPOT"}[light.kind]
        data = bpy.data.lights.new(light.name, kind)
        data.energy = float(light.intensity)
        data.color = tuple(light.color[:3])
        if light.kind == "area":
            data.shape = "RECTANGLE"
            data.size, data.size_y = map(float, light.size[:2])
        elif light.kind == "sun":
            size = light.size if isinstance(light.size, (int, float)) else 1.0
            data.angle = np.radians(size)
        obj = bpy.data.objects.new(light.name, data)
        self.scene.collection.objects.link(obj)
        obj.location = tuple(light.position)
        obj.rotation_mode = "QUATERNION"
        # Lights shine along their own -z, like cameras look.
        target = np.asarray(light.position, float) + np.asarray(light.direction, float)
        obj.rotation_quaternion = tuple(quat_looking_at(light.position, target))

    def _mask_output(self) -> None:
        """Write the object index pass to an EXR file next to each render."""
        self.scene.view_layers[0].use_pass_object_index = True
        self.scene.use_nodes = True
        tree = self.scene.node_tree
        layers = tree.nodes["Render Layers"]
        out = tree.nodes.new("CompositorNodeOutputFile")
        out.format.file_format = "OPEN_EXR"
        out.format.color_depth = "32"
        out.file_slots[0].path = "ids"
        tree.links.new(layers.outputs["IndexOB"], out.inputs[0])
        self.mask_node = out

    # --- Each frame ------------------------------------------------------

    def render(self, camera: Camera, position, quat) -> Picture:
        for item in self.world.moving():
            obj = self.objects[item.id]
            obj.location = tuple(map(float, item.position))
            obj.rotation_quaternion = tuple(map(float, item.quat))
        width, height = camera.resolution
        self.scene.render.resolution_x, self.scene.render.resolution_y = width, height
        self.camera.data.angle = np.radians(camera.fov)
        self.camera.location = tuple(map(float, position))
        self.camera.rotation_quaternion = tuple(map(float, quat))

        out = self.tmp / "frame"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        self.mask_node.base_path = str(out)
        self.scene.render.filepath = str(out / "rgb.png")
        bpy.ops.render.render(write_still=True)

        with Image.open(out / "rgb.png") as img:
            rgb = np.asarray(img.convert("RGB")).copy()
        (exr,) = out.glob("ids*.exr")
        image = bpy.data.images.load(str(exr))
        try:
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
        finally:
            bpy.data.images.remove(image)
        # Blender's rows start at the bottom.
        ids = pixels.reshape(height, width, -1)[::-1, :, 0]
        return Picture(rgb=rgb, ids=np.rint(ids).astype(np.uint16))

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def apply_scale() -> None:
    """Bake the new object's scale into its mesh, so texture coordinates
    are in meters."""
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def tinted(material: bpy.types.Material, tint, name: str) -> bpy.types.Material:
    """A copy of a material whose base color is multiplied by `tint`."""
    copy = material.copy()
    copy.name = name
    nodes, links = copy.node_tree.nodes, copy.node_tree.links
    for bsdf in [n for n in nodes if n.type == "BSDF_PRINCIPLED"]:
        base = bsdf.inputs["Base Color"]
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        if base.is_linked:
            links.new(base.links[0].from_socket, mix.inputs[6])
        else:
            mix.inputs[6].default_value = base.default_value
        mix.inputs[7].default_value = linear(tint)
        links.new(mix.outputs[2], base)
    return copy


def floor_texture(nodes, links, texture: str, color):
    """A procedural floor texture's color output. Coordinates are in meters."""
    coords = nodes.new("ShaderNodeTexCoord")
    tinted = linear(color)

    def mix(factor, a, b):
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = a
        ramp.color_ramp.elements[1].color = b
        links.new(factor, ramp.inputs["Fac"])
        return ramp.outputs["Color"]

    def multiply(a, b):
        node = nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.blend_type = "MULTIPLY"
        node.inputs["Factor"].default_value = 1.0
        links.new(a, node.inputs[6])
        links.new(b, node.inputs[7])
        return node.outputs[2]

    if texture.startswith("wood"):
        light, dark = {
            "wood": ((0.55, 0.36, 0.17, 1), (0.36, 0.2, 0.08, 1)),
            "wood_dark": ((0.25, 0.14, 0.06, 1), (0.12, 0.06, 0.02, 1)),
            "wood_red": ((0.45, 0.16, 0.07, 1), (0.25, 0.07, 0.03, 1)),
        }[texture]
        planks = nodes.new("ShaderNodeTexBrick")
        planks.offset, planks.squash_frequency = 0.37, 1
        planks.inputs["Scale"].default_value = 1.0
        planks.inputs["Brick Width"].default_value = 1.8
        planks.inputs["Row Height"].default_value = 0.12
        planks.inputs["Mortar Size"].default_value = 0.002
        planks.inputs["Color1"].default_value = light
        planks.inputs["Color2"].default_value = tuple(c * 0.85 for c in light[:3]) + (1,)
        planks.inputs["Mortar"].default_value = dark
        links.new(coords.outputs["Object"], planks.inputs["Vector"])
        grain = nodes.new("ShaderNodeTexWave")
        grain.inputs["Scale"].default_value = 6.0
        grain.inputs["Distortion"].default_value = 4.0
        grain.inputs["Detail"].default_value = 3.0
        links.new(coords.outputs["Object"], grain.inputs["Vector"])
        return multiply(planks.outputs["Color"], mix(grain.outputs["Fac"], (0.8, 0.8, 0.8, 1), (1, 1, 1, 1)))
    if texture in ("concrete", "carpet"):
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 3.0 if texture == "concrete" else 200.0
        noise.inputs["Detail"].default_value = 8.0
        links.new(coords.outputs["Object"], noise.inputs["Vector"])
        dark = tuple(c * 0.6 for c in tinted[:3]) + (1,)
        return mix(noise.outputs["Fac"], dark, tinted)
    if texture == "checker":
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = 1.0
        checker.inputs["Color1"].default_value = tinted
        checker.inputs["Color2"].default_value = (0.8, 0.8, 0.8, 1)
        links.new(coords.outputs["Object"], checker.inputs["Vector"])
        return checker.outputs["Color"]
    if texture == "tile":
        tiles = nodes.new("ShaderNodeTexBrick")
        tiles.offset = 0.0
        tiles.inputs["Scale"].default_value = 1.0
        tiles.inputs["Brick Width"].default_value = 0.3
        tiles.inputs["Row Height"].default_value = 0.3
        tiles.inputs["Mortar Size"].default_value = 0.004
        tiles.inputs["Color1"].default_value = tinted
        tiles.inputs["Color2"].default_value = tuple(c * 0.92 for c in tinted[:3]) + (1,)
        tiles.inputs["Mortar"].default_value = (0.3, 0.3, 0.3, 1)
        links.new(coords.outputs["Object"], tiles.inputs["Vector"])
        return tiles.outputs["Color"]
    raise ValueError(f"Unknown floor texture {texture!r}")
