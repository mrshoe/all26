"""Import FIRST's official field CAD for a season into the asset catalog.
For admins: it writes to the bucket, which students can only read.

    python -m sim.assets.frc 2026 [--only hub_red fuel ...] [--upload]

On kickoff day, write sim/assets/frc/<year>.toml (copy last year's: the
STEP's URL, WPILib's layout, the game piece's part number and mass), then
run this. It:

1. downloads FIRST's STEP of the full field, and converts it to glTF with
   OpenCascade, keeping the assembly, part names and colors (seconds);
2. splits it at the top level: each placed element (HUB, TOWER, ...)
   becomes a model of its own, named by the config and by alliance
   (hub_red, hub_blue), and each game piece one model, with every starting
   position;
3. drops what neither cameras nor physics need: hardware (rivets, bolts,
   ...) by name, parts under 2 cm, and parts hidden inside an element (no
   ray from them gets out), like the HUB's fuel counter;
4. draws each AprilTag's real 36h11 image on its vinyl, facing the way
   WPILib's layout says, and works out how the CAD's coordinates map to
   WPILib's from the tags' positions;
5. cuts each model to TRIANGLES and gives physics one convex hull per kept
   part (CoACD for the few that are far from convex);
6. writes each model to the cache folder (sim/assets) as model.glb and
   collision.npz, renders thumbnails and an overview, and writes the
   catalog, frc/<year>.json, last. --upload copies the models and the STEP
   to gs://<bucket>/assets/frc/<year>/. Commit the catalog once uploaded.

Needs bpy (as for rendering), and `pip install coacd cadquery-ocp
opencv-python-headless==4.10.0.84` (that OpenCV for numpy 1.26, which bpy
needs). --upload needs gcloud signed in as an admin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sim.assets import FILES, FRC, default_cache, md5

# Most triangles a model may have: elements are big but few are in view at
# once; game pieces are many.
TRIANGLES = {"element": 100_000, "piece": 2_000}
# Parts smaller than this across (m) are dropped: nuts, PEMs, tiny brackets.
SMALLEST = 0.02
# Parts thinner than this (m) have no collision: tape, vinyl, stickers.
FLAT = 0.002
HARDWARE = re.compile(
    r"\b(rivets?|pem|bolts?|nuts?|screws?|washers?|cable ties?|zip ties?|spacers?|standoffs?|bearings?"
    r"|shaft collars?|bushings?)\b", re.I)
TAG_VINYL = re.compile(r"AprilTag Vinyl", re.I)
TAG_ID = re.compile(r"\bID\s*(\d+)")
# The black square of an FRC tag (36h11 since 2024): 6.5 in.
TAG_SIZE = 0.1651
# Names in the CAD end with an instance number, "<2>", and Blender adds ".001".
INSTANCE = re.compile(r"(\s*<\d+>)?(\.\d+)?$")
HEADERS = {"User-Agent": "wildhat-sim (FRC team 100 asset import)"}


# --- Config and downloads --------------------------------------------------


def load_config(year: int) -> dict:
    path = FRC / f"{year}.toml"
    if not path.exists():
        raise SystemExit(f"No {path}. Copy last year's and change it for {year}.")
    config = tomllib.loads(path.read_text())
    config.setdefault("elements", {})
    config.setdefault("pieces", {})
    if config.get("red") not in ("-x", "+x"):
        raise SystemExit(f"{path}: red must be \"-x\" or \"+x\"")
    return config


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        print(f"Downloading {url}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers=HEADERS)
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(req, timeout=600) as resp, partial.open("wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        partial.replace(path)
    return path


def step_to_glb(step: str, glb: str) -> None:
    """OpenCascade's STEP reader and glTF writer: the assembly tree, part
    names and colors, in meters, z up. Run in a process of its own: it
    needs a few GB for a full field."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.Message import Message_ProgressRange
    from OCP.OCP.collections import IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString as FileInfo
    from OCP.RWGltf import RWGltf_CafWriter
    from OCP.RWMesh import RWMesh_CoordinateSystem
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(step) != 1:
        raise RuntimeError(f"OpenCascade couldn't read {step}")
    reader.Transfer(doc)
    # Meshing the whole shape meshes the document's parts too: they share faces.
    # 2 mm deviation (OpenCascade works in mm), 0.5 rad between facets.
    BRepMesh_IncrementalMesh(reader.ChangeReader().OneShape(), 2.0, False, 0.5, True)
    writer = RWGltf_CafWriter(TCollection_AsciiString(glb), True)
    converter = writer.ChangeCoordinateSystemConverter()
    converter.SetInputLengthUnit(0.001)
    converter.SetInputCoordinateSystem(RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup)
    if not writer.Perform(doc, FileInfo(), Message_ProgressRange()):
        raise RuntimeError(f"OpenCascade couldn't write {glb}")


# --- Naming (no Blender needed, so it's tested on its own) ------------------


def base_name(name: str) -> str:
    """A CAD node's name without its instance number: "GE-26300: Hub"."""
    return INSTANCE.sub("", name).strip()


def matching(name: str, table: dict) -> str | None:
    """The longest key in `table` that `name` starts with."""
    keys = [k for k in table if name.startswith(k)]
    return max(keys, key=len) if keys else None


def derived_name(name: str) -> str:
    """A model name from a CAD name nobody named: "FE-2026-02: Welded
    Perimeter Assy" -> "welded_perimeter_assy"."""
    words = name.split(":", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "_", words.lower()).strip("_") or "part"


def alliance_of(x: float, red: str) -> str | None:
    """Red, blue, or neither, from where an element's center is along x."""
    if abs(x) < 1.0:
        return None
    return "red" if (x < 0) == (red == "-x") else "blue"


def plan_assets(nodes: list[tuple[str, tuple[float, float]]], config: dict) -> tuple[dict, dict, list[str]]:
    """Which top-level CAD nodes make which models. `nodes` is each node's
    name and the (x, y) of its center. Returns {model name: {"nodes": [...],
    "element", "alliance", "label"}}, {piece name: [node indexes]}, and
    warnings."""
    pieces: dict[str, list[int]] = defaultdict(list)
    by_name: dict[tuple, list[int]] = defaultdict(list)
    info: dict[tuple, dict] = {}
    warnings = []
    part_of = {spec["part"]: name for name, spec in config["pieces"].items()}
    for i, (name, (x, y)) in enumerate(nodes):
        base = base_name(name)
        piece = matching(base, part_of)
        if piece:
            pieces[part_of[piece]].append(i)
            continue
        key = matching(base, config["elements"])
        spec = config["elements"][key] if key else {}
        element = spec.get("name") or derived_name(base)
        if not key:
            warnings.append(f"{base!r} isn't in the config's [elements]; calling it {element!r}")
        alliance = alliance_of(x, config["red"]) if spec.get("alliance", True) else None
        group = (element, alliance) if spec.get("merge") else (element, alliance, i)
        by_name[group].append(i)
        info[group] = {"element": element, "alliance": alliance, "label": spec.get("label")}
    # Number elements that appear more than once on a side, in order of y then x.
    counts = Counter((g[0], g[1]) for g in by_name)
    models, seen = {}, Counter()
    for group in sorted(by_name, key=lambda g: (g[0], g[1] or "", nodes[by_name[g][0]][1][1], nodes[by_name[g][0]][1][0])):
        element, alliance = group[0], group[1]
        name = element + (f"_{alliance}" if alliance else "")
        if counts[(element, alliance)] > 1:
            name += f"_{seen[(element, alliance)]}"
            seen[(element, alliance)] += 1
        models[name] = {"nodes": by_name[group], **info[group]}
    return models, dict(pieces), warnings


def fit_wpilib(cad: dict[int, tuple], wpilib: dict[int, tuple]) -> dict:
    """The turn about z and offset that take the CAD's (x, y) to WPILib's,
    from the tags both have. Returns {"yaw": degrees, "offset": [x, y],
    "error": RMS meters, "tags": n}."""
    import numpy as np

    ids = sorted(set(cad) & set(wpilib))
    if len(ids) < 3:
        raise RuntimeError(f"Only {len(ids)} AprilTags are in both the CAD and WPILib's layout")
    a = np.array([cad[i][:2] for i in ids], float)
    b = np.array([wpilib[i][:2] for i in ids], float)
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    turn = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    offset = cb - turn @ ca
    error = float(np.sqrt(np.mean(np.sum((a @ turn.T + offset - b) ** 2, axis=1))))
    return {"yaw": round(math.degrees(yaw), 4), "offset": [round(float(v), 5) for v in offset],
            "error": round(error, 4), "tags": len(ids)}


# --- In Blender ---------------------------------------------------------------


def world_mesh(obj):
    """A mesh object's vertices in world coordinates, its triangles, each
    triangle's material, and each triangle corner's normal."""
    import numpy as np

    mesh = obj.data
    mesh.calc_loop_triangles()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    verts = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    mat = np.empty(len(mesh.loop_triangles), dtype=np.int64)
    mesh.loop_triangles.foreach_get("material_index", mat)
    slots = [s.material for s in obj.material_slots] or [None]
    materials = [slots[min(i, len(slots) - 1)] for i in mat]
    return verts, tris.reshape(-1, 3), materials


def transparent(material) -> bool:
    if material is None or not material.use_nodes:
        return False
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    return bool(bsdf) and bsdf.inputs["Alpha"].default_value < 0.95


def descendants(obj) -> list:
    out = [obj] if obj.type == "MESH" else []
    for child in obj.children:
        out += descendants(child)
    return out


def tag_id(obj) -> int | None:
    o = obj
    while o is not None:
        found = TAG_ID.search(o.name)
        if found:
            return int(found.group(1))
        o = o.parent
    return None


# Seventeen directions: the upper half of a cube's faces, edges and corners.
# Nothing on the field is seen from below the carpet.
def directions():
    import numpy as np

    out = [np.array(d, float) for d in ((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (0, 1))
           if any(d)]
    return [d / np.linalg.norm(d) for d in out]


def hidden(parts: list[dict]) -> set[int]:
    """Parts no ray gets out from: from up to 16 of a part's vertices, in 17
    directions, a ray that passes only through the part itself or through
    transparent parts and leaves means it can be seen."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    verts, polys, owner = [], [], []
    for i, part in enumerate(parts):
        start = sum(len(v) for v in verts)
        verts.append(part["verts"])
        polys.append(part["tris"] + start)
        owner += [i] * len(part["tris"])
    tree = BVHTree.FromPolygons(np.concatenate(verts).tolist(), np.concatenate(polys).tolist(), all_triangles=True)
    see_through = [all(transparent(m) for m in part["materials"]) for part in parts]
    dirs = [Vector(d) for d in directions()]
    rng = np.random.default_rng(0)
    out = set()
    for i, part in enumerate(parts):
        v = part["verts"]
        # The extremes along each axis, then random vertices.
        picks = {int(v[:, a].argmin()) for a in range(3)} | {int(v[:, a].argmax()) for a in range(3)}
        picks |= set(rng.choice(len(v), size=min(len(v), 10), replace=False).tolist())
        if not any(escapes(tree, owner, see_through, i, Vector(v[p]), d) for p in picks for d in dirs):
            out.add(i)
    return out


def escapes(tree, owner, see_through, me: int, start, direction) -> bool:
    origin = start + direction * 1e-4
    for _ in range(16):
        hit, _normal, index, _distance = tree.ray_cast(origin, direction)
        if hit is None:
            return True
        o = owner[index]
        if o != me and not see_through[o]:
            return False
        origin = hit + direction * 1e-4
    return False


def build_mesh(name: str, parts: list[dict]):
    """One Blender object from parts' world-space triangles, keeping each
    part's materials. Smooth within a part, sharp at its creases."""
    import bpy
    import numpy as np

    materials, verts, tris, mat_index = [], [], [], []
    slot = {}
    start = 0
    for part in parts:
        verts.append(part["verts"])
        tris.append(part["tris"] + start)
        start += len(part["verts"])
        for m in part["materials"]:
            if id(m) not in slot:
                slot[id(m)] = len(materials)
                materials.append(m)
            mat_index.append(slot[id(m)])
    mesh = bpy.data.meshes.new(name)
    v = np.concatenate(verts)
    t = np.concatenate(tris)
    mesh.vertices.add(len(v))
    mesh.vertices.foreach_set("co", v.ravel())
    mesh.loops.add(len(t) * 3)
    mesh.loops.foreach_set("vertex_index", t.ravel())
    mesh.polygons.add(len(t))
    mesh.polygons.foreach_set("loop_start", np.arange(0, len(t) * 3, 3))
    mesh.polygons.foreach_set("material_index", np.array(mat_index))
    mesh.update()
    mesh.validate()
    for m in materials:
        mesh.materials.append(m)
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
    mesh.set_sharp_from_angle(angle=math.radians(35))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def decimate(obj, budget: int) -> None:
    import bpy

    triangles = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    if triangles <= budget:
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("decimate", "DECIMATE")
    mod.ratio = budget / triangles
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)


def tag_image(tag: int, vinyl: float, cell: int = 32):
    """A white square `vinyl` meters across, with tag `tag` (36h11)
    centered on it at its real size."""
    import bpy
    import cv2
    import numpy as np

    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                           tag, 8 * cell, borderBits=1)
    side = int(round(vinyl / TAG_SIZE * 8 * cell))
    canvas = np.full((side, side), 255, np.uint8)
    at = (side - 8 * cell) // 2
    canvas[at : at + 8 * cell, at : at + 8 * cell] = marker
    rgba = np.ones((side, side, 4), np.float32)
    # Blender's pixel rows go bottom to top.
    rgba[..., :3] = (canvas[::-1, :, None] / 255.0)
    image = bpy.data.images.new(f"apriltag_{tag}", side, side)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def tag_material(tag: int, vinyl: float):
    import bpy

    mat = bpy.data.materials.new(f"apriltag_{tag}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = tag_image(tag, vinyl)
    tex.interpolation = "Closest"  # crisp cell edges
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def tag_object(part: dict, tag: int, facing):
    """The vinyl as its own object, with UVs that put the tag upright,
    facing `facing` (a unit vector in the CAD's coordinates)."""
    import bpy
    import numpy as np

    obj = build_mesh(f"apriltag_{tag}", [part])
    v = part["verts"]
    center = (v.min(0) + v.max(0)) / 2
    size = float(max(v.max(0) - v.min(0)))
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(-np.asarray(facing, float), up)
    right /= np.linalg.norm(right)
    mesh = obj.data
    uv = mesh.uv_layers.new(name="UVMap")
    index = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", index)
    p = v[index] - center
    coords = np.column_stack([0.5 + p @ right / size, 0.5 + p @ up / size])
    uv.data.foreach_set("uv", coords.ravel())
    mesh.materials.clear()
    mesh.materials.append(tag_material(tag, size))
    mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), dtype=np.int64))
    return obj


def collision_parts(parts: list[dict], origin, pool) -> list:
    """One convex hull per part, or CoACD's pieces for a part that's far
    from convex (a funnel, a bent rail), split across `pool`'s processes.
    Thin parts get none."""
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    out, jobs = [], []
    for part in parts:
        v = part["verts"] - origin
        extent = v.max(0) - v.min(0)
        if extent.min() < FLAT:
            continue
        try:
            hull = ConvexHull(v)
        except QhullError:
            continue
        a, b, c = (v[part["tris"][:, k]] for k in range(3))
        volume = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6
        if extent.max() > 0.15 and volume > 0 and hull.volume / volume > 3:
            jobs.append((part["name"], v, hull, pool.submit(decompose, v, part["tris"])))
        else:
            out.append(v[hull.vertices].astype(np.float32))
    for name, v, hull, job in jobs:
        try:
            out += job.result()
        except Exception as e:  # CoACD fails on some broken meshes
            print(f"  CoACD failed on {name} ({e}); using its hull", flush=True)
            out.append(v[hull.vertices].astype(np.float32))
    return out


def decompose(verts, tris) -> list:
    """CoACD's convex pieces of one part. In a worker process."""
    import coacd
    import numpy as np

    coacd.set_log_level("error")
    pieces = coacd.run_coacd(coacd.Mesh(verts, tris), threshold=0.05, max_convex_hull=8, seed=0)
    return [np.asarray(p, np.float32) for p, _ in pieces]


def export(obj, folder: Path, parts: list) -> None:
    import bpy
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(folder / "model.glb"), export_format="GLB", use_selection=True,
                              export_image_format="AUTO", export_animations=False, export_extras=False)
    np.savez_compressed(folder / "collision.npz", **{f"part{i}": p for i, p in enumerate(parts)})


def make_model(name: str, kind: str, objects: list, tag_facing: dict, out: Path, pool) -> dict:
    """One model from these CAD mesh objects: drop, simplify, tag, center,
    save. Returns its catalog entry (and "position": where it goes)."""
    import bpy
    import numpy as np

    dropped = Counter()
    parts, tags = [], []
    for obj in objects:
        verts, tris, materials = world_mesh(obj)
        if not len(tris):
            continue
        part = {"name": obj.name, "verts": verts, "tris": tris, "materials": materials}
        if TAG_VINYL.search(obj.name) and tag_id(obj) is not None:
            tags.append((part, tag_id(obj)))
        elif HARDWARE.search(obj.name):
            dropped["hardware"] += 1
        elif kind == "element" and float((verts.max(0) - verts.min(0)).max()) < SMALLEST:
            dropped["small"] += 1
        else:
            parts.append(part)
    if kind == "element" and len(parts) > 1:
        inside = hidden(parts + [t for t, _ in tags])
        dropped["hidden"] = len([i for i in inside if i < len(parts)])
        parts = [p for i, p in enumerate(parts) if i not in inside]
    source = sum(len(p["tris"]) for p in parts) + sum(len(t["tris"]) for t, _ in tags)

    everything = np.concatenate([p["verts"] for p in parts] + [t["verts"] for t, _ in tags])
    lo, hi = everything.min(0), everything.max(0)
    # Elements keep their height (they stand on the carpet at z = 0);
    # pieces stand on their lowest point, like any Prop.
    origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] if kind == "piece" else 0.0])

    body = build_mesh(name, parts)
    decimate(body, TRIANGLES[kind] - sum(len(t["tris"]) for t, _ in tags))
    pieces = [tag_object(t, i, tag_facing[i]) for t, i in tags if i in tag_facing]
    if pieces:
        bpy.ops.object.select_all(action="DESELECT")
        for o in [body, *pieces]:
            o.select_set(True)
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.join()
    body.data.transform(_translation(-origin))
    triangles = sum(len(p.vertices) - 2 for p in body.data.polygons)
    hulls = collision_parts(parts, origin, pool)
    folder = out / name
    export(body, folder, hulls)
    size = (hi - lo).tolist()
    bpy.data.objects.remove(body)
    return {
        "kind": kind,
        "size": [round(s, 4) for s in size],
        "triangles": triangles,
        "source_triangles": source,
        "parts": len(hulls),
        "dropped": dict(dropped),
        "tags": sorted(i for _, i in tags),
        "files": {f: md5(folder / f) for f in FILES},
        "position": [round(float(x), 5) for x in origin],
    }


def _translation(offset):
    from mathutils import Matrix, Vector

    return Matrix.Translation(Vector(tuple(float(x) for x in offset)))


def node_center(node) -> tuple[float, float]:
    import numpy as np
    from mathutils import Vector

    corners = [np.array(m.matrix_world @ Vector(c)) for m in descendants(node) for c in m.bound_box]
    if not corners:
        return (0.0, 0.0)
    c = np.array(corners)
    mid = (c.min(0) + c.max(0)) / 2
    return float(mid[0]), float(mid[1])


def import_field(glb: Path, config: dict, layout: dict, out: Path, only: set[str] | None, pool) -> dict:
    """Everything that happens in Blender (CoACD runs in `pool`). Returns
    the catalog."""
    import bpy
    import numpy as np
    from scipy.spatial.transform import Rotation

    bpy.ops.wm.read_factory_settings(use_empty=True)
    started = time.time()
    bpy.ops.import_scene.gltf(filepath=str(glb))
    roots = [o for o in bpy.context.scene.objects if o.parent is None]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one top-level node in the field, found {len(roots)}")
    nodes = list(roots[0].children)
    print(f"Loaded {len(nodes)} top-level nodes in {time.time() - started:.0f}s", flush=True)

    # AprilTags: where the CAD has them, matched to WPILib's layout.
    cad_tags = defaultdict(list)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and TAG_VINYL.search(obj.name) and tag_id(obj) is not None:
            v, _, _ = world_mesh(obj)
            cad_tags[tag_id(obj)].append((v.min(0) + v.max(0)) / 2)
    cad_centers = {i: tuple(np.mean(c, axis=0)) for i, c in cad_tags.items()}
    wpilib = {t["ID"]: t["pose"] for t in layout["field-tags"]}
    fit = fit_wpilib(cad_centers, {i: (p["translation"]["x"], p["translation"]["y"]) for i, p in wpilib.items()})
    print(f"CAD -> WPILib: turn {fit['yaw']} deg, offset {fit['offset']}, {fit['tags']} tags within "
          f"{fit['error'] * 100:.1f} cm", flush=True)
    if fit["error"] > 0.05:
        print("WARNING: the tags don't line up with WPILib's layout. Is it the right field variant?", flush=True)
    back = Rotation.from_euler("z", -fit["yaw"], degrees=True)
    tag_facing = {}
    for i, pose in wpilib.items():
        q = pose["rotation"]["quaternion"]
        out_of_face = Rotation.from_quat([q["X"], q["Y"], q["Z"], q["W"]]).apply([1.0, 0.0, 0.0])
        tag_facing[i] = back.apply(out_of_face)

    models, pieces, warnings = plan_assets([(n.name, node_center(n)) for n in nodes], config)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    year = config["year"]
    catalog = {"source": f"frc/{year}", "year": year, "game": config.get("game", ""),
               "step": {"url": config["field"]},
               "field": {"length": layout["field-dimensions"]["length"],
                         "width": layout["field-dimensions"]["width"],
                         "wpilib": fit, "layout": config["layout"]},
               "assets": {}, "layout": [], "pieces": {}}

    for name, model in models.items():
        if only and name not in only:
            continue
        started = time.time()
        objects = [m for i in model["nodes"] for m in descendants(nodes[i])]
        entry = make_model(name, "element", objects, tag_facing, out, pool)
        position = entry.pop("position")
        catalog["assets"][name] = {"label": model["label"], "category": f"frc/{year}/element",
                                   "element": model["element"], "alliance": model["alliance"], **entry}
        catalog["layout"].append({"asset": name, "position": position})
        print(f"{name}: {entry['source_triangles']} -> {entry['triangles']} triangles, {entry['parts']} collision "
              f"parts, dropped {entry['dropped']}, tags {entry['tags']} ({time.time() - started:.0f}s)", flush=True)

    for name, indexes in pieces.items():
        spec = config["pieces"][name]
        starts = []
        for i in indexes:
            v = np.concatenate([world_mesh(m)[0] for m in descendants(nodes[i])])
            lo, hi = v.min(0), v.max(0)
            starts.append([round(float(x), 5) for x in ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])])
        if only and name not in only:
            continue
        entry = make_model(name, "piece", descendants(nodes[indexes[0]]), tag_facing, out, pool)
        entry.pop("position")
        catalog["assets"][name] = {"label": spec.get("label", name), "category": f"frc/{year}/piece",
                                   "element": name, "alliance": None, **entry,
                                   "physics": {k: spec[k] for k in ("mass", "bounciness", "friction")}}
        catalog["pieces"][name] = {"asset": name, "start": starts}
        print(f"{name}: a game piece, {entry['triangles']} triangles, {len(starts)} starting positions", flush=True)
    return catalog


def thumbnails(catalog: dict, out: Path, thumbs: Path) -> None:
    """Each model on its own, and the whole field from above a corner."""
    import bpy
    import numpy as np

    from sim.assets.polyhaven import thumbnail
    from sim.geometry import quat_looking_at

    for name, model in catalog["assets"].items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(out / name / "model.glb"))
        obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
        thumbnail(obj, model["size"], thumbs / f"{name}.png")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    placed = [(e["asset"], e["position"]) for e in catalog["layout"]]
    for piece in catalog["pieces"].values():
        placed += [(piece["asset"], p) for p in piece["start"]]
    first = {}
    for name, position in placed:
        if name not in catalog["assets"]:
            continue
        if name in first:
            obj = first[name].copy()
            bpy.context.scene.collection.objects.link(obj)
        else:
            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.import_scene.gltf(filepath=str(out / name / "model.glb"))
            obj = first[name] = [o for o in bpy.context.selected_objects if o.type == "MESH"][0]
        obj.location = tuple(position)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 1600, 900
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3
    sun.rotation_euler = (0.5, 0.2, 0.8)
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    position, target = np.array([9.0, -13.0, 9.0]), np.array([0.0, 0.0, 0.0])
    cam.location = tuple(position)
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = tuple(quat_looking_at(position, target))
    cam.data.lens = 24
    scene.render.filepath = str(thumbs / "field.png")
    bpy.ops.render.render(write_still=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.assets.frc", description=__doc__.split("\n\n")[0])
    parser.add_argument("year", type=int)
    parser.add_argument("--only", nargs="*", help="just these models, e.g. hub_red fuel")
    parser.add_argument("--work", default=str(Path.home() / ".cache" / "wildhat" / "frc-downloads"),
                        help="where FIRST's files, the converted field and the thumbnails go")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="processes for CoACD")
    parser.add_argument("--no-thumbnails", action="store_true")
    parser.add_argument("--upload", action="store_true", help="copy the models to the bucket, then write the catalog")
    args = parser.parse_args(argv)

    config = load_config(args.year)
    work = Path(args.work) / str(args.year)
    step = fetch(config["field"], work / Path(config["field"]).name)
    layout = json.loads(fetch(config["layout"], work / Path(config["layout"]).name).read_text())
    step_md5 = md5(step)
    glb = work / "field.glb"
    stamp = work / "field.glb.from"
    if not glb.exists() or not stamp.exists() or stamp.read_text() != step_md5:
        started = time.time()
        with ProcessPoolExecutor(1) as pool:
            pool.submit(step_to_glb, str(step), str(glb)).result()
        stamp.write_text(step_md5)
        print(f"Converted {step.name} to glTF in {time.time() - started:.0f}s", flush=True)

    out = default_cache() / "frc" / str(args.year)
    with ProcessPoolExecutor(args.workers) as pool:
        catalog = import_field(glb, config, layout, out, set(args.only) if args.only else None, pool)
    catalog["step"]["md5"] = step_md5
    if args.only:
        # Keep the models this run didn't touch.
        path = FRC / f"{args.year}.json"
        old = json.loads(path.read_text()) if path.exists() else {"assets": {}, "layout": [], "pieces": {}}
        done = set(catalog["assets"])
        catalog["assets"] = {**{k: v for k, v in old["assets"].items() if k not in done}, **catalog["assets"]}
        catalog["layout"] = [e for e in old["layout"] if e["asset"] not in done] + catalog["layout"]
        catalog["pieces"] = {**{k: v for k, v in old["pieces"].items() if k not in done}, **catalog["pieces"]}
    if not args.no_thumbnails:
        thumbs = work / "thumbnails"
        thumbnails(catalog, out, thumbs)
        from sim.assets.polyhaven import contact_sheets

        sheets = contact_sheets(sorted(catalog["assets"]), catalog["assets"], thumbs, work / "sheets")
        print(f"Thumbnails: {', '.join(map(str, sheets))}, and the field: {thumbs / 'field.png'}")

    if args.upload:
        from sim.assets.polyhaven import bucket_name

        dest = f"gs://{bucket_name()}/assets/frc/{args.year}"
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(out), f"{dest}/"], check=True)
        subprocess.run(["gcloud", "storage", "cp", str(step), f"{dest}/source/{step.name}"], check=True)
    path = FRC / f"{args.year}.json"
    path.write_text(json.dumps(catalog, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {path}" + (f" and uploaded to {dest}/. Commit it." if args.upload else
                             ". Run again with --upload, then commit it."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
