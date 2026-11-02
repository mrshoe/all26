"""Import Poly Haven's models (CC0) into the asset catalog. For admins: it
writes to the bucket, which students can only read.

    python -m sim.assets.polyhaven [--only ID ...] [--workers 4] [--upload]

For each model that fits (below), it downloads the glTF with 1k textures,
then in Blender joins it into one mesh standing on z = 0, centered in x and
y, cuts it down to at most TRIANGLES triangles, and saves `model.glb`. CoACD
splits that into convex parts for physics (`collision.npz`), and Cycles
renders a thumbnail. The files go to the cache folder (see sim/assets), so
this computer can render with them straight away, and --upload copies them
to gs://<bucket>/assets/polyhaven/. The catalog, polyhaven.json, is written
last; commit it once it's uploaded. Models already imported from the same
Poly Haven files are skipped, so it can be re-run to pick up new ones.

Needs bpy (as for rendering), `pip install coacd`, and for --upload, gcloud
signed in as an admin.

What fits: a model that would stand in a room and is couch-sized or smaller
(SIZE_LIMITS), outside the categories in SKIP (landscapes, buildings,
vehicles, weapons, things that hang on walls or ceilings), and not rigged.
EXCLUDE drops models that pass those but look wrong once imported; LABELS
turns Poly Haven's categories into our labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sim.assets import CATALOG, FILES, SOURCE, default_cache, md5

API = "https://api.polyhaven.com"
# Poly Haven turns away Python's default user agent, and asks API users to
# say who they are.
HEADERS = {"User-Agent": "wildhat-sim (FRC team 100 asset import)"}
TRIANGLES = 20_000
RESOLUTION = "1k"
# Meters: the longest side across the floor, and the height.
SIZE_LIMITS = {"footprint": (0.1, 2.5), "height": (0.0, 2.2)}
# Category prefixes (Poly Haven's paths) that never fit.
SKIP = (
    "Architecture",
    "Vehicles & Transport",
    "Weapons",
    "Decor & Art/Wall Decor",
    "Lighting/Ceiling",
    "Lighting/Wall",
    "Nature/Ground Cover & Moss",
    "Nature/Rocks & Stone/Cliffs & Rock Faces",
)
# Models that pass the rules but don't work once imported, and why.
EXCLUDE: dict[str, str] = {
    key: "hangs on a wall or ceiling"
    for key in ("ceiling_fan", "pull_chain_light_socket", "street_lamp_02", "brass_diya_lantern",
                "garden_hose_wall_mounted_01")
}
# Models whose label isn't their category's, by id: groups of small plants
# spread over the ground aren't "a plant".
LABEL_OVERRIDES = {
    key: "ground_plants"
    for key in ("calathea_orbifolia_01", "cheiridopsis_succulent", "crystalline_iceplant", "flower_empodium",
                "flower_gazania", "flower_stinkkruid", "flower_ursinia", "leipoldtia_schultzei", "nettle_plant",
                "periwinkle_plant", "shrub_03", "shrub_04", "shrub_sorrel_01", "fern_02", "wild_rooibos_bush")
}
# The last part of a category path, to our label. Everything else is "prop";
# Scenarios can still pick by category.
LABELS = {
    "Chairs": "chair",
    "Stools & Bar Seating": "stool",
    "Benches": "bench",
    "Outdoor Seating": "bench",
    "Sofas & Couches": "couch",
    "Ottomans & Footstools": "ottoman",
    "Beds": "bed",
    "Coffee Tables": "table",
    "Console Tables": "table",
    "Dining Tables": "table",
    "Outdoor Tables": "table",
    "Side & End Tables": "table",
    "Desks": "desk",
    "Cabinets & Cupboards": "cabinet",
    "Drawers & Dressers": "cabinet",
    "Sideboards": "cabinet",
    "Shelving & Bookcases": "shelf",
    "Carts & Trolleys": "cart",
    "Potted Plants": "plant",
    "Shrubs & Bushes": "plant",
    "Succulents & Cacti": "plant",
    "Tropical Plants": "plant",
    "Ferns": "plant",
    "Desert & Arid Plants": "plant",
    "Wildflowers": "plant",
    "Saplings": "plant",
    "Trash Cans": "trash_can",
    "Bins": "trash_can",
    "Cardboard Boxes": "box",
    "Wooden Crates": "box",
    "Plastic Crates": "box",
    "Military Crates": "box",
    "Metal Drums": "barrel",
    "Plastic Drums": "barrel",
    "Wooden Barrels": "barrel",
    "Balls": "sports_ball",
    "Boulders": "rock",
    "Desert Boulders": "rock",
    "Coastal Rocks": "rock",
    "Extraterrestrial Rocks": "rock",
    "Stone Sets": "rock",
}


def get(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def label_for(category: str, key: str = "") -> str:
    return LABEL_OVERRIDES.get(key) or LABELS.get(category.rsplit("/", 1)[-1], "prop")


def largest_side(model: dict) -> float:
    """In meters, from a model's listing."""
    return max(model.get("dimensions", (0, 0, 0))) / 1000


def choose(listing: dict) -> tuple[dict, dict]:
    """Split Poly Haven's model listing into what fits (id -> its listing)
    and what doesn't (id -> why not)."""
    keep, skipped = {}, {}
    for key, model in sorted(listing.items()):
        category = model.get("category", "")
        x, y, z = (d / 1000 for d in model.get("dimensions", (0, 0, 0)))
        footprint = max(x, y)
        if "rigged" in model.get("categories", []):
            skipped[key] = "rigged"
        elif any(category == s or category.startswith(s + "/") for s in SKIP):
            skipped[key] = f"category {category}"
        elif key in EXCLUDE:
            skipped[key] = EXCLUDE[key]
        elif not SIZE_LIMITS["footprint"][0] <= footprint <= SIZE_LIMITS["footprint"][1]:
            skipped[key] = f"{footprint:.2f} m across"
        elif not SIZE_LIMITS["height"][0] <= z <= SIZE_LIMITS["height"][1]:
            skipped[key] = f"{z:.2f} m tall"
        else:
            keep[key] = model
    return keep, skipped


def unit_scale(measured: float, listed: float) -> float:
    """What to scale a model by so its largest side matches Poly Haven's
    listing. A few files aren't in meters, so they're off by a power of ten;
    anything else that far off is a mistake we can't fix."""
    if not listed or 0.8 <= measured / listed <= 1.25:
        return 1.0
    power = round(math.log10(measured / listed))
    if power and 0.8 <= measured / listed / 10**power <= 1.25:
        return 10.0**-power
    raise RuntimeError(f"it measures {measured:.2f} m across, but Poly Haven lists {listed:.2f} m")


def download_gltf(key: str, folder: Path) -> Path:
    files = get(f"{API}/files/{key}")["gltf"][RESOLUTION]["gltf"]
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {f"{key}.gltf": files} | files.get("include", {})
    for name, f in wanted.items():
        path = folder / name
        if path.exists() and md5(path) == f["md5"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f["url"], headers=HEADERS)
        with urllib.request.urlopen(req, timeout=300) as resp, path.open("wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        if md5(path) != f["md5"]:
            raise RuntimeError(f"{key}: {name} doesn't match Poly Haven's md5")
    return folder / f"{key}.gltf"


# --- In a worker process (bpy and coacd) ---------------------------------


def process(key: str, listed: float, work: Path, out: Path, thumbs: Path) -> dict:
    """Download, clean up and save one model. Returns what the catalog
    needs to know about it. `listed`: its largest side in meters, by
    Poly Haven's listing (0 if it has none)."""
    import bpy
    import coacd
    import numpy as np
    from mathutils import Matrix, Vector

    gltf = download_gltf(key, work / key)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(gltf))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no meshes")
    # One mesh, with every transform baked in.
    for obj in meshes:
        obj.data = obj.data.copy()  # instances share data; bake each separately
        obj.data.transform(obj.matrix_world)
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
    for obj in list(scene.objects):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = key

    # In meters, standing on z = 0, centered in x and y.
    corners = np.array([v.co[:] for v in obj.data.vertices])
    scale = unit_scale(float((corners.max(0) - corners.min(0)).max()), listed)
    if scale != 1.0:
        print(f"{key}: scaling by {scale:g} to meters", flush=True)
        obj.data.transform(Matrix.Scale(scale, 4))
        corners *= scale
    lo, hi = corners.min(0), corners.max(0)
    shift = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    obj.data.transform(Matrix.Translation(shift))

    if obj.data.shape_keys:  # animation targets; a modifier can't be applied with them
        obj.shape_key_clear()
    obj.data.calc_loop_triangles()
    source_triangles = len(obj.data.loop_triangles)
    if source_triangles > TRIANGLES:
        mod = obj.modifiers.new("decimate", "DECIMATE")
        mod.ratio = TRIANGLES / source_triangles
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.data.calc_loop_triangles()
    triangles = len(obj.data.loop_triangles)

    folder = out / key
    folder.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(folder / "model.glb"), export_format="GLB", use_selection=True,
        export_image_format="AUTO", export_animations=False, export_extras=False,
    )

    vertices = np.array([v.co[:] for v in obj.data.vertices], dtype=float)
    faces = np.array([t.vertices[:] for t in obj.data.loop_triangles], dtype=np.int32)
    size = (vertices.max(0) - vertices.min(0)).tolist()
    collision = "parts"
    try:
        coacd.set_log_level("error")
        hulls = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=0.08, max_convex_hull=16, seed=0)
        parts = [np.asarray(v, dtype=np.float32) for v, _ in hulls]
    except Exception as e:  # CoACD fails on some broken meshes
        print(f"{key}: CoACD failed ({e}); using its box", flush=True)
        parts, collision = [box_corners(vertices)], "box"
    np.savez_compressed(folder / "collision.npz", **{f"part{i}": p for i, p in enumerate(parts)})

    thumbnail(obj, size, thumbs / f"{key}.png")
    return {
        "triangles": triangles,
        "source_triangles": source_triangles,
        "size": [round(s, 4) for s in size],
        "parts": len(parts),
        "collision": collision,
        "files": {name: md5(folder / name) for name in FILES},
    }


def box_corners(vertices):
    import itertools

    import numpy as np

    lo, hi = vertices.min(0), vertices.max(0)
    return np.array([[(lo, hi)[i][a] for a, i in enumerate(c)] for c in itertools.product((0, 1), repeat=3)], np.float32)


def thumbnail(obj, size, path: Path) -> None:
    """A three-quarter view on a gray floor, for checking the import."""
    import bpy
    import numpy as np

    from sim.geometry import quat_looking_at

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.85, 0.9, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world
    bpy.ops.mesh.primitive_plane_add(size=max(size) * 6)
    floor = bpy.data.materials.new("floor")
    floor.use_nodes = True
    floor.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
    bpy.context.object.data.materials.append(floor)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2
    sun.rotation_euler = (0.6, 0.2, 0.8)
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.angle = np.radians(40)
    target = np.array([0, 0, size[2] / 2])
    distance = 1.15 * np.linalg.norm(size) / (2 * np.tan(np.radians(20)))
    position = target + distance * np.array([0.6, -0.7, 0.45]) / np.linalg.norm([0.6, -0.7, 0.45])
    cam.location = tuple(position)
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = tuple(quat_looking_at(position, target))
    scene.render.filepath = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


def work_one(key: str, listed: float, work: str, out: str, thumbs: str) -> tuple[str, dict | None, str | None]:
    try:
        return key, process(key, listed, Path(work), Path(out), Path(thumbs)), None
    except Exception as e:
        return key, None, f"{type(e).__name__}: {e}"


# --- The command ---------------------------------------------------------


def contact_sheets(keys: list[str], catalog: dict, thumbs: Path, out: Path, per_sheet: int = 48) -> list[Path]:
    """Thumbnails in grids with their ids and labels, for checking by eye."""
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    made = []
    cols, cell = 8, 256
    for n in range(0, len(keys), per_sheet):
        chunk = keys[n : n + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, key in enumerate(chunk):
            x, y = (i % cols) * cell, (i // cols) * (cell + 30)
            if (thumbs / f"{key}.png").exists():
                with Image.open(thumbs / f"{key}.png") as img:
                    sheet.paste(img.convert("RGB"), (x, y))
            model = catalog[key]
            draw.text((x + 4, y + cell + 2), key[:40], fill="black")
            draw.text((x + 4, y + cell + 15), f"{model['label']}  {model['triangles']} tris  {model['parts']} parts",
                      fill="gray")
        path = out / f"sheet-{n // per_sheet + 1:02d}.png"
        sheet.save(path)
        made.append(path)
    return made


def bucket_name() -> str:
    config = Path(__file__).resolve().parents[4] / "wildhat" / "src" / "wildhat" / "config.json"
    return f"{json.loads(config.read_text())['project_id']}-wildhat"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.assets.polyhaven", description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", help="just these ids")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--work", default=str(Path.home() / ".cache" / "wildhat" / "polyhaven-downloads"),
                        help="where Poly Haven's files and the thumbnails go")
    parser.add_argument("--dry-run", action="store_true", help="say what would be imported, and stop")
    parser.add_argument("--upload", action="store_true", help="copy the files to the bucket, then write the catalog")
    args = parser.parse_args(argv)

    listing = get(f"{API}/assets?t=models")
    keep, skipped = choose(listing)
    if args.only:
        unknown = set(args.only) - set(listing)
        if unknown:
            raise SystemExit(f"Not on Poly Haven: {', '.join(sorted(unknown))}")
        keep = {k: v for k, v in keep.items() if k in args.only}
    print(f"{len(listing)} models on Poly Haven; {len(keep)} fit, {len(skipped)} don't")
    if args.dry_run:
        for key, why in skipped.items():
            print(f"  skip {key}: {why}")
        for key, model in keep.items():
            print(f"  {key}: {label_for(model['category'], key)} ({model['category']})")
        return 0

    old = json.loads(CATALOG.read_text())["assets"] if CATALOG.exists() else {}
    out = default_cache() / SOURCE
    work, thumbs = Path(args.work), Path(args.work) / "thumbnails"
    catalog, todo = {}, []
    for key, model in keep.items():
        entry = old.get(key)
        folder = out / key
        if (entry and entry.get("source_hash") == model["files_hash"]
                and all((folder / n).exists() and md5(folder / n) == entry["files"][n] for n in FILES)):
            catalog[key] = {**entry, "label": label_for(model["category"], key)}
        else:
            todo.append(key)
    print(f"{len(catalog)} already imported; importing {len(todo)} with {args.workers} workers")

    started, failed = time.time(), {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(work_one, key, largest_side(keep[key]), str(work), str(out), str(thumbs))
                for key in todo]
        for n, job in enumerate(as_completed(jobs), 1):
            key, made, error = job.result()
            if error:
                failed[key] = error
                print(f"[{n}/{len(todo)}] {key}: FAILED {error}", flush=True)
                continue
            model = keep[key]
            catalog[key] = {
                "name": model["name"],
                "label": label_for(model["category"], key),
                "category": model["category"],
                "authors": sorted(model.get("authors", {})),
                "source_hash": model["files_hash"],
                **made,
            }
            print(f"[{n}/{len(todo)}] {key}: {made['source_triangles']} -> {made['triangles']} triangles, "
                  f"{made['parts']} parts ({time.time() - started:.0f}s)", flush=True)

    if args.only:  # keep everything else already in the catalog
        catalog = {**old, **catalog}
    sheets = contact_sheets(sorted(catalog), catalog, thumbs, work / "sheets")
    print(f"Contact sheets: {', '.join(map(str, sheets))}")
    if failed:
        print(f"{len(failed)} failed: {', '.join(sorted(failed))}")
    CATALOG.write_text(json.dumps({"source": SOURCE, "triangles": TRIANGLES,
                                   "assets": dict(sorted(catalog.items()))}, indent=1) + "\n")
    if args.upload:
        dest = f"gs://{bucket_name()}/assets/{SOURCE}/"
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(out), dest], check=True)
        print(f"Uploaded to {dest} and wrote {CATALOG}. Commit it.")
    else:
        print(f"Wrote {CATALOG}, but didn't upload: cloud jobs can't use it until --upload.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
