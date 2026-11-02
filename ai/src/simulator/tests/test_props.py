"""Props: the asset catalog, fetching models, and props in physics, logs and
pictures. The model here is made up, two posts 0.8 m apart, so a ball
dropped between them falls to the floor (one convex hull would catch it)
and one dropped on a post lands on top. Rendering is skipped without bpy."""

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np

from sim import Camera, Floor, Light, Prop, Range, Scenario, Scene, Sphere, assets, runner
from sim.assets import FILES, Library, MissingAsset, md5
from sim.assets import polyhaven
from sim.logger import SimLogger
from sim.world import build

try:
    import bpy
except ImportError:
    bpy = None

POST, GAP, HEIGHT = 0.1, 0.8, 0.5


def box(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)


def posts() -> list[np.ndarray]:
    """Two posts, centered on x = ±GAP/2, standing on z = 0."""
    return [box((x - POST / 2, -POST / 2, 0), (x + POST / 2, POST / 2, HEIGHT)) for x in (-GAP / 2, GAP / 2)]


def make_glb(path: Path) -> None:
    """The posts as a model, gray, made in Blender."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("posts")
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    verts = np.concatenate(posts()).tolist()
    mesh.from_pydata(verts, [], [tuple(i + 8 * n for i in f) for n in range(2) for f in faces])
    material = bpy.data.materials.new("paint")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
    mesh.materials.append(material)
    obj = bpy.data.objects.new("posts", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True)


def library(folder: Path, with_files=True, glb=False) -> Library:
    """A catalog of one model, "posts", with its files in folder/polyhaven/posts
    (or without them)."""
    model = folder / "polyhaven" / "posts"
    model.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model / "collision.npz", **{f"part{i}": p for i, p in enumerate(posts())})
    if glb:
        make_glb(model / "model.glb")
    else:
        (model / "model.glb").write_bytes(b"not used without bpy")
    entry = {"label": "post", "category": "Test/Posts", "size": [GAP + POST, POST, HEIGHT],
             "files": {name: md5(model / name) for name in FILES}}
    if not with_files:
        shutil.rmtree(model)
    return Library({"assets": {"posts": entry}}, folder)


class Room(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(10, 10)))
        self.sun = self.add(Light(kind="sun", intensity=4, direction=(0.2, 0.4, -1)))


def drop_at(x):
    class Drop(Scenario[Room]):
        duration, fps = 1.5, 10

        def setup(self, rng):
            self.posts = self.add(Prop(asset="posts", label="posts"))
            self.ball = self.add(Sphere(radius=0.05, position=(x, 0, 1.0), bounciness=0))
            self.cam = self.add(Camera(position=(0, -3, 0.6), look_at=(0, 0, 0.3), resolution=(160, 120)))

    return Drop


def landed(scenario) -> float:
    return runner.run(scenario, 0, "physics").world.by_name("ball").position[2]


class Fixture(unittest.TestCase):
    with_files = True

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.lib = library(self.folder, self.with_files)
        patch = mock.patch.object(assets, "library", self.lib)
        patch.start()
        self.addCleanup(patch.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("WILDHAT_OUTPUT", None)


class TestCatalog(unittest.TestCase):
    lib = Library({"assets": {
        "chair_b": {"label": "chair", "category": "Furniture/Seating/Chairs", "size": [0.6, 0.6, 0.9]},
        "chair_a": {"label": "chair", "category": "Furniture/Seating/Chairs", "size": [0.5, 0.5, 1.0]},
        "couch": {"label": "couch", "category": "Furniture/Seating/Sofas & Couches", "size": [2.2, 0.9, 0.8]},
        "fern": {"label": "plant", "category": "Nature/Plants/Ferns", "size": [0.4, 0.4, 0.5]},
    }}, Path("/nowhere"))

    def test_find(self):
        self.assertEqual(self.lib.find("chair"), ["chair_a", "chair_b"])
        self.assertEqual(self.lib.find(category="furniture/seating"), ["chair_a", "chair_b", "couch"])
        self.assertEqual(self.lib.find(category="Furniture", max_size=1.0), ["chair_a", "chair_b"])
        self.assertEqual(self.lib.labels(), {"chair": 2, "couch": 1, "plant": 1})

    def test_unknown_model(self):
        with self.assertRaisesRegex(KeyError, "No model 'desk'"):
            self.lib.info("desk")

    def test_the_real_catalog_reads(self):
        lib = Library.load()
        for key, model in lib.models.items():
            self.assertEqual(set(model["files"]), set(FILES), key)
            self.assertEqual(len(model["size"]), 3, key)


class TestFetch(Fixture):
    def test_files_on_this_computer_are_used(self):
        self.assertEqual(self.lib.fetch("posts"), self.folder / "polyhaven" / "posts")
        self.assertEqual(len(self.lib.collision("posts")), 2)

    def test_changed_files_count_as_missing(self):
        (self.folder / "polyhaven" / "posts" / "collision.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(MissingAsset, "only downloaded in cloud jobs"):
            self.lib.fetch("posts")

    def test_cloud_jobs_download_from_the_bucket(self):
        good = {name: (self.folder / "polyhaven" / "posts" / name).read_bytes() for name in FILES}
        cache = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, cache)
        lib = Library({"assets": self.lib.models}, cache)
        fetched = []

        def download(bucket, name, path, tok):
            fetched.append((bucket, name))
            path.write_bytes(good[Path(name).name])

        os.environ["WILDHAT_OUTPUT"] = "gs://the-bucket/runs/kai/run1/"
        with mock.patch.object(assets, "token", lambda: "t"), mock.patch.object(assets, "download", download):
            self.assertEqual(lib.fetch("posts"), cache / "polyhaven" / "posts")
            lib.fetch("posts")  # once only
        self.assertEqual(fetched, [("the-bucket", f"assets/polyhaven/posts/{n}") for n in FILES])

    def test_a_bad_download_is_an_error(self):
        cache = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, cache)
        lib = Library({"assets": self.lib.models}, cache)
        os.environ["WILDHAT_OUTPUT"] = "gs://the-bucket/runs/kai/run1/"
        with mock.patch.object(assets, "token", lambda: "t"), \
                mock.patch.object(assets, "download", lambda b, n, p, t: p.write_bytes(b"wrong")):
            with self.assertRaisesRegex(RuntimeError, "doesn't match the catalog"):
                lib.fetch("posts")


class TestProps(Fixture):
    def test_mistakes_are_caught(self):
        class Unknown(Scenario[Room]):
            def setup(self, rng):
                self.add(Prop(asset="sofa_99"))

        class Moving(Scenario[Room]):
            def setup(self, rng):
                self.add(Prop(asset="posts", static=False))

        with self.assertRaisesRegex(ValueError, "No model 'sofa_99'"):
            build(Unknown, 0)
        with self.assertRaisesRegex(ValueError, "GamePiece is the kind that moves"):
            build(Moving, 0)

    def test_a_ball_lands_on_a_post_and_falls_between_them(self):
        self.assertAlmostEqual(landed(drop_at(GAP / 2)), HEIGHT + 0.05, delta=0.01)
        self.assertAlmostEqual(landed(drop_at(0)), 0.05, delta=0.01)

    def test_scale_and_rotation_move_the_posts(self):
        class Turned(Scenario[Room]):
            duration, fps = 1.5, 10

            def setup(self, rng):
                self.posts = self.add(Prop(asset="posts", scale=2, rotation=(0, 0, 90), position=(1, 0, 0)))
                # Turned 90°, the posts are at y = ±0.8 (scaled), 1 m along x.
                self.ball = self.add(Sphere(radius=0.05, position=(1, GAP, 2), bounciness=0))

        self.assertAlmostEqual(landed(Turned), 2 * HEIGHT + 0.05, delta=0.01)

    def test_settings_are_logged_without_the_unused_ones(self):
        world = build(drop_at(0), 0)
        logger = SimLogger(self.folder / "ep", world, {}, render=False)
        rows = logger.log.rows["objects"]
        posts = next(r for r in rows if r["name"] == "posts")
        self.assertEqual(posts["type"], "Prop")
        self.assertEqual(posts["label"], "posts")
        settings = json.loads(posts["settings"])
        self.assertEqual(settings["asset"], "posts")
        self.assertNotIn("color", settings)

    def test_random_models(self):
        class Any(Scenario[Room]):
            def setup(self, rng):
                self.thing = self.add(Prop(asset=Range.choice(assets.find("post"))))

        self.assertEqual(build(Any, 3).by_name("thing").asset, "posts")


class TestWithoutFiles(Fixture):
    with_files = False

    def test_a_laptop_uses_the_catalog_box(self):
        err = io.StringIO()
        with redirect_stderr(err):
            # The box spans both posts, so the ball lands on its top.
            self.assertAlmostEqual(landed(drop_at(0)), HEIGHT + 0.05, delta=0.01)
        self.assertIn("each is its box from the catalog", err.getvalue())

    def test_a_cloud_job_needs_the_files(self):
        os.environ["WILDHAT_OUTPUT"] = "gs://the-bucket/runs/kai/run1/"
        with mock.patch.object(assets, "token", lambda: "t"), \
                mock.patch.object(assets, "download", mock.Mock(side_effect=OSError("404"))):
            with self.assertRaises(OSError):
                landed(drop_at(0))


class TestImporter(unittest.TestCase):
    def test_choose(self):
        listing = {
            "chair": {"category": "Furniture/Seating/Chairs", "dimensions": [600, 600, 900], "categories": []},
            "cliff": {"category": "Nature/Rocks & Stone/Cliffs & Rock Faces", "dimensions": [900, 900, 900],
                      "categories": []},
            "doll": {"category": "Leisure/Games & Toys/Toys", "dimensions": [300, 300, 500], "categories": ["rigged"]},
            "tree": {"category": "Nature/Trees/Broadleaf Trees", "dimensions": [4000, 4000, 7000], "categories": []},
            "cabinet": {"category": "Furniture/Storage Furniture/Sideboards", "dimensions": [1000, 400, 2300],
                        "categories": []},
            "ring": {"category": "Apparel & Personal Items/Accessories", "dimensions": [20, 20, 5], "categories": []},
        }
        keep, skipped = polyhaven.choose(listing)
        self.assertEqual(list(keep), ["chair"])
        self.assertEqual(skipped["doll"], "rigged")
        self.assertIn("category", skipped["cliff"])
        self.assertEqual(skipped["tree"], "4.00 m across")
        self.assertEqual(skipped["cabinet"], "2.30 m tall")
        self.assertEqual(skipped["ring"], "0.02 m across")

    def test_unit_scale(self):
        self.assertEqual(polyhaven.unit_scale(1.05, 1.0), 1.0)
        self.assertEqual(polyhaven.unit_scale(0.5, 0), 1.0)  # nothing listed
        self.assertAlmostEqual(polyhaven.unit_scale(21.4, 2.14), 0.1)  # in decimeters
        self.assertAlmostEqual(polyhaven.unit_scale(0.0214, 2.14), 100)  # 100 times too small
        with self.assertRaisesRegex(RuntimeError, "measures 3.00 m across, but Poly Haven lists 1.00 m"):
            polyhaven.unit_scale(3.0, 1.0)

    def test_labels(self):
        self.assertEqual(polyhaven.label_for("Furniture/Tables/Coffee Tables"), "table")
        self.assertEqual(polyhaven.label_for("Leisure/Sports/Balls"), "sports_ball")
        self.assertEqual(polyhaven.label_for("Tools & Equipment/Hand Tools/Wrenches & Spanners"), "prop")


class Shown(Scenario[Room]):
    def setup(self, rng):
        self.plain = self.add(Prop(asset="posts", position=(-0.7, 0, 0), label="posts"))
        self.red = self.add(Prop(asset="posts", position=(0.7, 0, 0), tint=(1, 0.1, 0.1), label="posts"))
        self.cam = self.add(Camera(position=(0, -3, 0.8), look_at=(0, 0, 0.25), resolution=(240, 160), fov=60))


@unittest.skipIf(bpy is None, "bpy isn't installed")
class TestRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.backends.blender import Blender
        from sim.runner import camera_pose

        folder = Path(tempfile.mkdtemp())
        try:
            lib = library(folder, glb=True)
            with mock.patch.object(assets, "library", lib):
                cls.world = build(Shown, 0)
                backend = Blender(samples=4)
                backend.build(cls.world)
                cam = cls.world.cameras[0]
                cls.picture = backend.render(cam, *camera_pose(cam))
                plain, red = (backend.objects[cls.world.by_name(n).id] for n in ("plain", "red"))
                cls.shared_mesh = plain.data == red.data
                backend.close()
        finally:
            shutil.rmtree(folder)

    def test_each_prop_has_its_own_id_in_the_mask(self):
        ids = set(np.unique(self.picture.ids).tolist())
        self.assertTrue({self.world.by_name("plain").id, self.world.by_name("red").id} <= ids)

    def test_props_of_one_model_share_its_mesh(self):
        self.assertTrue(self.shared_mesh)

    def test_tint(self):
        def mean(name):
            return self.picture.rgb[self.picture.ids == self.world.by_name(name).id].mean(0)

        plain, red = mean("plain"), mean("red")
        self.assertLess(abs(plain[0] - plain[1]), 15)  # gray
        self.assertGreater(red[0], 2 * red[1])  # red


if __name__ == "__main__":
    unittest.main()
