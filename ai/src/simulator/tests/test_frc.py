"""FIRST's field: the importer's naming and tag fitting (no Blender or CAD
needed), and FrcField and GamePiece with a made-up season, 2099. Run from
ai/src/simulator:

    python -m unittest discover -s tests
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim import Camera, Floor, GamePiece, Range, Scenario, Scene, assets
from sim.assets import FILES, Library, md5
from sim.assets.frc import importer
from sim.frc import FrcField
from sim.runner import run
from sim.world import build

YEAR = 2099
BALL = 0.15  # m across


def box(size):
    sx, sy, sz = size
    return np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0, sz)], np.float32)


def ball(diameter):
    # An icosahedron's corners, standing on z = 0.
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v *= diameter / 2 / np.linalg.norm(v[0])
    return (v - [0, 0, v[:, 2].min()]).astype(np.float32)


def fake_season(folder: Path) -> Library:
    """A field with a hub on each side, a carpet, and three balls."""
    shapes = {"hub_red": box((1, 1, 1.8)), "hub_blue": box((1, 1, 1.8)), "carpet": box((16, 8, 0.004)),
              "ball": ball(BALL)}
    models = {}
    for name, part in shapes.items():
        where = folder / "frc" / str(YEAR) / name
        where.mkdir(parents=True)
        (where / "model.glb").write_bytes(b"not a real model")
        np.savez_compressed(where / "collision.npz", part0=part)
        size = (part.max(0) - part.min(0)).tolist()
        element = name.split("_")[0]
        models[name] = {"label": "hub" if element == "hub" else ("ball" if name == "ball" else None),
                        "category": f"frc/{YEAR}/element", "element": element, "size": size,
                        "files": {f: md5(where / f) for f in FILES}}
    models["ball"]["physics"] = {"mass": 0.2, "bounciness": 0.5, "friction": 0.8}
    # The importer finds keypoints from the CAD, in each model's coordinates.
    models["hub_red"]["keypoints"] = models["hub_blue"]["keypoints"] = {"top": [[0.5, 0, 1.8], [-0.5, 0, 1.8]]}
    catalog = {
        "source": f"frc/{YEAR}", "year": YEAR,
        "field": {"length": 16.0, "width": 8.0, "wpilib": {"yaw": 180.0, "offset": [8.0, 4.0], "error": 0, "tags": 3}},
        "assets": models,
        "layout": [{"asset": "carpet", "position": [0, 0, 0]}, {"asset": "hub_red", "position": [-4, 0, 0]},
                   {"asset": "hub_blue", "position": [4, 0, 0]}],
        # The middle ball starts on something (on a hub, say), so it's held.
        "pieces": {"ball": {"asset": "ball", "start": [[0, -1, 0], [0, 0, 0], [0, 1, 0]],
                            "held": [False, True, False]}},
    }
    return Library(catalog, folder, source=f"frc/{YEAR}")


class Arena(Scene):
    def setup(self, rng):
        self.floor = self.add(Floor(size=(30, 20)))


class FieldTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        assets.frc_libraries[f"frc/{YEAR}"] = fake_season(Path(self.tmp.name))

    def tearDown(self):
        assets.frc_libraries.pop(f"frc/{YEAR}", None)
        self.tmp.cleanup()


def names(world):
    return {item.name: item for item in world.items}


class TestFrcField(FieldTestCase):
    def test_elements_and_pieces_are_named_and_labeled(self):
        class Match(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR))

        items = names(build(Match, 1))
        self.assertEqual(sorted(n for n in items if n.startswith("field.")),
                         ["field.ball.0", "field.ball.1", "field.ball.2", "field.carpet.0", "field.hub.0",
                          "field.hub.1"])
        self.assertEqual(items["field.hub.0"].label, "hub")
        self.assertIsNone(items["field.carpet.0"].label)
        self.assertIsInstance(items["field.ball.1"], GamePiece)
        self.assertEqual(items["field.ball.1"].mass, 0.2)  # from the catalog

    def test_the_field_moves_with_its_position_and_rotation(self):
        class Turned(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR, position=(1, 2, 0), rotation=(0, 0, 90)))

        items = names(build(Turned, 1))
        hub_red = next(i for i in items.values() if getattr(i, "asset", None) == f"frc/{YEAR}/hub_red")
        np.testing.assert_allclose(hub_red.position, (1, -2, 0), atol=1e-9)

    def test_choosing_elements_and_no_pieces(self):
        class Hubs(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR, elements=("hub",), starting_pieces=False))

        self.assertEqual(sorted(n for n in names(build(Hubs, 1)) if n.startswith("field.")),
                         ["field.hub.0", "field.hub.1"])

        class Typo(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR, elements=("hubb",)))

        with self.assertRaisesRegex(ValueError, "No hubb on the 2099 field. Its elements: carpet, hub"):
            build(Typo, 1)

    def test_an_unknown_year(self):
        class Future(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=2199))

        with self.assertRaisesRegex(KeyError, "No FRC models for 2199"):
            build(Future, 1)

    def test_to_wpilib(self):
        class Turned(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR, position=(1, 2, 0), rotation=(0, 0, 90)))

        field = build(Turned, 1).scenario.field
        # The field's center is WPILib's (8, 4); the CAD's +x is WPILib's -x.
        np.testing.assert_allclose(field.to_wpilib((1, 2, 0))[:3], (8, 4, 0), atol=1e-9)
        # 1 m along the CAD's +x, which the turn points along the world's +y.
        x, y, z, heading = field.to_wpilib((1, 3, 0.5), yaw=90)
        np.testing.assert_allclose((x, y, z), (7, 4, 0.5), atol=1e-9)
        self.assertAlmostEqual(abs(heading), 180)  # facing the CAD's +x: WPILib's -x

    def test_balls_fall_and_rest_on_the_floor(self):
        class Drop(Scenario[Arena]):
            duration = 1.0

            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR, elements=("hub",), position=(0, 0, 1)))
                self.cam = self.add(Camera(position=(0, -6, 2), look_at=(0, 0, 0)))

        world = run(Drop, 1, "physics").world
        for ball in world.scenario.field.ball:
            if not ball.static:  # the held one stays where it started
                self.assertAlmostEqual(ball.position[2], 0, delta=0.02)  # its base on the floor
        self.assertAlmostEqual(world.scenario.field.hub[0].position[2], 1)  # static

    def test_a_game_piece_needs_its_physics(self):
        library = assets.frc_libraries[f"frc/{YEAR}"]
        del library.models["ball"]["physics"]
        with self.assertRaisesRegex(ValueError, "doesn't say its mass, bounciness, friction"):
            GamePiece(asset=f"frc/{YEAR}/ball")
        GamePiece(asset=f"frc/{YEAR}/ball", mass=0.1, bounciness=0.2, friction=0.5)

    def test_held_pieces_stay_put(self):
        class WithField(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR))

        world = build(WithField, 0)
        self.assertEqual([p.static for p in world.scenario.field.ball], [False, True, False])

    def test_keypoints_come_from_the_catalog(self):
        from sim import labels

        class WithField(Scenario[Arena]):
            def setup(self, rng):
                self.field = self.add(FrcField(year=YEAR))

        world = build(WithField, 0)
        hub = names(world)["field.hub.0"]
        np.testing.assert_allclose(labels.keypoints_of(hub)["top"], [[0.5, 0, 1.8], [-0.5, 0, 1.8]])
        # The carpet has none.
        self.assertEqual(labels.keypoints_of(names(world)["field.carpet.0"]), {})

    def test_asset_ids(self):
        self.assertEqual(assets.info(f"frc/{YEAR}/carpet")["element"], "carpet")
        with self.assertRaisesRegex(KeyError, "frc/<year>/<name>"):
            assets.info("frc/hub")


class TestImporter(unittest.TestCase):
    CONFIG = {
        "red": "-x",
        "pieces": {"fuel": {"part": "GE-26900"}},
        "elements": {
            "GE-26300": {"name": "hub", "label": "hub"},
            "GE-26100": {"name": "bump"},
            "FE-2026-01": {"name": "carpet", "alliance": False, "merge": True},
            "Gaffer Tape": {"name": "carpet", "alliance": False, "merge": True},
        },
    }

    def test_names(self):
        self.assertEqual(importer.base_name("GE-26300: Hub <2>.001"), "GE-26300: Hub")
        self.assertEqual(importer.derived_name("FE-2026-02: Welded Perimeter Assy"), "welded_perimeter_assy")
        self.assertEqual(importer.alliance_of(-4.0, "-x"), "red")
        self.assertEqual(importer.alliance_of(-4.0, "+x"), "blue")
        self.assertIsNone(importer.alliance_of(0.2, "-x"))

    def test_plan(self):
        nodes = [
            ("GE-26300: Hub <1>", (-4.0, 0.0)), ("GE-26300: Hub <2>", (4.0, 0.0)),
            ("GE-26100: Bump <1>", (-4.0, 2.0)), ("GE-26100: Bump <2>", (-4.0, -2.0)),
            ("GE-26100: Bump <3>", (4.0, 2.0)), ("GE-26100: Bump <4>", (4.0, -2.0)),
            ("FE-2026-01: Playing Field Carpet <1>", (0.0, 0.0)), ("Gaffer Tape, Red, 2in Wide", (-4.0, 0.0)),
            ("GE-26900: Fuel", (0.0, 0.0)), ("GE-26900: Fuel.001", (0.1, 0.0)),
            ("Mystery Thing", (0.0, 3.0)),
        ]
        models, pieces, warnings = importer.plan_assets(nodes, self.CONFIG)
        self.assertEqual(sorted(models), ["bump_blue_0", "bump_blue_1", "bump_red_0", "bump_red_1", "carpet",
                                          "hub_blue", "hub_red", "mystery_thing"])
        self.assertEqual(models["hub_red"]["nodes"], [0])
        self.assertEqual(models["bump_red_0"]["nodes"], [3])  # y = -2 comes first
        self.assertEqual(models["carpet"]["nodes"], [6, 7])  # tape and carpet together
        self.assertEqual(models["hub_red"]["label"], "hub")
        self.assertEqual(pieces, {"fuel": [8, 9]})
        self.assertEqual(len(warnings), 1)
        self.assertIn("Mystery Thing", warnings[0])

    def test_fit_wpilib(self):
        # WPILib's frame: the CAD's turned 180 degrees, its center at (8.27, 4.03).
        cad = {1: (1.0, 2.0, 1.0), 2: (-3.0, 0.5, 1.0), 3: (5.0, -2.0, 0.5), 4: (0.0, 0.0, 1.2)}
        wpilib = {i: (8.27 - x, 4.03 - y) for i, (x, y, _) in cad.items()}
        fit = importer.fit_wpilib(cad, wpilib)
        self.assertAlmostEqual(abs(fit["yaw"]), 180, places=3)
        np.testing.assert_allclose(fit["offset"], (8.27, 4.03), atol=1e-4)
        self.assertLess(fit["error"], 1e-4)
        self.assertEqual(fit["tags"], 4)

    def test_tags_by_where_they_are(self):
        # WPILib's frame is the CAD's moved by (8, 4). The CAD has tag 1 twice
        # (its vinyl repeated, 1 cm apart), and calls the tag where WPILib's
        # 3 is "2", so 2 appears twice.
        wpilib = {1: (9.0, 6.0), 2: (5.0, 4.5), 3: (13.0, 2.0), 4: (8.0, 4.0), 5: (10.0, 1.0)}
        found = [("vinyl 1", 1, (1.0, 2.0, 1.0)), ("vinyl 1.001", 1, (1.0, 2.01, 1.0)),
                 ("vinyl 2", 2, (-3.0, 0.5, 1.0)), ("vinyl 2.001", 2, (5.0, -2.0, 1.0)),
                 ("vinyl 4", 4, (0.0, 0.0, 1.0)), ("vinyl 5", 5, (2.0, -3.0, 1.0))]
        fit, ids, warnings = importer.assign_tags(found, wpilib)
        self.assertEqual(ids, {"vinyl 1": 1, "vinyl 1.001": None, "vinyl 2": 2, "vinyl 2.001": 3, "vinyl 4": 4,
                               "vinyl 5": 5})
        self.assertEqual(len(warnings), 1)
        self.assertIn("WPILib's 3 is there", warnings[0])
        np.testing.assert_allclose(fit["offset"], (8, 4), atol=1e-6)
        self.assertEqual(fit["tags"], 5)

    def test_pipe_ends(self):
        # A pipe 0.3 m long from (1, 2, 0) up along a slant: its end faces'
        # middles, whichever way round.
        axis = np.array([0.0, 0.6, 0.8])
        ring = [np.array([np.cos(a), 0, 0]) * 0.05 + np.array([0, -0.8, 0.6]) * np.sin(a) * 0.05
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        start = np.array([1.0, 2.0, 0.0])
        verts = np.array([start + r + axis * t for t in (0.0, 0.3) for r in ring])
        ends = sorted(importer.pipe_ends(verts), key=lambda p: p[2])
        np.testing.assert_allclose(ends[0], start, atol=1e-9)
        np.testing.assert_allclose(ends[1], start + axis * 0.3, atol=1e-9)

    def test_a_zipped_step(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with zipfile.ZipFile(folder / "field.zip", "w") as z:
                z.writestr("inside/FE-2099.STEP", "ISO-10303-21;")
            step = importer.unzipped(folder / "field.zip")
            self.assertEqual(step, folder / "FE-2099.STEP")
            self.assertEqual(step.read_text(), "ISO-10303-21;")
            self.assertEqual(importer.unzipped(step), step)  # not a zip: itself


if __name__ == "__main__":
    unittest.main()
