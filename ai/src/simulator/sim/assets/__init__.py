"""Models to put in a Scene, from the asset catalogs, for `Prop`.

    from sim import assets
    assets.find("chair")                  # every chair's id, sorted
    assets.find(category="furniture")     # by Poly Haven's category path
    assets.info("WoodenChair_01")         # its label, size in meters, ...
    assets.info("frc/2026/hub_red")       # FIRST's field models, by year

Ids without a slash are Poly Haven's (polyhaven.json). `frc/<year>/<name>`
are from FIRST's official field CAD for that season (frc/<year>.json, made
by `python -m sim.assets.frc`; `sim.frc.FrcField` places them).

The catalog (polyhaven.json, next to this file) lists each model's label,
category and size, so choosing models needs no network. The models
themselves are in the bucket, under assets/polyhaven/<id>/ (and
assets/frc/<year>/<name>/):

    model.glb        what cameras see: the mesh and its textures
    collision.npz    what physics sees: convex parts (part0, part1, ...),
                     each an array of vertices, in meters

`python -m sim.assets.polyhaven` (admins only) makes them. In a cloud job,
an episode downloads only the models it uses. On a laptop they're read from
the cache folder ($WILDHAT_ASSETS, or ~/.cache/wildhat/assets), which the
importer fills; without them, `python -m sim check` stands in each Prop's
box from the catalog for its collision parts.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

CATALOG = Path(__file__).with_name("polyhaven.json")
FRC = Path(__file__).with_name("frc")
SOURCE = "polyhaven"
FILES = ("model.glb", "collision.npz")
METADATA_TOKEN = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
STORAGE = "https://storage.googleapis.com"


class MissingAsset(RuntimeError):
    """A model's files aren't on this computer, and it isn't a cloud job."""


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache() -> Path:
    return Path(os.environ.get("WILDHAT_ASSETS") or Path.home() / ".cache" / "wildhat" / "assets")


def cloud_bucket() -> Optional[str]:
    """The bucket, when running as a cloud job (WILDHAT_OUTPUT is
    gs://<bucket>/runs/...)."""
    out = os.environ.get("WILDHAT_OUTPUT", "")
    return urllib.parse.urlsplit(out).netloc if out.startswith("gs://") else None


class Library:
    """A catalog of models, and where their files are."""

    def __init__(self, catalog: dict, cache: Path, source: str = SOURCE):
        self.catalog = catalog
        self.models: dict[str, dict] = catalog["assets"]
        self.cache, self.source = Path(cache), source
        self.checked: set[str] = set()

    @classmethod
    def load(cls, path: Path = CATALOG, cache: Optional[Path] = None, source: str = SOURCE) -> "Library":
        catalog = json.loads(path.read_text()) if path.exists() else {"assets": {}}
        return cls(catalog, cache or default_cache(), source)

    def find(self, label: Optional[str] = None, category: Optional[str] = None,
             max_size: Optional[float] = None) -> list[str]:
        """Ids of the models with this label, under this category (a prefix
        of the path, any case: "furniture/seating"), and whose longest side
        is at most max_size meters. Sorted, so a Range.choice of them draws
        the same model for the same seed."""
        out = []
        for key, model in self.models.items():
            if label is not None and model["label"] != label:
                continue
            if category is not None and not model["category"].lower().startswith(category.lower().strip("/")):
                continue
            if max_size is not None and max(model["size"]) > max_size:
                continue
            out.append(key)
        return sorted(out)

    def labels(self) -> dict[str, int]:
        """Each label, and how many models have it."""
        counts: dict[str, int] = {}
        for model in self.models.values():
            counts[model["label"]] = counts.get(model["label"], 0) + 1
        return dict(sorted(counts.items()))

    def info(self, key: str) -> dict:
        if key not in self.models:
            raise KeyError(f"No model {key!r} in the asset catalog. assets.find() lists them.")
        return self.models[key]

    def folder(self, key: str) -> Path:
        return self.cache / self.source / key

    def fetch(self, key: str) -> Path:
        """The folder holding this model's files, downloading them first in a
        cloud job. Raises MissingAsset on a laptop that doesn't have them."""
        folder = self.folder(key)
        if key in self.checked:
            return folder
        want = self.info(key)["files"]
        missing = [name for name in FILES if not (folder / name).exists() or md5(folder / name) != want[name]]
        if missing:
            bucket = cloud_bucket()
            if bucket is None:
                raise MissingAsset(
                    f"{key}'s files aren't in {folder}. Props are only downloaded in cloud jobs for now."
                )
            folder.mkdir(parents=True, exist_ok=True)
            tok = token()
            for name in missing:
                download(bucket, f"assets/{self.source}/{key}/{name}", folder / name, tok)
                if md5(folder / name) != want[name]:
                    raise RuntimeError(f"{key}/{name} in the bucket doesn't match the catalog. Re-run the importer.")
        self.checked.add(key)
        return folder

    def collision(self, key: str) -> list[np.ndarray]:
        """The model's convex parts, each an (n, 3) array of vertices."""
        with np.load(self.fetch(key) / "collision.npz") as parts:
            return [parts[f"part{i}"] for i in range(len(parts.files))]


def token() -> str:
    """An access token for the job's service account."""
    req = urllib.request.Request(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["access_token"]


def download(bucket: str, name: str, path: Path, tok: str) -> None:
    url = f"{STORAGE}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, partial.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    os.replace(partial, path)


# The libraries Scenes and the backends use. Tests swap in their own.
library = Library.load()
# FIRST's, by source ("frc/2026"), loaded when first used.
frc_libraries: dict[str, Library] = {}


def frc_library(year: int) -> Library:
    """FIRST's field models for a season. Its `catalog` also has the
    field's layout, game pieces and size."""
    source = f"frc/{year}"
    if source not in frc_libraries:
        path = FRC / f"{year}.json"
        if not path.exists():
            have = sorted(p.stem for p in FRC.glob("*.json"))
            raise KeyError(f"No FRC models for {year}. There are for: {', '.join(have) or 'no years yet'}. "
                           "An admin imports a season with `python -m sim.assets.frc <year>`.")
        frc_libraries[source] = Library.load(path, source=source)
    return frc_libraries[source]


def library_for(key: str) -> tuple[Library, str]:
    """The library a model id is in, and its id there."""
    if isinstance(key, str) and key.startswith("frc/"):
        parts = key.split("/")
        if len(parts) != 3 or not parts[1].isdigit():
            raise KeyError(f"{key!r}: FRC models are frc/<year>/<name>, e.g. frc/2026/hub_red")
        return frc_library(int(parts[1])), parts[2]
    return library, key


def find(label: Optional[str] = None, category: Optional[str] = None, max_size: Optional[float] = None) -> list[str]:
    return library.find(label, category, max_size)


def info(key: str) -> dict:
    lib, name = library_for(key)
    return lib.info(name)


def fetch(key: str) -> Path:
    lib, name = library_for(key)
    return lib.fetch(name)


def collision(key: str) -> list[np.ndarray]:
    lib, name = library_for(key)
    return lib.collision(name)


def labels() -> dict[str, int]:
    return library.labels()
