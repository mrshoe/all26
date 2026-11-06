"""Scoring a model: how good are its predictions, in numbers that mean the
same for everyone. Compares a `shared.predictions` folder with the ground
truth of the dataset's pictures it was run on.

    python -m shared.score <predictions folder> [--split validation] [--min-pixels 20]

prints a table and writes score.json next to the predictions. From Python:

    from shared.score import score, metrics
    result = score("predictions", min_pixels=20)
    run.final(**metrics(result, "val"))    # val_box_ap50=..., val_box_ap=...

**Boxes.** In each picture and for each class, the predictions are taken
highest score first; each claims the unclaimed true box it overlaps most,
if their IoU (intersection over union) is at least the threshold. One that
claims a box is a hit, one that claims nothing a false alarm, and a true box
nobody claims a miss. Sweeping the score threshold traces precision against
recall; average precision (AP) is the area under that curve, as COCO
computes it (101 recall points). AP50 uses IoU 0.5; AP averages IoU 0.5 to
0.95, so it also rewards tight boxes. Objects showing fewer than
`min_pixels` pixels, or less than `min_share` of themselves, are ignored:
missing one isn't a miss, and finding one isn't a false alarm. (Photos
don't count pixels: for them, min_pixels compares with the box's area.)

**Keypoints.** The same, with distance in place of overlap: a predicted
point is a hit if it's within r pixels of an unclaimed true point of its
class, for r = 2, 4 and 8. Reported: the share of true points found, AP, and
false points per picture. By default only points that are seen count (a
model isn't expected to find a corner behind the REEF); `--all-points`
counts hidden ones too. Either way, the breakdown says how many of each kind
were found: seen, hidden by the object itself, hidden by something else.

**Breakdowns** matter more than the headline: boxes by size (small, under
32 x 32 pixels; medium, under 96 x 96; large), keypoints by visibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from shared import predictions as preds_io
from shared.datasets import SPLITS, Dataset, box_pixels

IOUS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.5, 0.55, ... 0.95
RADII = (2, 4, 8)
RECALLS = np.linspace(0, 1, 101)
# COCO's object sizes, by box area in pixels.
SIZES = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, float("inf"))}
SEEN, SELF, OTHER = "seen", "hidden by itself", "hidden by something else"


# --- Matching ---------------------------------------------------------------


def iou(a, b) -> np.ndarray:
    """IoU of box a (x0, y0, x1, y1) with each of boxes b (n x 4)."""
    b = np.asarray(b, float).reshape(-1, 4)
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match(scores, closeness, ignored, good) -> list[Optional[bool]]:
    """Greedy matching in one picture, one class. `closeness` is each
    prediction (rows) against each true object (columns), higher closer;
    `good(c)` says whether it's close enough. Predictions go highest score
    first; each claims the closest unclaimed true object it's good with,
    preferring ones that aren't ignored. Returns, per prediction, True (a
    hit), False (a false alarm) or None (it matched an ignored object, so it
    doesn't count)."""
    out: list[Optional[bool]] = [False] * len(scores)
    claimed = np.zeros(closeness.shape[1], bool)
    for p in np.argsort(-np.asarray(scores, float), kind="stable"):
        row = closeness[p]
        choices = [t for t in np.argsort(-row, kind="stable") if not claimed[t] and good(row[t])]
        if not choices:
            continue
        kept = [t for t in choices if not ignored[t]]
        t = kept[0] if kept else choices[0]
        claimed[t] = True
        out[p] = None if ignored[t] else True
    return out


def average_precision(results: list[tuple[float, Optional[bool]]], truths: int) -> Optional[float]:
    """COCO's AP from every (score, hit) over the dataset, and how many true
    objects there are: precision made non-increasing, read at 101 recall
    levels, averaged. None when there's nothing to find."""
    if truths == 0:
        return None
    counted = sorted([(s, h) for s, h in results if h is not None], key=lambda r: -r[0])
    if not counted:
        return 0.0
    hits = np.cumsum([h for _, h in counted])
    precision = hits / np.arange(1, len(counted) + 1)
    recall = hits / truths
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    at = np.searchsorted(recall, RECALLS, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in at]))


def best_f1(results: list[tuple[float, Optional[bool]]], truths: int) -> Optional[dict]:
    """The score threshold with the best F1, and its precision and recall:
    a starting point for the threshold a robot uses."""
    counted = sorted([(s, h) for s, h in results if h is not None], key=lambda r: -r[0])
    if truths == 0 or not counted:
        return None
    hits = np.cumsum([h for _, h in counted])
    precision = hits / np.arange(1, len(counted) + 1)
    recall = hits / truths
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    i = int(np.argmax(f1))
    return {"threshold": round(float(counted[i][0]), 4), "f1": round(float(f1[i]), 4),
            "precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4)}


# --- Scoring ----------------------------------------------------------------


def score(folder, dataset=None, split: Optional[str] = None, min_pixels: int = 0, min_share: float = 0.0,
          boxes: str = "visible", all_points: bool = False, classes: Optional[list[str]] = None) -> dict:
    """Score a predictions folder against the truth. `dataset`, `split` and
    `classes` (what the model finds; other labels in the truth aren't
    scored) default to the ones its predictions.json names. `boxes` is
    "visible" (what's seen of each object) or "full" (all of it, hidden
    parts too), whichever the model was trained to find."""
    info, predicted = preds_io.load(folder)
    if dataset is None:
        if not info.get("dataset"):
            raise ValueError("predictions.json doesn't say which dataset: pass --dataset")
        dataset = info["dataset"]
    if not isinstance(dataset, Dataset):
        dataset = Dataset.load(dataset)
    split = split or info.get("split") or "validation"
    if split not in SPLITS:
        raise ValueError(f"There's no split {split!r}; they're {', '.join(SPLITS)}")
    if boxes not in ("visible", "full"):
        raise ValueError('boxes must be "visible" or "full"')
    looked = {tuple(p) for p in info["pictures"]}
    classes = set(classes if classes is not None else info.get("classes") or [])
    # What the model predicts: boxes, keypoints or both (a box detector
    # isn't scored on keypoints). One that found nothing at all is scored on
    # both.
    kinds = {k for k in ("boxes", "keypoints") if info.get(k)} or {"boxes", "keypoints"}
    samples = dataset.samples(split)
    if not samples:
        raise ValueError(f"The dataset's {split} split has no pictures")
    missing = [s for s in samples if (s.run, s.index, s.episode.objects[s.camera]["object"], s.frame) not in looked]
    if len(missing) == len(samples):
        raise ValueError(f"These predictions aren't on the {split} split's pictures (none of its "
                         f"{len(samples)} were predicted on). Is it the right split?")

    box_results = defaultdict(lambda: defaultdict(list))  # class -> iou -> [(score, hit)]
    box_truths = defaultdict(int)  # class -> counted true boxes
    size_results = defaultdict(lambda: defaultdict(list))  # size -> iou -> [...]
    size_truths = defaultdict(int)
    point_results = defaultdict(lambda: defaultdict(list))  # class -> radius -> [...]
    point_truths = defaultdict(int)
    kinds_found = defaultdict(lambda: defaultdict(int))  # radius -> kind -> found
    kinds_total = defaultdict(int)
    false_points = defaultdict(int)  # radius -> false points
    pictures = 0

    by_picture: dict[tuple, dict[str, list]] = defaultdict(lambda: {"boxes": [], "keypoints": []})
    for (run, index), tables in predicted.items():
        for table in ("boxes", "keypoints"):
            for r in tables[table]:
                by_picture[(run, index, r["camera"], r["frame"])][table].append(r)

    for s in samples:
        camera = s.episode.objects[s.camera]["object"]
        key = (s.run, s.index, camera, s.frame)
        if key not in looked:
            continue
        pictures += 1
        mine = by_picture.get(key, {"boxes": [], "keypoints": []})
        if "boxes" in kinds:
            _score_boxes(s, mine["boxes"], classes, min_pixels, min_share, boxes, box_results, box_truths,
                         size_results, size_truths)
        if "keypoints" in kinds:
            _score_points(s, mine["keypoints"], classes, all_points, point_results, point_truths, kinds_found,
                          kinds_total, false_points)

    result = {"model": info["model"], "dataset": str(dataset.path), "split": split, "pictures": pictures,
              "not_predicted": len(missing), "classes": sorted(classes),
              "options": {"min_pixels": min_pixels, "min_share": min_share, "boxes": boxes, "all_points": all_points}}
    if box_truths or box_results:
        result["boxes"] = _box_summary(box_results, box_truths, size_results, size_truths)
    if point_truths or point_results:
        result["keypoints"] = _point_summary(point_results, point_truths, kinds_found, kinds_total, false_points,
                                             pictures)
    return result


def _score_boxes(sample, predicted, classes, min_pixels, min_share, mode, results, truths, size_results,
                 size_truths):
    true_boxes = sample.boxes()
    labels = ({b.label for b in true_boxes} | {r["label"] for r in predicted}) & classes
    for label in labels:
        truth = [b for b in true_boxes if b.label == label]
        if mode == "full" and any(b.full is None for b in truth):
            raise ValueError(f"{sample.run} episode {sample.index} has no full boxes: it was simulated before "
                             "they were added. Score its visible boxes, or simulate it again.")
        coords = np.array([b.full if mode == "full" else (b.x0, b.y0, b.x1, b.y1) for b in truth], float).reshape(-1, 4)
        ignored = np.array([box_pixels(b) < min_pixels or (min_share > 0 and (b.share or 0) < min_share)
                            for b in truth], bool)
        mine = [r for r in predicted if r["label"] == label]
        scores = [r["score"] for r in mine]
        boxes = np.array([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in mine], float).reshape(-1, 4)
        overlaps = np.array([iou(b, coords) for b in boxes]).reshape(len(mine), len(truth))
        truths[label] += int((~ignored).sum())
        areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
        mine_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for threshold in IOUS:
            hits = match(scores, overlaps, ignored, lambda o, t=threshold: o >= t)
            results[label][threshold] += list(zip(scores, hits))
            # By size: true boxes of other sizes are ignored, and so are false
            # alarms of other sizes (COCO's way).
            for size, (lo, hi) in SIZES.items():
                other = (areas < lo) | (areas >= hi)
                sized = match(scores, overlaps, ignored | other, lambda o, t=threshold: o >= t)
                sized = [None if h is False and not lo <= mine_areas[i] < hi else h for i, h in enumerate(sized)]
                size_results[size][threshold] += list(zip(scores, sized))
                if threshold == IOUS[0]:
                    size_truths[size] += int((~(ignored | other)).sum())


def _point_kind(p) -> str:
    return SEEN if p.visible else SELF if p.occluded_by == "self" else OTHER


def _score_points(sample, predicted, classes, all_points, results, truths, kinds_found, kinds_total, false_points):
    true_points = [p for p in sample.keypoints() if p.in_frame and p.label in classes]
    labels = ({p.label for p in true_points} | {r["label"] for r in predicted}) & classes
    for label in labels:
        truth = [p for p in true_points if p.label == label]
        where = np.array([(p.x, p.y) for p in truth], float).reshape(-1, 2)
        kinds = [_point_kind(p) for p in truth]
        ignored = np.array([not all_points and k != SEEN for k in kinds], bool)
        mine = [r for r in predicted if r["label"] == label]
        scores = [r["score"] for r in mine]
        at = np.array([(r["x"], r["y"]) for r in mine], float).reshape(-1, 2)
        # Closer is better: negative distance.
        closeness = -np.linalg.norm(at[:, None, :] - where[None, :, :], axis=2) if len(mine) and len(truth) \
            else np.zeros((len(mine), len(truth)))
        truths[label] += int((~ignored).sum())
        for k in kinds:
            kinds_total[k] += 1
        for radius in RADII:
            hits = match(scores, closeness, ignored, lambda c, r=radius: -c <= r)
            results[label][radius] += list(zip(scores, hits))
            false_points[radius] += sum(1 for h in hits if h is False)
            # Which true points of each kind were found, ignoring nothing.
            for t in _claimed(scores, closeness, radius):
                kinds_found[radius][kinds[t]] += 1


def _claimed(scores, closeness, radius) -> set[int]:
    """The true points some prediction claims within `radius`."""
    claimed = np.zeros(closeness.shape[1], bool)
    for p in np.argsort(-np.asarray(scores, float), kind="stable"):
        row = closeness[p]
        for t in np.argsort(-row, kind="stable"):
            if not claimed[t] and -row[t] <= radius:
                claimed[t] = True
                break
    return set(np.nonzero(claimed)[0].tolist())


def _rounded(v):
    return None if v is None else round(v, 4)


def _box_summary(results, truths, size_results, size_truths) -> dict:
    classes = {}
    for label in sorted(set(truths) | set(results)):
        aps = [average_precision(results[label][t], truths[label]) for t in IOUS]
        found = [h for _, h in results[label][0.5] if h]
        classes[label] = {
            "ap": _rounded(None if aps[0] is None else float(np.mean(aps))),
            "ap50": _rounded(aps[0]), "ap75": _rounded(aps[IOUS.index(0.75)]),
            "truth": truths[label], "predicted": len(results[label][0.5]),
            "recall50": _rounded(len(found) / truths[label] if truths[label] else None),
            "best_f1": best_f1(results[label][0.5], truths[label]),
        }
    scored = [c for c in classes.values() if c["ap"] is not None]
    mean = lambda key: _rounded(float(np.mean([c[key] for c in scored]))) if scored else None
    by_size = {}
    for size in SIZES:
        aps = [average_precision(size_results[size][t], size_truths[size]) for t in IOUS]
        by_size[size] = {"ap": _rounded(None if aps[0] is None else float(np.mean(aps))), "ap50": _rounded(aps[0]),
                         "truth": size_truths[size]}
    return {"ap": mean("ap"), "ap50": mean("ap50"), "ap75": mean("ap75"), "classes": classes, "by_size": by_size}


def _point_summary(results, truths, kinds_found, kinds_total, false_points, pictures) -> dict:
    classes = {}
    for label in sorted(set(truths) | set(results)):
        entry = {"truth": truths[label], "predicted": len(results[label][RADII[0]])}
        for r in RADII:
            hits = sum(1 for _, h in results[label][r] if h)
            entry[f"ap@{r}"] = _rounded(average_precision(results[label][r], truths[label]))
            entry[f"found@{r}"] = _rounded(hits / truths[label] if truths[label] else None)
        classes[label] = entry
    scored = [c for c in classes.values() if c["truth"]]
    out = {}
    for r in RADII:
        out[f"ap@{r}"] = _rounded(float(np.mean([c[f"ap@{r}"] for c in scored]))) if scored else None
        out[f"found@{r}"] = _rounded(float(np.mean([c[f"found@{r}"] for c in scored]))) if scored else None
        out[f"false_per_picture@{r}"] = _rounded(false_points[r] / pictures if pictures else None)
    out["classes"] = classes
    out["by_visibility"] = {kind: {"truth": kinds_total[kind],
                                   **{f"found@{r}": _rounded(kinds_found[r][kind] / kinds_total[kind]
                                                             if kinds_total[kind] else None) for r in RADII}}
                            for kind in (SEEN, SELF, OTHER)}
    return out


def metrics(result: dict, prefix: str = "val") -> dict[str, float]:
    """The headline numbers, flat, for TrainingRun.final(**metrics(...))."""
    out = {}
    for key in ("ap", "ap50", "ap75"):
        if result.get("boxes", {}).get(key) is not None:
            out[f"{prefix}_box_{key}"] = result["boxes"][key]
    for r in RADII:
        for key in (f"ap@{r}", f"found@{r}"):
            value = result.get("keypoints", {}).get(key)
            if value is not None:
                out[f"{prefix}_kp_{key.replace('@', '')}"] = value
    return out


# --- The table ----------------------------------------------------------------


def _pct(v) -> str:
    return "  -  " if v is None else f"{v:5.3f}"


def report(result: dict) -> str:
    o = result["options"]
    lines = [f"{result['model']} on {result['dataset']} ({result['split']}): {result['pictures']} pictures"
             + (f", {result['not_predicted']} not predicted on" if result["not_predicted"] else "")]
    ignore = [f"objects under {o['min_pixels']} px" for _ in [0] if o["min_pixels"]] + \
             [f"objects under {o['min_share']:.0%} seen" for _ in [0] if o["min_share"]]
    if ignore:
        lines.append("ignoring " + " and ".join(ignore))
    if "boxes" in result:
        b = result["boxes"]
        lines += ["", f"Boxes ({o['boxes']})       AP    AP50  AP75  truth  predicted  best F1 (at score)"]
        for label, c in b["classes"].items():
            f1 = c["best_f1"]
            f1_text = f"{f1['f1']:.3f} (P {f1['precision']:.2f}, R {f1['recall']:.2f} at {f1['threshold']:.2f})" \
                if f1 else "-"
            lines.append(f"  {label:18} {_pct(c['ap'])} {_pct(c['ap50'])} {_pct(c['ap75'])} {c['truth']:6} "
                         f"{c['predicted']:10}  {f1_text}")
        if len(b["classes"]) > 1:
            lines.append(f"  {'mean':18} {_pct(b['ap'])} {_pct(b['ap50'])} {_pct(b['ap75'])}")
        lines.append("  by size: " + ", ".join(f"{size} AP50 {_pct(s['ap50']).strip()} ({s['truth']})"
                                               for size, s in b["by_size"].items()))
    if "keypoints" in result:
        k = result["keypoints"]
        heads = "  ".join(f"AP@{r}  found@{r}" for r in RADII)
        lines += ["", f"Keypoints ({'all' if o['all_points'] else 'seen'})   {heads}   truth  predicted"]
        for label, c in k["classes"].items():
            cells = "  ".join(f"{_pct(c[f'ap@{r}'])} {_pct(c[f'found@{r}'])}  " for r in RADII)
            lines.append(f"  {label:16} {cells} {c['truth']:6} {c['predicted']:10}")
        lines.append("  false points per picture: " + ", ".join(
            f"{k[f'false_per_picture@{r}']:.2f} within {r} px" for r in RADII if k[f"false_per_picture@{r}"] is not None))
        lines.append("  found, by kind: " + "; ".join(
            f"{kind} ({v['truth']}): " + ", ".join(f"{_pct(v[f'found@{r}']).strip()} @{r}" for r in RADII)
            for kind, v in k["by_visibility"].items() if v["truth"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shared.score", description=__doc__.split("\n\n")[0])
    parser.add_argument("predictions", help="a folder Predictions(...).finish() wrote")
    parser.add_argument("--dataset", help="its dataset.toml or model directory (default: the one it names)")
    parser.add_argument("--split", choices=SPLITS, help="default: the one it names, or validation")
    parser.add_argument("--min-pixels", type=int, default=0, metavar="N", help="ignore objects showing fewer pixels")
    parser.add_argument("--min-share", type=float, default=0.0, metavar="S",
                        help="ignore objects less than S (0 to 1) of which is seen")
    parser.add_argument("--boxes", choices=["visible", "full"], default="visible",
                        help="the truth to compare with: what's seen of each object, or all of it")
    parser.add_argument("--all-points", action="store_true", help="count hidden keypoints too")
    parser.add_argument("--classes", nargs="+", metavar="LABEL",
                        help="score these labels and keypoint classes (default: the ones predictions.json names)")
    args = parser.parse_args(argv)
    try:
        result = score(args.predictions, args.dataset, args.split, args.min_pixels, args.min_share, args.boxes,
                       args.all_points, args.classes)
    except (ValueError, KeyError) as e:
        print(f"score: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1
    print(report(result))
    path = Path(args.predictions) / "score.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
