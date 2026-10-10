"""External open-vocabulary detectors used as a counting instrument (OWLv2 and Grounding DINO via transformers).

The detector is an instrument in the strict sense: a fixed procedure (phrase -> boxes) whose reading (box count
after tiling + cross-tile NMS, at several score thresholds) is converted to the answer by code.  The only
model-supplied input is the phrase.  Stability of the count across thresholds is the instrument's own
consistency check.  Reached through the `lib.detect` / `lib.count_objects` primitives (the refcoco program).
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

_LOCK = threading.Lock()
_MODELS: dict[str, tuple] = {}
OWL = os.environ.get("VC_OWL", "google/owlv2-base-patch16-ensemble")
GDINO = os.environ.get("VC_GDINO", "IDEA-Research/grounding-dino-base")
THRESHOLDS = (0.1, 0.15, 0.2, 0.3)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(name: str):
    with _LOCK:
        if name in _MODELS:
            return _MODELS[name]
        if "owl" in name.lower():
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            proc = Owlv2Processor.from_pretrained(name)
            model = Owlv2ForObjectDetection.from_pretrained(name).to(_device()).eval()
        else:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            proc = AutoProcessor.from_pretrained(name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(_device()).eval()
        _MODELS[name] = (proc, model)
        return _MODELS[name]


def detect(img: Image.Image, phrase: str, backend: str = "owl", floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """All boxes (N,4: x0,y0,x1,y1 px of `img`) with score >= floor and their scores, for one phrase."""
    import torch
    name = OWL if backend == "owl" else GDINO
    proc, model = _load(name)
    img = img.convert("RGB")
    if backend == "owl":
        inputs = proc(text=[["a photo of a " + phrase.strip().lower()]], images=img, return_tensors="pt").to(_device())
        with _LOCK:
            with torch.no_grad():
                out = model(**inputs)
        m = max(img.size)   # OWLv2 pads the image to a square before resizing
        res = proc.post_process_object_detection(out, threshold=floor, target_sizes=torch.tensor([[m, m]]))[0]
    else:
        text = phrase.strip().lower().rstrip(".") + "."
        inputs = proc(images=img, text=text, return_tensors="pt").to(_device())
        with _LOCK:
            with torch.no_grad():
                out = model(**inputs)
        res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=floor, text_threshold=floor,
                                                          target_sizes=[img.size[::-1]])[0]
    boxes = res["boxes"].detach().cpu().numpy().astype(float).reshape(-1, 4)
    scores = res["scores"].detach().cpu().numpy().astype(float).reshape(-1)
    w, h = img.size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    ok = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    return boxes[ok], scores[ok]


def drop_containers(boxes: np.ndarray, scores: np.ndarray, min_inside: int = 2, area_ratio: float = 4.0) -> np.ndarray:
    """Indices of boxes that are NOT 'group boxes': a box that contains (>= 70 % of their area) at least
    `min_inside` other boxes and is >= area_ratio x their median area is a whole-group detection, not an instance."""
    n = len(boxes)
    if n < 3:
        return np.arange(n)
    areas = ((boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0))
    xx0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0]); yy0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    xx1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2]); yy1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = (xx1 - xx0).clip(0) * (yy1 - yy0).clip(0)
    frac_inside = inter / np.maximum(1e-6, areas[None, :])      # [i, j] = fraction of j inside i
    np.fill_diagonal(frac_inside, 0.0)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        inside = np.where(frac_inside[i] >= 0.7)[0]
        if len(inside) >= min_inside and areas[i] >= area_ratio * np.median(areas[inside]):
            keep[i] = False
    return np.where(keep)[0]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Indices kept by greedy IoU NMS."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores)
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx1 - xx0).clip(0) * (yy1 - yy0).clip(0)
        iou = inter / np.maximum(1e-6, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def clean(boxes: np.ndarray, scores: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    m = scores >= thr
    b, s = boxes[m], scores[m]
    if len(b):
        k = nms(b, s, 0.5); b, s = b[k], s[k]
        k = drop_containers(b, s); b, s = b[k], s[k]
    return b, s


def count_objects(img: Image.Image, phrase: str, backend: str = "owl", thresholds=THRESHOLDS, floor: float = 0.05,
                  small: float = 0.035, max_depth: int = 2, min_tile: int = 200, overlap: float = 0.15) -> dict:
    """Detect on the whole image; when the instances are small relative to the region (median box side < `small`
    of the region's long side at the lowest threshold) the region is re-detected in 2x2 overlapping sub-tiles
    (recursively, up to max_depth), so that every instance is seen at a usable scale.  All boxes are pooled and
    cleaned (NMS + group-box removal) once per threshold.  Returns counts per threshold and diagnostics."""
    w, h = img.size
    pool_b, pool_s, tiles = [], [], []

    def region(x0, y0, x1, y1, depth):
        crop = img.crop((x0, y0, x1, y1))
        b, s = detect(crop, phrase, backend, floor)
        bb, _ = clean(b, s, thresholds[0])
        tiles.append((x0, y0, x1, y1, int(len(bb))))
        if len(bb) >= 3:
            med = float(np.median(np.minimum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])))
            is_small = med < small * max(x1 - x0, y1 - y0)
        else:
            is_small = False
        if is_small and depth < max_depth and min(x1 - x0, y1 - y0) >= 2 * min_tile:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ox, oy = (x1 - x0) * overlap / 2, (y1 - y0) * overlap / 2
            for (a0, b0, a1, b1) in ((x0, y0, mx + ox, my + oy), (mx - ox, y0, x1, my + oy),
                                     (x0, my - oy, mx + ox, y1), (mx - ox, my - oy, x1, y1)):
                region(int(a0), int(b0), int(a1), int(b1), depth + 1)
            return
        if len(b):
            b = b.copy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            pool_b.append(b)
            pool_s.append(s)

    region(0, 0, w, h, 0)
    B = np.concatenate(pool_b) if pool_b else np.zeros((0, 4))
    S = np.concatenate(pool_s) if pool_s else np.zeros(0)
    counts = {}
    kept = {}
    for thr in thresholds:
        b, s = clean(B, S, thr)
        counts[thr] = int(len(b))
        kept[thr] = (b, s)
    ref = thresholds[len(thresholds) // 2]
    b, s = kept[ref]
    if len(b):
        sides = np.sqrt((b[:, 2] - b[:, 0]).clip(1) * (b[:, 3] - b[:, 1]).clip(1))
        size_cv = float(np.std(sides) / max(1e-6, np.mean(sides)))
        med_side = float(np.median(sides)) / max(w, h)
    else:
        size_cv, med_side = 0.0, 0.0
    cs = np.array([counts[t] for t in thresholds], dtype=float)
    spread = float((cs.max() - cs.min()) / max(1.0, np.median(cs)))
    return {"counts": {str(k): v for k, v in counts.items()}, "count": counts[ref], "ref_thr": ref, "spread": round(spread, 3),
            "size_cv": round(size_cv, 3), "med_side": round(med_side, 4), "n_tiles": len(tiles), "tiles": tiles,
            "boxes": b.round(1).tolist(), "scores": s.round(3).tolist()}
