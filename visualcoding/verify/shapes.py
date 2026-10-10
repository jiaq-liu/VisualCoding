"""Code-only shape geometry behind the `lib` primitives: ink / hue masks, connected components with hole filling,
contour statistics (area, circularity, vertex counts over an epsilon sweep) and a shape class from them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

HUES = {"red": [(0, 12), (345, 360)], "orange": [(12, 40)], "yellow": [(40, 70)], "green": [(70, 165)],
        "cyan": [(165, 200)], "blue": [(200, 255)], "purple": [(255, 300)], "pink": [(300, 345)], "magenta": [(290, 335)]}


def _arr(img: Image.Image, max_side: int = 2000) -> tuple[np.ndarray, float]:
    im = img.convert("RGB")
    s = 1.0
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return np.asarray(im), s


def ink_mask(a: np.ndarray, thr: int = 200) -> np.ndarray:
    """Non-white pixels (any channel below thr)."""
    return a.min(axis=2) < thr


def hue_mask(a: np.ndarray, name: str, s_min: float = 0.12, v_min: float = 0.2) -> np.ndarray | None:
    import cv2
    if name not in HUES:
        return None
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    m = np.zeros(H.shape, dtype=bool)
    for lo, hi in HUES[name]:
        m |= (H >= lo) & (H < hi)
    return m & (S > s_min) & (V > v_min)


def components(mask: np.ndarray, min_area: int = 20) -> list[dict]:
    import cv2
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({"id": i, "cx": float(cents[i][0]), "cy": float(cents[i][1]), "area": int(area),
                    "box": (int(x), int(y), int(x + w), int(y + h)), "w": int(w), "h": int(h)})
    return out, lab


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Mask with interior holes filled (outline shapes -> solid shapes)."""
    import cv2
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (1 - m).copy()
    cv2.floodFill(inv, ff, (0, 0), 2)          # background reachable from the corner -> 2
    holes = inv == 1                            # unreachable background = holes
    return mask | holes


def contour_stats(mask: np.ndarray) -> list[dict]:
    """Per external contour: area, perimeter, circularity, vertex count from approxPolyDP at several epsilons."""
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / max(per * per, 1e-9)
        verts = {}
        for eps in (0.01, 0.02, 0.03, 0.04):
            verts[eps] = len(cv2.approxPolyDP(c, eps * per, True))
        x, y, w, h = cv2.boundingRect(c)
        out.append({"area": float(area), "perimeter": float(per), "circularity": round(float(circ), 3), "verts": verts,
                    "box": (int(x), int(y), int(x + w), int(y + h)), "cx": x + w / 2, "cy": y + h / 2, "contour": c})
    return out


SHAPE_VERTS = {"triangle": 3, "square": 4, "rectangle": 4, "quadrilateral": 4, "diamond": 4, "rhombus": 4, "pentagon": 5,
               "hexagon": 6, "heptagon": 7, "octagon": 8}


def classify(c: dict) -> str:
    """circle / ellipse / n-gon / other from circularity and the vertex count stability."""
    v = c["verts"]
    if c["circularity"] >= 0.85 and v[0.02] >= 8:
        return "circle"
    if c["circularity"] >= 0.6 and v[0.02] >= 8:
        return "ellipse"
    if v[0.02] == v[0.03] == v[0.04] and v[0.02] in SHAPE_VERTS.values():
        return {n: k for k, n in SHAPE_VERTS.items() if k not in ("rectangle", "quadrilateral", "diamond", "rhombus")}.get(v[0.02], "other")
    return "other"
