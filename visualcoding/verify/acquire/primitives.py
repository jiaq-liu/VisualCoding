"""Primitive library the acquired instruments compose: the `lib` argument of `measure(img, task, lib)`.

The library is fixed: image geometry, colour, components, projections, orientation panels, external detectors,
and ONE kind of model call (`lib.ask`) for a discrete input.  No family solver is provided: the model has to acquire
the skill itself.
`Lib.doc()` renders the signatures + one-line docs that go into the model prompt; only documented names are meant
to be used, everything else is reachable through numpy / cv2 / PIL which the code may import.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
from PIL import Image

from .. import shapes as S

_DOC = [
    ("arr(img, max_side=2000) -> (a, scale)", "RGB uint8 array (H,W,3), downscaled to max_side; scale = a-pixels per img-pixel. Every primitive below that takes `img` or `a` accepts EITHER the PIL image or an RGB array (a crop of `a` too)"),
    ("gray(img | a) -> g", "uint8 (H,W) luminance of the PIL image or of the array from arr (same shape as a)"),
    ("background(a | img) -> rgb", "median colour of the border pixels (use instead of assuming white; if the image border is a frame / wall, this IS the frame colour)"),
    ("fg_mask(a | img, tol=60, bg=None) -> mask", "pixels whose |rgb - background| L1 distance > tol (with a dark frame the mask is the light interior, not the frame)"),
    ("ink_mask(a | img | g, thr=200) -> mask", "pixels with ANY channel below thr: dark pixels and also saturated colours (pure red has two channels at 0); on a 2-D gray array it is g < thr; for a luminance-only threshold use gray(a) < thr"),
    ("hue_mask(a | img, name, s_min=0.12, v_min=0.2) -> mask|None", "pixels of a named hue: red orange yellow green cyan blue purple pink magenta"),
    ("stretch(img | a) -> img", "percentile (p1-p99) brightness stretch for dark / flat images"),
    ("upscale(img | a, min_side=400) -> img", "LANCZOS upscale so the short side is >= min_side"),
    ("rotate(img | a, deg) -> img", "rotate counter-clockwise, expanded canvas filled with the background colour"),
    ("components(mask, min_area=20) -> (comps, labels)", "8-connected components: dicts with id cx cy (floats: int() before indexing) area box w h; labels = int32 map"),
    ("fill_holes(mask) -> mask", "interior holes filled (outlines become solids)"),
    ("holes(labels, comp | id) -> n", "number of holes inside one component (a dict from components, or its id; rings have exactly one)"),
    ("contour_stats(mask) -> list", "per external contour: area perimeter circularity verts{eps: n} box cx cy"),
    ("classify(c) -> str", "circle / ellipse / triangle / square / pentagon / hexagon / ... / other from a contour_stats item"),
    ("min_area_rect(mask) -> dict", "cv2.minAreaRect of the mask pixels: angle w h fill cx cy n (fill = mask area / rect area)"),
    ("color_name(rgb) -> str", "nearest named hue of an RGB triple (or black / white / gray)"),
    ("n_colors(a | img, levels=16, min_share=0.002) -> int", "distinct colours after quantising each channel to `levels`, ignoring colours below min_share"),
    ("color_share(a | img, name) -> pct", "percentage of pixels of a named hue"),
    ("mean_color(a | img, mask=None) -> (rgb, hsv)", "mean RGB and HSV(deg, %, %) over the mask"),
    ("projection(mask, axis) -> profile", "sum of the mask along axis (0 = column profile, 1 = row profile)"),
    ("period(profile) -> dict", "dominant period of a 1-D profile by autocorrelation: period (float or None; int() before use as a step), strength (0-1)"),
    ("panels8(img | a) -> (sheet, variants)", "8-panel sheet of 4 rotations x mirror (labels 0-7) to ask which is the normal orientation"),
    ("apply_variant(img, variant) -> img", "apply one variant from panels8 to the image"),
    ("detect(img | a, phrase, backend='gdino', floor=0.1) -> (boxes, scores)", "open-vocabulary detector (GPU): boxes (N,4) xyxy pixels, scores (N,), both NUMPY ARRAYS (test emptiness with len(boxes) == 0; `if not boxes` raises on an array); the model is loaded once before the task loop (not inside the per-task timeout)"),
    ("count_objects(img | a, phrase, backend='owl') -> dict", "detector count over a threshold sweep; keys: count (at the reference threshold ref_thr), counts {thr: n}, spread (count variation across thresholds, 0 = stable), size_cv, med_side, n_tiles, boxes, scores - no other keys"),
    ("ask(prompt, img=None) -> dict|list|number|str|None", "ONE model call for a discrete input (a phrase, a box, a panel index, a transcription); img may be the PIL image, an array or a crop; the reply is JSON-parsed when it is JSON (so it may be a dict / list / number, not only a string); never for the final judgement"),
    ("question(task) -> str", "question text with the option lines removed"),
    ("numbers_in(x) -> list[float]", "all numbers in x: a string, a number, or a list / dict (e.g. the parsed reply of ask) flattened in order"),
    ("debug(img | a | mask | labels, name) -> None", "save an intermediate picture (a mask, a crop, a labels map, an annotated copy) under `name`, max 3 per task; next round the harness attaches, for the worst tasks, the pictures you saved next to their originals - the only way to SEE what your code computed instead of guessing from numbers"),
    ("check(name, ok, value=None) -> ok", "record a SUB-STEP assertion for this task (lib.check('grid_found', n_lines == 4, n_lines)); next round the harness reports per name how many train tasks pass, which fail, and how that splits between right and wrong readings - verify each stage of a construction on ALL train tasks before building the next on top of it (max 12 names per task)"),
]


class Lib:
    DEBUG_MAX = 3

    AUTO_MAX = 2

    def __init__(self, call=None, debug_dir=None, auto_render=False):
        self._call = call
        self.n_calls = 0
        self.contents: list[str] = []
        self.debug_dir = debug_dir            # where lib.debug saves pictures (None outside the acquisition runner)
        self.debug_files: list[dict] = []
        self.auto_render = bool(auto_render)  # the harness renders the first mask / labels map by itself
        self.auto_files: list[dict] = []
        self._auto_seen: set = set()
        self.checks: dict = {}                # sub-step assertions recorded with lib.check (per task)

    CHECK_MAX = 12

    def check(self, name, ok, value=None):
        """record a SUB-STEP assertion for this task, e.g. lib.check("grid_found", n_lines == 4, n_lines).  The
        harness reports, per name, how many train tasks pass, which fail, and how that splits between right and wrong
        readings - a stage of a construction (piece extraction, axis found, grid pitch) is verified on ALL train tasks
        before the next stage is written on top of it.  Returns `ok` so it can be used inline."""
        try:
            key = str(name)[:40]
            if key in self.checks or len(self.checks) < self.CHECK_MAX:
                v = value if (value is None or isinstance(value, (bool, int, float, str))) else str(value)[:40]
                if isinstance(v, float) and v != v:
                    v = None
                self.checks[key] = {"ok": bool(ok), "value": v}
        except Exception:  # noqa: BLE001
            pass
        return ok

    def _auto(self, kind: str, arr):
        """Save the FIRST result of each mask primitive / the first labels map (max AUTO_MAX per task) so the writer sees
        what its code segmented even when it saved nothing itself."""
        if not self.auto_render or self.debug_dir is None or kind in self._auto_seen or len(self.auto_files) >= self.AUTO_MAX or arr is None:
            return
        self._auto_seen.add(kind)
        try:
            self.auto_files.append(self._save_picture(arr, "auto:" + kind, auto=True))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ pictures for the writer
    def debug(self, img, name="debug"):
        """Save what the code computed so that the model writing the instrument can LOOK at it next round.  Accepts a
        PIL image, an RGB array, a gray array, a labels map (rescaled) or a boolean mask (white on black).  A no-op
        outside the acquisition runner and after DEBUG_MAX saves per task."""
        if self.debug_dir is None or len(self.debug_files) >= self.DEBUG_MAX:
            return None
        self.debug_files.append(self._save_picture(img, name, auto=False))
        return None

    def _save_picture(self, img, name, auto=False) -> dict:
        from pathlib import Path
        if isinstance(img, Image.Image):
            im = img.convert("RGB")
        else:
            a = np.asarray(img)
            if a.dtype == bool:
                im = Image.fromarray((a.astype(np.uint8) * 255), "L").convert("RGB")
            elif a.ndim == 2:
                if a.dtype != np.uint8:
                    mx = float(a.max()) if a.size else 0.0
                    a = (a.astype(np.float64) * (255.0 / mx)).astype(np.uint8) if mx > 0 else np.zeros(a.shape, np.uint8)
                im = Image.fromarray(a, "L").convert("RGB")
            else:
                im = self._pil(a)
        im = im.copy()
        im.thumbnail((640, 640))
        d = Path(self.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:40] or "debug"
        p = d / f"{'a' if auto else ''}{len(self.auto_files) if auto else len(self.debug_files)}_{safe}.png"
        im.save(p)
        return {"name": str(name)[:60], "path": str(p), "size": [im.width, im.height], "auto": bool(auto)}

    # ------------------------------------------------------------------ input coercion (13 of the 14 library crashes in the
    # the null runs of early versions were an array handed to a PIL-only primitive, a PIL image handed to an array primitive, or the
    # parsed reply of ask() handed to numbers_in; the model's expectation is reasonable, so the library accepts both)
    @staticmethod
    def _pil(img):
        if isinstance(img, Image.Image):
            return img
        a = np.asarray(img)
        if a.ndim == 2:
            return Image.fromarray(a.astype(np.uint8), "L").convert("RGB")
        return Image.fromarray(np.ascontiguousarray(a[..., :3]).astype(np.uint8), "RGB")

    @staticmethod
    def _a(a):
        if isinstance(a, Image.Image):
            return np.asarray(a.convert("RGB"))
        a = np.asarray(a)
        if a.ndim == 2:
            return np.repeat(a[..., None].astype(np.uint8), 3, axis=2)
        return a[..., :3]

    # ------------------------------------------------------------------ image basics
    def arr(self, img, max_side=2000):
        return S._arr(self._pil(img), max_side)

    def gray(self, img):
        """Luminance of the PIL image OR of an RGB array from `arr` (same shape as that array).  19 of 36 pro
        candidates of early runs crashed on gray(a) when this accepted only the PIL image."""
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return img.astype(np.uint8)
            a = img[..., :3].astype(np.float32)
            return np.clip(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2] + 0.5, 0, 255).astype(np.uint8)
        return np.asarray(img.convert("L"), dtype=np.uint8)

    def background(self, a):
        a = self._a(a)
        return np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)

    def fg_mask(self, a, tol=60, bg=None):
        a = self._a(a)
        bg = self.background(a) if bg is None else np.asarray(bg)
        m = np.abs(a.astype(np.int32) - bg[None, None, :].astype(np.int32)).sum(-1) > tol
        self._auto(f"fg_mask(tol={tol})", m)
        return m

    def ink_mask(self, a, thr=200):
        if isinstance(a, np.ndarray) and a.ndim == 2:      # a gray array: luminance threshold (seen twice on maze: AxisError)
            m = a < thr
        else:
            m = S.ink_mask(self._a(a), thr)
        self._auto(f"ink_mask(thr={thr})", m)
        return m

    def hue_mask(self, a, name, s_min=0.12, v_min=0.2):
        m = S.hue_mask(np.ascontiguousarray(self._a(a)), name, s_min, v_min)
        self._auto(f"hue_mask({name})", m)
        return m

    def stretch(self, img):
        from ..instruments import stretch_contrast
        return stretch_contrast(self._pil(img))

    def upscale(self, img, min_side=400):
        from ..instruments import upscale
        return upscale(self._pil(img), min_side)

    def rotate(self, img, deg):
        img = self._pil(img)
        a, _ = self.arr(img, 10_000)
        bg = tuple(int(v) for v in self.background(a))
        return img.convert("RGB").rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=bg)

    # ------------------------------------------------------------------ components / shapes
    def components(self, mask, min_area=20):
        comps, labels = S.components(np.asarray(mask, dtype=bool), min_area)
        self._auto("components labels", labels)
        return comps, labels

    def fill_holes(self, mask):
        return S.fill_holes(np.asarray(mask, dtype=bool))

    def holes(self, labels, comp):
        import cv2
        if isinstance(comp, (int, np.integer)):        # the component id instead of the dict
            ys, xs = np.where(np.asarray(labels) == int(comp))
            if len(ys) == 0:
                return 0
            comp = {"id": int(comp), "box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)}
        x0, y0, x1, y1 = comp["box"]
        sub = (labels[y0:y1, x0:x1] == comp["id"]).astype(np.uint8)
        _, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return 0
        return int(sum(1 for h in hier[0] if h[3] != -1))

    def contour_stats(self, mask):
        out = S.contour_stats(np.asarray(mask, dtype=bool))
        for c in out:
            c.pop("contour", None)
        return out

    def classify(self, c):
        return S.classify(c)

    def min_area_rect(self, mask):
        import cv2
        m = np.asarray(mask, dtype=bool)
        ys, xs = np.where(m)
        if len(xs) < 20:
            return {"ok": False, "n": int(len(xs))}
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        return {"ok": True, "angle": float(ang), "w": float(w), "h": float(h), "cx": float(cx), "cy": float(cy),
                "fill": float(m.sum() / max(1.0, w * h)), "n": int(len(xs))}

    # ------------------------------------------------------------------ colour
    def color_name(self, rgb):
        import cv2
        px = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
        h, s, v = h * 2.0, s / 255.0, v / 255.0
        if v < 0.15:
            return "black"
        if s < 0.12:
            return "white" if v > 0.85 else "gray"
        for name, ranges in S.HUES.items():
            if any(lo <= h < hi for lo, hi in ranges):
                return name
        return "other"

    def n_colors(self, a, levels=16, min_share=0.002):
        a = self._a(a)
        q = (a.astype(np.int32) * levels // 256)
        key = q[..., 0] * levels * levels + q[..., 1] * levels + q[..., 2]
        vals, counts = np.unique(key, return_counts=True)
        return int((counts / key.size >= min_share).sum())

    def color_share(self, a, name):
        a = self._a(a)
        m = S.hue_mask(a, name)
        return float(m.mean() * 100.0) if m is not None else 0.0

    def mean_color(self, a, mask=None):
        a = self._a(a)
        import cv2
        sel = a.reshape(-1, 3) if mask is None else a[np.asarray(mask, dtype=bool)]
        if len(sel) == 0:
            return None, None
        rgb = sel.mean(axis=0)
        px = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
        return [round(float(x), 1) for x in rgb], [round(h * 2.0, 1), round(s / 255.0 * 100, 1), round(v / 255.0 * 100, 1)]

    # ------------------------------------------------------------------ periodic structure
    def projection(self, mask, axis):
        return np.asarray(mask, dtype=np.float64).sum(axis=axis)

    def period(self, profile):
        """Dominant period of a 1-D profile: the highest autocorrelation peak AFTER the first local minimum (the
        initial decay from lag 0 is not a period)."""
        p = np.asarray(profile, dtype=np.float64)
        p = p - p.mean()
        if len(p) < 8 or np.abs(p).sum() == 0:
            return {"period": None, "strength": 0.0}
        ac = np.correlate(p, p, mode="full")[len(p) - 1:]
        ac = ac / max(ac[0], 1e-9)
        half = max(len(ac) // 2, 4)
        lo = 1
        while lo + 1 < half and ac[lo + 1] <= ac[lo]:
            lo += 1                                       # descend to the first local minimum
        seg = ac[lo:half]
        if len(seg) < 2:
            return {"period": None, "strength": 0.0}
        i = int(np.argmax(seg)) + lo
        strength = float(ac[i] - np.median(ac[1:half]))  # peak prominence over the autocorrelation floor
        return {"period": float(i), "strength": round(min(max(strength, 0.0), 1.0), 3)}

    # ------------------------------------------------------------------ panels
    def panels8(self, img):
        from ..instruments import orientation_panels
        return orientation_panels(self._pil(img))

    def apply_variant(self, img, variant):
        from ..instruments import apply_variant
        return apply_variant(self._pil(img), variant)

    # ------------------------------------------------------------------ detectors (lazy, GPU)
    def detect(self, img, phrase, backend="gdino", floor=0.1):
        from .. import detector
        return detector.detect(self._pil(img), phrase, backend=backend, floor=floor)

    def count_objects(self, img, phrase, backend="owl"):
        from .. import detector
        return detector.count_objects(self._pil(img), phrase, backend=backend)

    @staticmethod
    def warm_detectors(code: str) -> list[str]:
        """Load the detector weights named in `code` BEFORE the per-task timeout starts (the first call took
        26 s on a busy GPU; with the 30 s alarm the load was killed mid-import and every later call failed with a
        partially initialised `accelerate` module - 3 candidates in an early run)."""
        if not re.search(r"lib\.(detect|count_objects)\(", code or ""):
            return []
        from .. import detector
        names = []
        if re.search(r"lib\.count_objects\(", code) or re.search(r"backend\s*=\s*['\"]owl['\"]", code):
            names.append(detector.OWL)
        if re.search(r"lib\.detect\(", code) and not re.search(r"backend\s*=\s*['\"]owl['\"]", code) or re.search(r"['\"]gdino['\"]", code):
            names.append(detector.GDINO)
        for n in names:
            detector._load(n)
        return names

    # ------------------------------------------------------------------ model call for a discrete input
    def ask(self, prompt, img=None):
        if self._call is None:
            raise RuntimeError("lib.ask is not available in this run (no model configured)")
        from ..instruments import json_load, msgs, part
        content = self._call(msgs([part(self._pil(img))] if img is not None else [], prompt))
        self.n_calls += 1
        self.contents.append(content)
        obj = json_load(content)
        return obj if obj is not None else content

    # ------------------------------------------------------------------ task helpers
    def question(self, task):
        from ..instruments import strip_question
        return strip_question(task)

    def numbers_in(self, text):
        """Numbers in a string; a number, list or dict (the parsed reply of ask) is flattened first (6 candidates
        crashed on `.replace` of a dict / list / int / float)."""
        if text is None:
            return []
        if isinstance(text, bool):
            return [float(text)]
        if isinstance(text, (int, float)):
            return [float(text)]
        if isinstance(text, dict):
            return [x for v in text.values() for x in self.numbers_in(v)]
        if isinstance(text, (list, tuple)):
            return [x for v in text for x in self.numbers_in(v)]
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))]

    # ------------------------------------------------------------------ documentation
    def doc(self) -> str:
        lines = ["Library `lib` (call as lib.<name>). The library is a starting point, not a boundary: numpy as np, cv2, PIL.Image, scipy "
                 "(ndimage / signal / optimize), skimage, math, re, json, collections, itertools, statistics, functools may be imported and any "
                 "algorithm written from them; the constraint is on WHAT is measured (a value computed from pixels), not on which code computes it:"]
        for sig, d in _DOC:
            lines.append(f"- lib.{sig}: {d}")
        return "\n".join(lines)
