"""Dependency-light drop-in for ``qwen_vl_utils.fetch_image``.

This is a pure-PIL port of qwen-vl-utils 0.0.14's ``smart_resize`` + ``fetch_image``
so the API-based pipeline no longer hard-imports ``qwen_vl_utils`` (which drags in
``torch`` / ``torchvision`` transitively) just to load and pixel-budget-resize an
image before base64-encoding it into the request. Semantics are byte-for-byte
compatible with the callers' usage: ``fetch_image({'image': path_or_url,
'max_pixels': N})`` -> RGB PIL image resized to the nearest 28-multiple that fits
the [min_pixels, max_pixels] budget, using PIL's default resample (matching qwen).
"""

import math
import base64
from io import BytesIO

from PIL import Image

# qwen-vl-utils 0.0.14 defaults (do not change — kept for resize parity).
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28          # 3136
MAX_PIXELS = 16384 * 28 * 28      # 12845056
MAX_RATIO = 200


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=IMAGE_FACTOR,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Return (h_bar, w_bar), each a multiple of ``factor``, such that
    height*width is within [min_pixels, max_pixels] and the aspect ratio is
    preserved as closely as possible. Identical to qwen-vl-utils 0.0.14."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _open_image(image):
    """Load a PIL image from a PIL object, http(s) URL, file:// URI, data-URL
    base64, or a local path (mirrors qwen-vl-utils fetch_image's loaders)."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            import requests  # lazy: only needed for remote images
            return Image.open(requests.get(image, stream=True).raw)
        if image.startswith("file://"):
            return Image.open(image[len("file://"):])
        if image.startswith("data:image"):
            if "base64," in image:
                _, b64 = image.split("base64,", 1)
                return Image.open(BytesIO(base64.b64decode(b64)))
            raise ValueError(f"Unsupported data-URL image: {image[:32]}...")
        return Image.open(image)
    raise ValueError(f"Unsupported image reference type: {type(image)}")


def fetch_image(ele, size_factor=IMAGE_FACTOR):
    """Drop-in for qwen_vl_utils.fetch_image. ``ele`` is a dict with an
    ``image`` (or ``image_url``) plus optional ``min_pixels`` / ``max_pixels``
    (or explicit ``resized_height`` / ``resized_width``)."""
    image_ref = ele["image"] if "image" in ele else ele["image_url"]
    image = _open_image(image_ref).convert("RGB")
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height, width, factor=size_factor,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
    return image.resize((resized_width, resized_height))
