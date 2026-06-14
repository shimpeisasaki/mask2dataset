from __future__ import annotations

import colorsys
from typing import List, Tuple


def default_palette(count: int) -> List[Tuple[int, int, int]]:
    """Return a deterministic RGB palette with at least ``count`` entries."""
    n = max(0, int(count))
    if n == 0:
        return []

    # Keep the original first colors for backwards visual consistency.
    base: List[Tuple[int, int, int]] = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 127, 0),
    ]
    if n <= len(base):
        return base[:n]

    out = list(base)
    for i in range(len(base), n):
        # Golden-ratio stepping yields stable, well-spaced hues.
        h = (i * 0.6180339887498949) % 1.0
        s = 0.68
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return out


def default_palette_8() -> List[Tuple[int, int, int]]:
    return default_palette(8)
