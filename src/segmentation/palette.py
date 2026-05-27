from __future__ import annotations

from typing import List, Tuple


def default_palette_8() -> List[Tuple[int, int, int]]:
    # Stable, visually distinct colors for preview overlay only.
    return [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 127, 0),
    ]
