"""Flag a PNG that came out tiled (the page's top-left repeated across the image).

A tiled capture repeats itself at some shift, so shifted copies of the image match
almost exactly; a real figure does not. Prints "ok" or "TILED" per file, exits 1 if any
is tiled.

Run: uv run --with pillow --with numpy python tiled_check.py <png> [<png> ...]
"""

import sys

import numpy as np
from PIL import Image

# Mean absolute grey difference (0-255) below which a shift counts as a repeat. Calibrated
# 2026-10-16 at 1600 px: the four good figures score 8.1-10.7, two tiled renders 0.00-0.12
# (tiles can be as small as 1/6 of the width).
THRESHOLD = 3.0
MIN_SHIFT = 16  # pixels; smaller shifts match any image with broad flat areas


def repeat_score(path: str) -> tuple[float, str]:
    """Smallest mean difference between the image and a shifted copy of itself."""
    im = Image.open(path).convert("L")
    im.thumbnail((1600, 1600))
    a = np.asarray(im, dtype=np.float32)
    h, w = a.shape
    best = (float("inf"), "")
    for k in range(MIN_SHIFT, h - h // 8):
        best = min(best, (float(np.abs(a[: h - k] - a[k:]).mean()), f"rows shifted {k}"))
    for k in range(MIN_SHIFT, w - w // 8):
        best = min(best, (float(np.abs(a[:, : w - k] - a[:, k:]).mean()), f"columns shifted {k}"))
    return best


def main() -> int:
    tiled = False
    for path in sys.argv[1:]:
        score, where = repeat_score(path)
        bad = score < THRESHOLD
        tiled |= bad
        print(f"{'TILED' if bad else 'ok'}  {score:.2f} ({where})  {path}")
    return 1 if tiled else 0


if __name__ == "__main__":
    sys.exit(main())
