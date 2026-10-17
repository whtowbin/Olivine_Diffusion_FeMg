"""Illustrative zoned olivine section with two traverses, and its Fo map after diffusion.

Frame: x along c [001] (page right), y along a [100] (page down), b [010] normal to the section
(an illustrative choice). Length in arbitrary units (the same units as make_profiles.py).

* outline: a generic convex bevelled polygon (habit unknown, no faces claimed);
* zones: the outline scaled about the core centre, by different factors along x and y (growth
  differs by direction), giving segments rim zone, mantle, core;
* traverse 1 at 15 deg from c, traverse 2 at 15 deg from a, orthogonal, crossing at X in the core;
* the Fo map: 2D diffusion with D_x = D[001] = 6, D_y = D[100] = 1 (explicit finite differences),
  Fo held at the edge value outside the outline, for the same time as make_profiles.py.

Run (from the project folder): uv run --with numpy python figure/data/make_crystal.py
Writes figure/data/crystal.json (geometry, steps per traverse, and the map as a base64 PNG).
"""

import base64
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

OUTLINE = [(-70, -12), (-50, -41), (46, -42), (71, -14), (72, 10), (49, 41), (-47, 42), (-71, 13)]
CORE_CENTRE = (2.0, 1.0)
ZONES = {"mantle": (0.78, 0.80), "core": (0.47, 0.55)}  # (fx, fy) scale of each inner boundary
LEVELS = {"edge": 0.0, "rim": 0.9, "mantle": 2.0, "core": 3.0}  # same as make_profiles.py
CROSS = (8.0, 3.0)  # crossing point X
ANG = math.radians(15.0)
U1 = (math.cos(ANG), -math.sin(ANG))  # traverse 1 direction: 15 deg from c, inward from left rim
U2 = (math.sin(ANG), math.cos(ANG))  # traverse 2 direction: 15 deg from a, inward from top rim
EDGE_OFF = {"p1": 3.0, "p2": 2.0}  # first spot this far inside the rim (drawn enlarged)
SPACING = 2.0
BEYOND = 6.0  # spots continue this far past the crossing
D_X, D_Y = 6.0, 1.0
D1 = 1.0 * math.cos(math.pi / 2 - ANG) ** 2 + 6.0 * math.cos(ANG) ** 2
T_TRUE = 16.0 / (4 * D1)  # as make_profiles.py
RAMP = ["#73733a", "#a3ad5f", "#cfd99c", "#f1f4e1"]  # low Fo (edge) -> high Fo (core)


def scaled(poly, f, c=CORE_CENTRE):
    return [(c[0] + (x - c[0]) * f[0], c[1] + (y - c[1]) * f[1]) for x, y in poly]


POLYS = {"rim": OUTLINE, **{k: scaled(OUTLINE, f) for k, f in ZONES.items()}}


def exit_distance(poly, p, d):
    """Distance from interior point p along unit d to the convex polygon's boundary."""
    best = math.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        den = d[0] * ey - d[1] * ex
        if abs(den) < 1e-12:
            continue
        s = ((a[0] - p[0]) * ey - (a[1] - p[1]) * ex) / den
        u = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / den
        if s > 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            best = min(best, s)
    return best


def traverse(name, u):
    back = (-u[0], -u[1])
    rim_to_x = exit_distance(OUTLINE, CROSS, back)
    rim = (CROSS[0] + back[0] * rim_to_x, CROSS[1] + back[1] * rim_to_x)
    steps = [rim_to_x - exit_distance(POLYS[k], CROSS, back) for k in ("mantle", "core")]
    off = EDGE_OFF[name]
    n = int((rim_to_x - off + BEYOND) // SPACING) + 1
    spots = [
        (rim[0] + u[0] * (off + i * SPACING), rim[1] + u[1] * (off + i * SPACING)) for i in range(n)
    ]
    return {
        "rim": rim,
        "dir": u,
        "rim_to_cross": rim_to_x,
        "steps_model": steps,  # from the true rim
        "edge_off": off,
        "cross_meas": rim_to_x - off,  # distance of X from the first spot
        "x_end_meas": (n - 1) * SPACING,
        "spots": spots,
    }


def inside(poly, X, Y):
    m = np.ones_like(X, dtype=bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        m &= (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]) >= 0
    return m


def png_rgba(rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


# orientation check: the polygon must be counter-clockwise in (x right, y down) for inside()
area = sum(
    OUTLINE[i][0] * OUTLINE[(i + 1) % 8][1] - OUTLINE[(i + 1) % 8][0] * OUTLINE[i][1]
    for i in range(8)
)
if area < 0:
    raise SystemExit("outline orientation flipped; inside() would be empty")

H = 0.2
X0, X1, Y0, Y1 = -74.0, 76.0, -45.0, 45.0
xs = np.arange(X0 + H / 2, X1, H)
ys = np.arange(Y0 + H / 2, Y1, H)
X, Y = np.meshgrid(xs, ys)
m_rim, m_mantle, m_core = (inside(POLYS[k], X, Y) for k in ("rim", "mantle", "core"))
if not (m_rim.any() and m_core.any()):
    raise SystemExit("empty zone mask")
c = np.full(X.shape, LEVELS["edge"])
c[m_rim] = LEVELS["rim"]
c[m_mantle] = LEVELS["mantle"]
c[m_core] = LEVELS["core"]
dt = 0.9 * H * H / (2 * (D_X + D_Y))
nsteps = math.ceil(T_TRUE / dt)
dt = T_TRUE / nsteps
for _ in range(nsteps):
    lap_x = np.zeros_like(c)
    lap_y = np.zeros_like(c)
    lap_x[:, 1:-1] = c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]
    lap_y[1:-1, :] = c[2:, :] - 2 * c[1:-1, :] + c[:-2, :]
    c = c + dt / (H * H) * (D_X * lap_x + D_Y * lap_y)
    c[~m_rim] = LEVELS["edge"]  # Fo held at the edge value at and beyond the outline


def ramp(t):
    t = np.clip(t, 0, 1) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    f = t - i
    rgb = np.array([[int(h[k : k + 2], 16) for k in (1, 3, 5)] for h in RAMP], float)
    return rgb[i] * (1 - f[..., None]) + rgb[i + 1] * f[..., None]


rgb = ramp((c - LEVELS["edge"]) / (LEVELS["core"] - LEVELS["edge"]))
alpha = np.where(m_rim, 255, 0)
img = np.dstack([rgb, alpha]).round().astype(np.uint8)

out = {
    "note": "illustrative geometry; habit unknown; computed map",
    "units": "arbitrary length units, as make_profiles.py",
    "outline": OUTLINE,
    "zones": {k: POLYS[k] for k in ("mantle", "core")},
    "cross": CROSS,
    "core_centre": CORE_CENTRE,
    "p1": traverse("p1", U1),
    "p2": traverse("p2", U2),
    "map": {
        "extent": [X0, Y0, X1 - X0, Y1 - Y0],
        "px": [img.shape[1], img.shape[0]],
        "png": "data:image/png;base64," + base64.b64encode(png_rgba(img)).decode(),
    },
    "ramp": RAMP,
    "levels": LEVELS,
    "time": T_TRUE,
    "steps": nsteps,
}
Path(__file__).with_name("crystal.json").write_text(json.dumps(out))
for k in ("p1", "p2"):
    t = out[k]
    print(
        k,
        "rim->X",
        round(t["rim_to_cross"], 2),
        "steps",
        [round(s, 2) for s in t["steps_model"]],
        "cross_meas",
        round(t["cross_meas"], 2),
        "n spots",
        len(t["spots"]),
        "x_end",
        t["x_end_meas"],
    )
print("map px", out["map"]["px"], "nsteps", nsteps)
