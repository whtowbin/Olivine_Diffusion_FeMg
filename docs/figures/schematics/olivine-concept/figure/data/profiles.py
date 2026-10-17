"""Illustrative Fo profiles for the zoned-olivine concept figure (panel b).

One step at distance X0 from the rim; the rim (x = 0) is held at the rim-zone Fo (FORSTER's
edge 1), the core side is semi-infinite. Exact solution by the method of images:

    u(x, t) = dFo/2 * [erfc((x0 - x) / (2 sqrt(D t))) - erfc((x0 + x) / (2 sqrt(D t)))]

Arbitrary units: x0 = 1, dFo = 1 (0 = rim Fo, 1 = core Fo), D along a = 1, D along c = 6
(the Fe-Mg anisotropy in FORSTER's laws: [001] six times faster than [100] and [010]).
The times are chosen only so the curves are distinct. Nothing here is a real value.
"""

import json
import math
from pathlib import Path

import numpy as np

X0 = 1.0
XMAX = 2.6
D = {"a": 1.0, "c": 6.0}
DT_A = [0.003, 0.015, 0.05]  # D_a * t for t1, t2, t3 (illustrative)

erfc = np.vectorize(math.erfc)


def profile(x: np.ndarray, dt: float) -> np.ndarray:
    s = 2.0 * math.sqrt(dt)
    return 0.5 * (erfc((X0 - x) / s) - erfc((X0 + x) / s))


def main() -> None:
    x = np.linspace(0.0, XMAX, 441)
    out: dict = {"x0": X0, "xmax": XMAX, "x": x.round(5).tolist(), "curves": {}, "widths": {}}
    for axis, d in D.items():
        curves = []
        widths = []
        for dta in DT_A:
            dt = d * dta
            u = profile(x, dt)
            if not np.all(np.isfinite(u)) or u.max() <= 0:
                raise RuntimeError(f"bad profile for {axis} at D_a t = {dta}")
            curves.append(u.round(5).tolist())
            widths.append(math.sqrt(dt))  # the characteristic length sqrt(D t)
        out["curves"][axis] = curves
        out["widths"][axis] = widths
    ratio = out["widths"]["c"][-1] / out["widths"]["a"][-1]
    assert abs(ratio - math.sqrt(6.0)) < 1e-9, ratio
    path = Path(__file__).with_name("profiles.json")
    path.write_text(json.dumps(out))
    print(f"wrote {path}; width ratio c/a = {ratio:.4f} (sqrt 6 = {math.sqrt(6):.4f})")
    for axis in D:
        u3 = np.array(out["curves"][axis][-1])
        print(axis, "u at rim", u3[0], "u at far end", round(u3[-1], 4))


if __name__ == "__main__":
    main()
