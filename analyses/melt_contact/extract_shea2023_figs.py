"""Fe-Mg points read from Shea et al. (2023) Figs 5 and 6: Jurewicz & Watson (1988), Shea's own.

We do not have Jurewicz and Watson (1988) itself. Shea et al. plot its Fe-Mg diffusion
coefficients twice, as vector graphics in the accepted manuscript:

- Fig 5, Fe-Mg panel (pdf page 23): log D against 1000/T, with T in degrees C as
  their axis is labelled (1000/1300 = 0.769, top axis 1300 C at the same place);
- Fig 6, Fe-Mg panel (pdf page 24): log D measured against log D predicted by
  Dohmen and Chakraborty (2007) at the experiments' own conditions.

The grey squares (fill 0.82, 0.83, 0.83) are Jurewicz and Watson; in Fig 6 the
orange-red circles (fill 0.94, 0.35, 0.16) are Shea et al.'s own Fe-Mg points, read too
as the full-set check on the 8 profiles with a known axis (data/shea2023_table1.csv).
Each marker's centre
is mapped through a linear fit to that panel's labelled ticks. Fig 6 gives each
point's residual against DC07 as Shea et al. computed it, without our knowing the fO2,
Fo or orientation they used. The two figures do not plot the same set (18 points in
Fig 5, 19 in Fig 6, measured values differing by up to about 0.1 log), so they are
recorded separately and not paired.

These are figure readings, to be confirmed against Jurewicz and Watson's tables
(docs/references/figure-extracted-data.md).

Run: uv run --with pymupdf python analyses/melt_contact/extract_shea2023_figs.py [shea2023-am.pdf]
Writes data/shea2023_figs_extracted.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

HERE = Path(__file__).parent
PDF = Path.home() / "Downloads" / "1-s2.0-S0012821X23003837-am.pdf"
OUT = HERE / "data" / "shea2023_figs_extracted.csv"
GREY = (0.82, 0.83, 0.83)  # Jurewicz and Watson squares
SHEA_FEMG = (0.94, 0.35, 0.16)  # Shea et al. Fe-Mg circles
LEGEND_WORDS = ("Jurewicz", "seed", "phenocryst")  # legend entries; their swatches are dropped
# Fe-Mg panel boxes (pt) and their tick labels: (page index, x0, y0, x1, y1).
FIG5 = (22, 58.0, 45.0, 205.0, 200.0)
FIG6 = (23, 60.0, 0.0, 215.0, 145.0)


def _labels(page, box, axis: str) -> list[tuple[float, float]]:
    """(value, centre) of numeric labels left of (y) or below (x) the panel box."""
    _, x0, y0, x1, y1 = box
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip().replace(chr(0x2212), "-")  # Unicode minus
                try:
                    value = float(text)
                except ValueError:
                    continue
                bx0, by0, bx1, by1 = span["bbox"]
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                if axis == "y" and bx1 <= x0 + 2 and y0 - 10 <= cy <= y1 + 10 and abs(value) > 5:
                    out.append((value, cy))
                if axis == "x" and x0 - 10 <= cx <= x1 + 10 and y1 - 5 <= by0 <= y1 + 25:
                    out.append((value, cx))
    return out


def _axis_map(pairs: list[tuple[float, float]]) -> tuple[np.poly1d, float]:
    """Linear map from page coordinate to value, and its largest residual (value units)."""
    v, c = np.array(pairs).T
    fit = np.poly1d(np.polyfit(c, v, 1))
    return fit, float(np.abs(fit(c) - v).max())


def _squares(page, box, fill_rgb=GREY) -> list[tuple[float, float]]:
    """Centres of markers filled ``fill_rgb`` inside the panel box, legend excluded."""
    _, x0, y0, x1, y1 = box
    legend = [
        span["bbox"]
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if any(word in span["text"] for word in LEGEND_WORDS)
    ]
    out = []
    for d in page.get_drawings():
        fill = d.get("fill")
        if not fill or tuple(round(f, 2) for f in fill) != fill_rgb:
            continue
        r = d["rect"]
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        # A legend sample sits on the line of the legend text, just left of it.
        if any(b[0] - 15 <= cx <= b[0] and b[1] - 2 <= cy <= b[3] + 2 for b in legend):
            continue
        out.append((cx, cy))
    return out


def read_panel(doc, box, x_labels, y_labels, fill_rgb=GREY) -> tuple[pd.DataFrame, dict]:
    page = doc[box[0]]
    # Log D ticks only: the unit's superscripts ("2", "-1") are numbers too.
    x_labels = x_labels or [(v, c) for v, c in _labels(page, box, "x") if abs(v) > 5]
    fx, rx = _axis_map(x_labels)
    fy, ry = _axis_map(_labels(page, box, "y") if y_labels is None else y_labels)
    pts = _squares(page, box, fill_rgb)
    df = pd.DataFrame(
        {
            "x": [fx(x) for x, _ in pts],
            "y": [fy(y) for _, y in pts],
            "page_xy": [f"p{box[0] + 1}:{x:.1f},{y:.1f}" for x, y in pts],
        }
    )
    return df, {"x_residual": rx, "y_residual": ry, "n": len(df)}


def main() -> None:
    doc = pymupdf.open(sys.argv[1] if len(sys.argv) > 1 else PDF)
    # Fig 5: x labels sit under the bottom row of panels (same column, same scale).
    x5 = [
        (v, c)
        for v, c in _labels(doc[FIG5[0]], (FIG5[0], FIG5[1], 0, FIG5[3], 360.0), "x")
        if 0.7 < v < 0.9
    ]
    fig5, q5 = read_panel(doc, FIG5, x5, None)
    fig6, q6 = read_panel(doc, FIG6, None, None)
    fig5 = pd.DataFrame(
        {
            "figure": "Shea 2023 Fig 5 (Fe-Mg)",
            "page_xy": fig5["page_xy"],
            "T_C": 1000 / fig5["x"],
            "log10_D_m2s": fig5["y"],
            "extraction": f"axis map residual x {q5['x_residual']:.4f} (3 ticks), "
            f"y {q5['y_residual']:.4f} log (3 ticks); x is 1000/T with T in C, as labelled",
        }
    )
    fig6 = pd.DataFrame(
        {
            "figure": "Shea 2023 Fig 6 (Fe-Mg)",
            "page_xy": fig6["page_xy"],
            "log10_D_m2s": fig6["y"],
            "log10_D_pred_dc07": fig6["x"],
            "resid_dc07_shea": fig6["y"] - fig6["x"],
            "extraction": f"axis map residual y {q6['y_residual']:.4f} log (3 ticks); "
            "x from 2 labelled ticks (no check)",
        }
    )
    own, _ = read_panel(doc, FIG6, None, None, SHEA_FEMG)
    own = pd.DataFrame(
        {
            "source": "shea2023",
            "figure": "Shea 2023 Fig 6 (Fe-Mg)",
            "page_xy": own["page_xy"],
            "log10_D_m2s": own["y"],
            "log10_D_pred_dc07": own["x"],
            "resid_dc07_shea": own["y"] - own["x"],
            "extraction": fig6["extraction"].iloc[0],
        }
    )
    fig5.insert(0, "source", "jurewicz_watson1988")
    fig6.insert(0, "source", "jurewicz_watson1988")
    out = pd.concat([fig5, fig6, own], ignore_index=True)
    out["confirm"] = np.where(
        out["source"] == "shea2023",
        "read from a figure: match to Shea et al. Table 1 rows and confirm the DC07 "
        "prediction's conditions (orientation, Fo) in their supplementary table",
        "read from a figure: confirm T, log D, fO2, Fo, orientation and run against "
        "Jurewicz & Watson (1988) Contrib Mineral Petrol 99, 186-201; the DC07 prediction "
        "is Shea et al.'s, at conditions not stated in their paper",
    )
    out.to_csv(OUT, index=False)
    print(f"Fig 5: {q5}\nFig 6: {q6}")
    show = out.drop(columns=["extraction", "confirm", "source"])
    print(show.round(3).to_string(index=False))
    for src, sub in out.dropna(subset=["resid_dc07_shea"]).groupby("source"):
        r = sub["resid_dc07_shea"]
        print(f"{src}: Fig 6 residual vs DC07 mean {r.mean():+.2f}, sd {r.std():.2f}, n {len(r)}")


if __name__ == "__main__":
    main()
