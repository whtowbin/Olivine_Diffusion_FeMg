"""Transcribe Oeser, Dohmen and Weyer (2026) Tables 1-3 from the PDF's text.

Oeser, M., Dohmen, R. and Weyer, S. (2026), Fe-Mg in olivine: simultaneous experimental
determination of the diffusion-driven isotope fractionation along with tracer- and
inter-diffusion coefficients. GCA 413, 48-66, doi:10.1016/j.gca.2025.09.036 (CC BY).

Powder source (Fo0.6, 25Mg- and 57Fe-doped) on oriented San Carlos olivine cuboids,
1 atm gas mixing (CO-CO2), 1100-1303 C, fO2 1e-7.1 to 1e-2.3 Pa. Table 2: runs at about
1e-5 Pa; Table 3: variable fO2. One row per profile. The values are the tables' own text
(read from the PDF text layer, not from figures), with italic cells (convolution
corrected, laser linescans) flagged from the font. Each row takes its run's T and fO2
from Table 1. D_Fe-Mg is for X_Fe = 0.085 (Fo 91.5), as the paper's Figs 6 and 7 state.

Run: uv run --with pymupdf python analyses/femg_refit/transcribe_oeser2026.py [pdf]
Writes data/oeser2026.csv in the column layout of the other data files, plus the
tracer coefficients, beta values and techniques.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pymupdf

HERE = Path(__file__).parent
PDF = Path.home() / "Downloads" / "1-s2.0-S0016703725005228-main.pdf"
OUT = HERE / "data" / "oeser2026.csv"
X_FO = 0.915  # X_Fe = 0.085 (Figs 6 and 7 captions)
MINUS = chr(0x2212)
DASH = (MINUS, "-", chr(0x2013))  # minus, hyphen, en dash: an empty cell
TECH = {"dp", "LS", "EPMA"}
# Table 1: run -> (T C, log10 fO2 Pa, delta QFM, hours).
RUNS = {
    2: (1303, -2.3, -0.1, 19.2),
    3: (1100, -5.0, -0.5, 146.3),
    4: (1199, -4.8, -1.5, 138.5),
    5: (1250, -4.7, -2.0, 93.2),
    6: (1150, -5.0, -1.1, 168.0),
    7: (1300, -5.0, -2.8, 72.0),
    14: (1255, -2.7, -0.1, 95.5),
    15: (1256, -7.0, -4.4, 71.5),
    16: (1150, -3.7, 0.2, 172.7),
    17: (1156, -7.1, -3.3, 162.4),
}
COLUMNS = ("beta_Fe", "beta_Mg", "beta_ratio", "dFo")
COLUMNS_D = ("log10_Dstar_Fe", "log10_Dstar_Mg", "Dstar_ratio", "log10_D_FeMg")


Token = tuple[str, bool, tuple[float, float, float, float]]  # text, italic, bbox
# pt: a sign span this close to the next span belongs to it. Measured: a minus sign sits
# 3.3 pt from its number (its glyph box is offset), an empty cell's dash >= 22 pt away.
TOUCH = 5.0


def tokens(page, start: str, stop: str) -> list[Token]:
    """(text, italic, bbox) of each span between the first span containing ``start`` and
    the first later span containing ``stop``, in the page's reading order."""
    out, on = [], False
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if not on and start in text:
                    on = True
                    continue
                if on and stop in text:
                    return out
                if on and text:
                    italic = bool(span["flags"] & 2) or "Italic" in span["font"]
                    out.append((text, italic, tuple(span["bbox"])))
    raise ValueError(f"{stop!r} not found after {start!r}")


def _touch(a, b) -> bool:
    """Whether two span boxes touch (the gap between them is under TOUCH in x and y)."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return dx < TOUCH and dy < TOUCH


def _merge_markers(toks: list[Token]) -> list[tuple[str, bool, str]]:
    """One token per cell. The PDF sets signs and comparisons (a minus as U+2212, "<", ">",
    "~") as their own spans: one that touches the next span belongs to it (the value keeps
    its own italic flag); a lone minus sign that does not is an empty cell. Footnote
    markers ("+", "*", "‡") attach to the value before them."""
    out: list[tuple[str, bool, str]] = []
    prefix = ""
    for k, (raw, italic, box) in enumerate(toks):
        text, mark = raw, ""
        while len(text) > 1 and text[-1] in "+*‡":
            mark, text = text[-1] + mark, text[:-1]
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if text in {MINUS, "<", ">", "~"} and nxt is not None and _touch(box, nxt[2]):
            prefix += text
            continue
        if text in {"+", "*", "‡"} and out:
            prev = out[-1]
            out[-1] = (prev[0], prev[1], prev[2] + text)
            continue
        out.append((prefix + text, italic, mark))
        prefix = ""
    return out


def _value(text: str) -> float | None:
    t = text.replace(MINUS, "-").rstrip("+*")
    if t in DASH or t.startswith(("<", ">", "~", "n.q")):
        return None
    return float(t)


def _is_direction(text: str) -> bool:
    return "//" in text


def _axis(text: str) -> str:
    letter = text.replace("~", "").replace("//", "").strip()
    return {"a": "[100]", "b": "[010]", "c": "[001]"}[letter]


def parse(toks, table: str, run_of, has_conv: bool) -> list[dict]:
    """Rows of Table 2 (``has_conv``: an optional conventional-fit D before the techniques)
    or Table 3. A group starts with T (Table 2) or T and fO2 (Table 3)."""
    toks = _merge_markers(toks)
    # Skip the header: the body starts at the first temperature (1100-1303 C).
    i = next(k for k, (t, _, _) in enumerate(toks) if t.isdigit() and 1000 < int(t) < 1400)
    rows, run = [], None
    while i < len(toks):
        text = toks[i][0]
        if not _is_direction(text):
            # Group header: T, and in Table 3 the fO2 too.
            try:
                t_c = float(text.replace(MINUS, "-"))
            except ValueError:
                break  # the footnotes: end of the table body
            fo2 = None
            if not has_conv:
                fo2 = float(toks[i + 1][0].replace(MINUS, "-"))
                i += 1
            run = run_of(t_c, fo2)
            i += 1
            continue
        direction, profile = toks[i][0], toks[i + 1][0]
        cells = toks[i + 2 : i + 10]
        i += 10
        conv = None
        if has_conv and toks[i][0] not in TECH:
            conv = toks[i]
            i += 1
        techs = [toks[i + k][0] for k in range(3)]
        i += 3
        t_c, fo2, dqfm, hours = RUNS[run]
        rec = {
            "source": "oeser2026",
            "table": table,
            "page": "56" if table == "Table 2" else "57",
            "run": run,
            "profile": int(profile),
            "species": "Fe-Mg",
            "axis": _axis(direction),
            "direction_as_printed": direction.replace("~", "approx. "),
            "T_C": t_c,
            "P_Pa": 1e5,
            "log10_fO2_as_printed": fo2,
            "fO2_unit_as_printed": "Pa",
            "log10_fO2_Pa": fo2,
            "delta_QFM": dqfm,
            "duration_h": hours,
            "X_Fo": X_FO,
        }
        for name, (text, italic, mark) in zip(COLUMNS + COLUMNS_D, cells, strict=True):
            rec[name] = _value(text)
            rec[f"{name}_as_printed"] = text.replace(MINUS, "-") + mark
            if name in COLUMNS_D:
                rec[f"{name}_italic"] = italic
        rec["log10_D_m2s"] = rec["log10_D_FeMg"]
        rec["D_as_printed"] = rec["log10_D_FeMg_as_printed"]
        if conv is not None:
            rec["log10_D_FeMg_conv"] = _value(conv[0])
            rec["log10_D_FeMg_conv_italic"] = conv[1]
        techs = ["" if t == MINUS else t for t in techs]  # a minus sign: empty cell
        rec["technique_Fe_isotopes"], rec["technique_Mg_isotopes"], rec["technique_Fo"] = techs
        rows.append(rec)
    return rows


def main() -> None:
    doc = pymupdf.open(sys.argv[1] if len(sys.argv) > 1 else PDF)

    def run_t2(t_c: float, _fo2) -> int:  # Table 2: fO2 about 1e-5 Pa
        return next(r for r, v in RUNS.items() if v[0] == t_c and -5.1 <= v[1] <= -4.6)

    def run_t3(t_c: float, fo2: float) -> int:
        return next(r for r, v in RUNS.items() if v[0] == t_c and v[1] == fo2)

    foot = "represents the chemical gradient"  # the footnote that ends each table
    t2 = parse(tokens(doc[8], "Table 2", foot), "Table 2", run_t2, has_conv=True)
    t3 = parse(tokens(doc[9], "Table 3", foot), "Table 3", run_t3, has_conv=False)
    df = pd.DataFrame(t2 + t3)
    # Table 2's header says fO2 about 1e-5 Pa: each row takes its run's value (Table 1).
    df["log10_fO2_Pa"] = [RUNS[r][1] for r in df["run"]]
    df["notes"] = (
        "D_FeMg from the multicomponent model (Table column 'log10(DFe-Mg)'), for X_Fe = "
        "0.085; *_italic = convolution-corrected (LA linescans); T and fO2 from Table 1"
    )
    df.to_csv(OUT, index=False)
    show = ["table", "run", "T_C", "log10_fO2_Pa", "axis", "profile", "log10_D_FeMg"]
    show += ["log10_D_FeMg_italic", "log10_Dstar_Fe", "log10_Dstar_Mg"]
    print(df[show].to_string(index=False))
    print(f"{len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
