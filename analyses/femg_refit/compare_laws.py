"""How far the published Fe-Mg laws sit from the compiled data, and which terms are missing.

Residuals of the Dohmen and Chakraborty (2007) and Mutch et al. (2019) laws, and of
our refits, against the 135 TaMED points (by study); then F tests of extra terms
added to the Mutch form with fitted anisotropy. Run:
uv run --with odrpack python analyses/femg_refit/compare_laws.py
"""

import math
import warnings

import fit_models as F
import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
df = F.tamed()
LN10 = math.log(10)
obs = df["log10_D_m2s"].to_numpy()


def preset_log_d(name):
    axes = {"[100]": 0, "[010]": 1, "[001]": 2}
    return np.array(
        [
            F.preset_log10_d(name, r.T_C, r.P_Pa, r.log10_fO2_Pa, r.X_Fo, axes[r.axis])
            for r in df.itertuples()
        ]
    )


print("Residuals of each law against the 135 TaMED points (observed - law, log10):")
rows = {}
for name in ("dohmen_chakraborty2007", "mutch2019_tamed", "mutch2019_global"):
    r = obs - preset_log_d(name)
    rows[name] = r
v = F.Vars.of(df)
for fname in ("dc07_fixed", "dc07_free", "mutch_aniso"):
    f = F.FORMS[fname]
    beta = F.fit_ls(f, df, weighted=False)["beta"]
    rows[f"refit {fname}"] = (df.ln_D.to_numpy() - f.offset(v) - f.design(v) @ beta) / LN10
for name, r in rows.items():
    by = pd.Series(r).groupby(df.source.to_numpy()).mean()
    print(
        f"  {name:24s} mean {r.mean():+.3f} sd {r.std(ddof=1):.3f} "
        f"rms {np.sqrt((r**2).mean()):.3f} | by study: "
        + ", ".join(f"{k[:6]} {x:+.2f}" for k, x in by.items())
    )

# Where do the laws diverge? Refit (dc07_free) minus the published DC07 at the data's points.
dd = rows["dohmen_chakraborty2007"] - rows["refit dc07_free"]
print(
    "\nRefit (DC07 form, V fixed at 7 cm3/mol) minus published DC07 at the data's own conditions: "
    f"mean {dd.mean():+.3f}, range {dd.min():+.2f} to {dd.max():+.2f} log"
)

# Missing terms: add to the Mutch form (anisotropy fitted) and test by F.
base = F.FORMS["mutch_aniso"]
X0 = base.design(v)
y = df.ln_D.to_numpy() - base.offset(v)
lnf = (v.log_fo2_pa - 5) * LN10
cands = {
    "fO2 x 1/T": lnf / v.T,
    "X_Fo x 1/T": v.x_fo / v.T,
    "(ln fO2)^2": lnf**2,
    "X_Fo^2": v.x_fo**2,
    "study offsets": None,
}


def rss(X):
    s = np.abs(X).max(axis=0)
    b, *_ = np.linalg.lstsq(X / s, y, rcond=None)
    r = y - (X / s) @ b
    return float(r @ r), X.shape[1]


r0, p0 = rss(X0)
print(
    f"\nBase: Mutch form, anisotropy fitted: rss {r0:.2f}, p {p0}, n {len(y)}, "
    f"sd {math.sqrt(r0 / (len(y) - p0)) / LN10:.3f} log"
)
for name, col in cands.items():
    if col is None:
        dummies = pd.get_dummies(df.source, drop_first=True).to_numpy(float)
        X1 = np.column_stack([X0, dummies])
    else:
        X1 = np.column_stack([X0, col])
    r1, p1 = rss(X1)
    k = p1 - p0
    Fst = ((r0 - r1) / k) / (r1 / (len(y) - p1))
    pv = stats.f.sf(Fst, k, len(y) - p1)
    print(
        f"  + {name:14s}: sd {math.sqrt(r1 / (len(y) - p1)) / LN10:.3f} log, "
        f"F = {Fst:6.2f}, p = {pv:.2g}"
    )

# The unpublished Meissner (2000) points, read from Dohmen and Chakraborty (2007)
# Part II Fig. 7b: do they move the refits? Added at fO2 1e-7 Pa, [001], 1 atm, as
# plotted, with either Fo90 (the caption) or Fo86 (how Chakraborty's are plotted).
fig7b = pd.read_csv(F.Path(__file__).parent / "data" / "dohmen_chakraborty2007_fig7b_extracted.csv")
mei = fig7b[fig7b["series"].str.contains("Mei", case=False)]
ks20 = F._conditions()["KS20, 1200 C"]
print(f"\nMeissner (2000), {len(mei)} points from DC07 Fig. 7b: KS20 vs default, without -> with")
for x_fo in (0.90, 0.86):
    add = pd.DataFrame(
        {
            "source": "meissner2000_fig7b",
            "run": [f"M{i}" for i in range(len(mei))],
            "axis": "[001]",
            "T_C": mei["T_C"].to_numpy(),
            "T_K": mei["T_C"].to_numpy() + 273.15,
            "P_Pa": 1e5,
            "log10_fO2_Pa": -7.0,
            "X_Fo": x_fo,
            "log10_D_m2s": mei["log10_D_normalised"].to_numpy(),
        }
    )
    add["ln_D"] = add["log10_D_m2s"] * LN10
    both = pd.concat([df, add], ignore_index=True)
    for name in ("dc07_fixed", "dc07_free", "mutch_aniso"):
        f = F.FORMS[name]
        shifts = []
        for data in (df, both):
            fit = F.fit_ls(f, data, weighted=False)
            shifts.append(F.predict(f, fit["beta"], fit["cov"], ks20)[0] - F._dc07_preset(ks20))
        print(f"  Fo{x_fo * 100:.0f} {name:12s}: {shifts[0]:+.3f} -> {shifts[1]:+.3f}")
