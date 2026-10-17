# FORSTER

**F**itting **O**livine **R**esidence & **S**torage **T**imescales from **E**lemental
**R**elaxation.

[![Tests](https://github.com/whtowbin/Olivine_Diffusion_FeMg/actions/workflows/tests.yml/badge.svg)](https://github.com/whtowbin/Olivine_Diffusion_FeMg/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-forster.readthedocs.io-blue)](https://forster.readthedocs.io)

FORSTER fits Fe-Mg diffusion times to forsterite (Fo) profiles measured across zoned
olivine. Sharp growth zoning is smoothed by Fe-Mg exchange while a crystal sits in hot
magma; the wider the smoothed step, the longer it sat. Give FORSTER the measured
profile, the temperature, pressure and oxygen fugacity, the profile's direction in the
crystal and a step-shaped starting profile; it reports the best-fit time with 1σ and
2σ ranges, can fit the starting profile itself, and propagates the uncertainty of the
conditions and of the diffusion law into the time.

![The FORSTER desktop app with a fitted profile](https://raw.githubusercontent.com/whtowbin/Olivine_Diffusion_FeMg/master/docs/site/images/app-monte-carlo.png)

**Full documentation: [forster.readthedocs.io](https://forster.readthedocs.io)** (a
user guide with the method, every table column and the Python API).

## Install

Python 3.11 or newer. Until the first release on PyPI, install from GitHub:

```bash
uv tool install "forster[gui,fast] @ git+https://github.com/whtowbin/Olivine_Diffusion_FeMg"
# or
pip install "forster[gui,fast] @ git+https://github.com/whtowbin/Olivine_Diffusion_FeMg"
```

`gui` adds the desktop app (Qt), `fast` adds Numba (fits about 5x faster). After the
release: `pip install "forster[gui,fast]"`.

## Use it

**Command line**, with two filled-in tables:

```bash
forster template my_fit/          # profiles.csv and parameters.csv with a worked example
forster fit my_fit/profiles.csv my_fit/parameters.csv --out my_fit/results
```

The worked example (a published melt-vein olivine profile) prints `5.14 days
(4.24-6.21)` and writes a summary table and a figure for each run:

![The figure forster fit writes for the worked example](https://raw.githubusercontent.com/whtowbin/Olivine_Diffusion_FeMg/master/docs/site/images/quickstart-fit.png)

**Your own spreadsheets**: `forster import data.xlsx` proposes a column mapping
(profile, distance, Fo or MgO and FeO, totals) for you to check, then writes the
profiles table.

**Desktop app**: `forster gui` opens a project folder: draw the starting profile with
the mouse, fit, fit the starting profile, run the Monte Carlo, and export.

**Python**:

```python
import forster

runs = forster.read_runs("my_fit/parameters.csv")
profiles = forster.read_profiles("my_fit/profiles.csv")
result = forster.fit_run(runs[0], profiles)
print(result.time_days, result.time_range_days())
```

## What it does

- Implicit finite-difference model of Fe-Mg interdiffusion with a diffusivity that
  depends on Fo, temperature, pressure, oxygen fugacity and crystal direction; four
  named diffusion laws (Dohmen and Chakraborty 2007 as in DIPRA, FORSTER's 2026 refit,
  and the two DFENS laws of Mutch et al. 2019).
- Ranges from a noise model that allows for smooth misfit between model and data, with
  checks that warn when the range is probably too narrow (misfit concentrated near a
  step, a profile that ends before its plateau).
- Starting profiles of any number of segments, fitted with their uncertainties by
  optimisation and importance sampling; growth stages; a Monte Carlo over the
  conditions and the law; joint fits of several profiles of one crystal (Python).

## Development

```bash
git clone https://github.com/whtowbin/Olivine_Diffusion_FeMg.git
cd Olivine_Diffusion_FeMg
uv sync
uv run pytest -n 4
uv run --group docs mkdocs serve      # the documentation site, locally
```

Some tests use the olivine diffusion round robin profiles, other laboratories'
samples kept in a private submodule (`Diffusion Round Robin`); without access those
tests skip. Changes are listed in [CHANGELOG.md](CHANGELOG.md); releases follow
[docs/releasing.md](docs/releasing.md).

## Citing

Towbin, W. H., and Plank, T. (2026). Journal of Petrology 67(9), egag077. The
citations for each diffusion law and method are in the documentation.

## License

MIT ([LICENSE](LICENSE)).
