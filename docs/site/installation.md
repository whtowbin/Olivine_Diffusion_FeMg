# Installation

FORSTER is a Python package. It needs **Python 3.11 or newer**; it is tested on 3.11
and 3.13 and declared for 3.11-3.13. (Python 3.14 works in the developers'
measurements; 3.15 pre-releases do not, because SciPy and pandas do not yet run on
them.)

FORSTER is not yet on PyPI, so for now it installs from GitHub. Once it is released,
`pip install forster` will work and the commands below get shorter.

## With uv (recommended)

[uv](https://docs.astral.sh/uv/) installs Python itself if you do not have it, and
keeps FORSTER in its own environment so it cannot clash with other packages.

```bash
# the `forster` command, available everywhere
uv tool install git+https://github.com/whtowbin/Olivine_Diffusion_FeMg

# or as a library inside a uv project (for scripts and notebooks)
uv add git+https://github.com/whtowbin/Olivine_Diffusion_FeMg
```

## With pip

```bash
pip install git+https://github.com/whtowbin/Olivine_Diffusion_FeMg
```

Use a virtual environment (`python -m venv .venv`, then activate it) rather than your
system Python.

## Optional extras

Extras are optional groups of dependencies, named in square brackets.

| Extra | Adds | Why you might want it |
|---|---|---|
| `gui` | PySide6-Essentials (Qt) | The desktop app, `forster gui`. |
| `fast` | [Numba](https://numba.pydata.org/) | Compiles the model's inner loops: fits and starting-profile fits run about 5× faster. Results are the same to rounding. |

```bash
uv tool install "forster[gui,fast] @ git+https://github.com/whtowbin/Olivine_Diffusion_FeMg"
# or
pip install "forster[gui,fast] @ git+https://github.com/whtowbin/Olivine_Diffusion_FeMg"
```

Without Numba the same code runs in NumPy. Setting the environment variable
`FORSTER_NO_NUMBA=1` switches Numba off even when it is installed (useful to check
that a result does not depend on it).

## Check the installation

```bash
forster --version
forster template my_first_fit/
forster fit my_first_fit/profiles.csv my_first_fit/parameters.csv --out my_first_fit/results
```

The last command should print `5.14 days (4.24-6.21)` for the worked example. The
[Quick start](quickstart.md) explains what it wrote.

## Standalone installers

Standalone versions of the desktop app for macOS, Windows and Linux, which need no
Python, are built automatically on GitHub (each build runs the app's own self-test:
a fit, a starting-profile fit, a Monte Carlo, the law comparison, exports, save and
reopen). They have **not been released yet**. When they are, they will be on the
project's GitHub releases page. They are not signed, so macOS and Windows will warn
the first time you open them.

## For developers

```bash
git clone https://github.com/whtowbin/Olivine_Diffusion_FeMg.git
cd Olivine_Diffusion_FeMg
uv sync
uv run pytest -n 4
```

Some tests use profiles from an interlaboratory comparison (a "round robin") that
belong to other laboratories and are kept in a private git submodule until that study
is published. Without access those tests are skipped and the rest run.

This site is built with MkDocs from `docs/site/`:

```bash
uv run --group docs mkdocs serve      # preview at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict
```
