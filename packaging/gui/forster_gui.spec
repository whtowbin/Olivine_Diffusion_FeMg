# PyInstaller spec for the FORSTER desktop app (forster.gui). Built by CI only
# (.github/workflows/gui-installers.yml, through bundle.py):
#   pyinstaller --noconfirm forster_gui.spec
# Output: dist/forster-gui/ (Windows, Linux) or dist/FORSTER.app (macOS).
# Packaging lessons (docs/plans/2026-10-15-gui.md): forster/data and forster's metadata
# must be added by hand, scipy's Sobol direction numbers too, and the matplotlib font
# cache needs a fixed folder (run_gui.frozen_setup).
import sys

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = (
    # forster reads its version (importlib.metadata) and its example tables. Only data/:
    # a bare collect_data_files("forster") also copies Numba's cache files from
    # src/forster/__pycache__ in a development checkout.
    collect_data_files("forster", includes=["data/*"])
    + copy_metadata("forster")
    # Missed by PyInstaller's scipy hook; without it scipy.stats.qmc.Sobol (used by the
    # starting-profile sampler) prints a FileNotFoundError and carries on.
    + collect_data_files("scipy.stats", includes=["*.npz"])
)

a = Analysis(
    ["run_gui.py"],
    pathex=[SPECPATH],  # noqa: F821 (defined by PyInstaller)
    datas=datas,
    hiddenimports=[
        # Imported inside a try block in forster.compiled, or only when the app starts.
        "forster._kernels",
        "forster.gui.app",
        "forster.gui.selftest",
        # matplotlib loads a backend by name when a figure is saved, so PyInstaller does
        # not see it: PDF export (the app's own) and the plot toolbar's SVG and EPS/PS.
        # Missing, the bundle failed its first self-test on PDF export (2026-10-20).
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_svg",
        "matplotlib.backends.backend_ps",
    ],
    excludes=[
        # Present in a development environment, never used by the app.
        "tkinter",
        "IPython",
        "jupyter_client",
        "ipykernel",
        "marimo",
        "pytest",
        "seaborn",
        "sklearn",
        "selenium",
        "great_tables",
        "PyQt5",
        "PyQt6",
        # Kriging is only for the legacy uncertainty method, which the app refuses
        # (forster.gui.state.app_refusal): the app never loads pykrige.
        "pykrige",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="forster-gui",
    console=False,  # a windowed app; stdout goes to a log file when there is no console
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="forster-gui", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="FORSTER.app",
        bundle_identifier="org.forster.gui",
        info_plist={"NSHighResolutionCapable": True},
    )
