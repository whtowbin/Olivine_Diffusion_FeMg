# PyInstaller spec for the FORSTER Qt prototype. Build from this folder:
#   pyinstaller --noconfirm forster_qt.spec
# Output: dist/forster-qt/ (Windows, Linux) or dist/ForsterQt.app (macOS).
import sys

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = (
    # forster reads its version (importlib.metadata) and its example tables. Only data/:
    # a bare collect_data_files("forster") also copies Numba's cache files from
    # src/forster/__pycache__ in a development checkout.
    collect_data_files("forster", includes=["data/*"])
    + copy_metadata("forster")
    # Missed by PyInstaller's scipy hook; without it scipy.stats.qmc.Sobol (used by the
    # initial-condition sampler) prints a FileNotFoundError and carries on.
    + collect_data_files("scipy.stats", includes=["*.npz"])
)

a = Analysis(
    ["run_gui.py"],
    pathex=[SPECPATH],  # noqa: F821 (defined by PyInstaller)
    datas=datas,
    # Imported inside a try block in forster.compiled; listed so it is never missed.
    hiddenimports=["forster._kernels"],
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
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="forster-qt",
    console=False,  # a windowed app; stdout goes to a log file when there is no console
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="forster-qt", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ForsterQt.app",
        bundle_identifier="org.forster.gui-qt",
        info_plist={"NSHighResolutionCapable": True},
    )
