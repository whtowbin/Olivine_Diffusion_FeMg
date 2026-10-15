"""Build the prototype into a desktop bundle with PyInstaller (what `nicegui-pack` wraps).

    uv run --with-requirements prototypes/gui_web/requirements.txt \
        python prototypes/gui_web/build.py

Output: prototypes/gui_web/dist/FORSTER-web.app on macOS, dist/FORSTER-web/ elsewhere
(one folder: starts faster than one file). Prints the bundle size and the executable's
path. Every PyInstaller option beyond nicegui-pack's own is listed in EXTRA with why.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import forster
import nicegui
import PyInstaller.__main__

HERE = Path(__file__).resolve().parent
NAME = "FORSTER-web"
SEP = os.pathsep

EXTRA = [
    # nicegui-pack's one data option: NiceGUI's static files (Vue, Quasar, Plotly ESM).
    "--add-data",
    f"{Path(nicegui.__file__).parent}{SEP}nicegui",
    # forster's packaged example tables (forster.template reads them with importlib).
    # Not --collect-data forster: that also sweeps the developer's Numba cache files
    # (__pycache__/*.nbi, *.nbc) into the bundle. A frozen app caches compiled kernels
    # in Numba's user-wide folder (~/Library/Caches/numba on macOS) by itself.
    "--add-data",
    f"{Path(forster.__file__).parent / 'data'}{SEP}forster/data",
    # forster.version() reads the installed distribution's metadata.
    "--copy-metadata",
    "forster",
    # pykrige and dynesty import submodules lazily.
    "--collect-submodules",
    "pykrige",
    "--collect-submodules",
    "dynesty",
    # uvicorn picks its loop, protocol and lifespan implementations by name at run time.
    "--collect-submodules",
    "uvicorn",
    # Not used; keeps the bundle smaller.
    "--exclude-module",
    "tkinter",
    "--exclude-module",
    "IPython",
    "--exclude-module",
    "marimo",
    "--exclude-module",
    "sklearn",
    "--exclude-module",
    "seaborn",
    "--exclude-module",
    "selenium",
]


def size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files if not f.is_symlink()) / 1e6


def main() -> int:
    dist, work = HERE / "dist", HERE / "build"
    shutil.rmtree(dist, ignore_errors=True)
    args = [
        str(HERE / "main.py"),
        "--name",
        NAME,
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",  # a .app on macOS, no console window on Windows
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        "--specpath",
        str(work),
        "--paths",
        str(HERE),
        "--osx-bundle-identifier",
        "org.forster.gui-web-prototype",
        *EXTRA,
    ]
    PyInstaller.__main__.run(args)
    bundle = dist / f"{NAME}.app" if sys.platform == "darwin" else dist / NAME
    exe = (
        bundle / "Contents" / "MacOS" / NAME
        if sys.platform == "darwin"
        else bundle / (f"{NAME}.exe" if sys.platform == "win32" else NAME)
    )
    print(f"bundle {bundle} {size_mb(bundle):.0f} MB")
    print(f"executable {exe}")
    return 0 if exe.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
