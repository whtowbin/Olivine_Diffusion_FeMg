"""Build the PyInstaller bundle and test it; the same commands locally and in CI.

    python bundle.py build             # dist/ForsterQt.app (macOS) or dist/forster-qt/
    python bundle.py test [--out J]    # the bundle's self-test and cold starts, headless
    python bundle.py archive           # pack it for upload (zip, or .tar.gz on Linux)

``test`` prints (and writes to ``--out`` and the GitHub step summary) the bundle's size,
the self-test's wall time and exit code, and the wall time of three cold starts to the
first painted window (``--startup-probe``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST, WORK = HERE / "dist", HERE / "build"


def bundle_path() -> Path:
    return DIST / "ForsterQt.app" if sys.platform == "darwin" else DIST / "forster-qt"


def executable() -> Path:
    if sys.platform == "darwin":
        return bundle_path() / "Contents" / "MacOS" / "forster-qt"
    return bundle_path() / ("forster-qt.exe" if sys.platform == "win32" else "forster-qt")


def size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            f = Path(root) / name
            if not f.is_symlink():
                total += f.stat().st_size
    return total / 1e6


def build() -> int:
    import PyInstaller.__main__

    spec = str(HERE / "forster_qt.spec")
    PyInstaller.__main__.run(
        [spec, "--noconfirm", "--distpath", str(DIST), "--workpath", str(WORK)]
    )
    print(f"built {bundle_path()} ({size_mb(bundle_path()):.0f} MB)")
    return 0


def run_timed(args: list[str], timeout: float, log_dir: Path | None = None) -> tuple[int, float]:
    env = {"QT_QPA_PLATFORM": "offscreen", **os.environ}
    start = time.perf_counter()
    done = subprocess.run(
        [str(executable()), *args],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    seconds = time.perf_counter() - start
    print(done.stdout, done.stderr, sep="\n")
    log = log_dir / "gui_qt.log" if log_dir else None
    if log is not None and log.exists():  # a windowed Windows build logs here
        print(log.read_text(encoding="utf-8", errors="replace"))
    return done.returncode, seconds


def test(out: Path | None) -> int:
    exe = executable()
    if not exe.exists():
        print(f"no bundle at {exe}; run `python bundle.py build` first")
        return 2
    report: dict = {
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "bundle": bundle_path().name,
        "bundle_mb": round(size_mb(bundle_path()), 1),
    }
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        code, seconds = run_timed(["--self-test", tmp], 900, Path(tmp))
    report["self_test_exit"], report["self_test_s"] = code, round(seconds, 2)
    starts = []
    for _ in range(3):
        start_code, seconds = run_timed(["--startup-probe"], 300)
        starts.append(round(seconds, 2) if start_code == 0 else None)
    report["startup_s"] = starts
    text = json.dumps(report, indent=2)
    print(text)
    if out is not None:
        out.write_text(text + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### FORSTER Qt prototype, {report['os']}\n\n")
            f.write("| what | value |\n|---|---|\n")
            for key, value in report.items():
                f.write(f"| {key} | {value} |\n")
    return code


def archive() -> int:
    """Pack the bundle for upload, keeping symlinks and permissions (an upload of the
    bare folder loses both): a zip made by ditto on macOS, a .tar.gz on Linux, a zip
    on Windows."""
    source = bundle_path()
    tag = {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux")
    if sys.platform == "darwin":
        target = DIST / f"ForsterQt-{tag}.zip"
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(source), str(target)], check=True)
    elif sys.platform == "win32":
        base = str(DIST / f"forster-qt-{tag}")
        target = Path(shutil.make_archive(base, "zip", DIST, source.name))
    else:
        target = DIST / f"forster-qt-{tag}.tar.gz"
        with tarfile.open(target, "w:gz") as tar:
            tar.add(source, arcname=source.name)
    mb = target.stat().st_size / 1e6
    print(f"archive {target} ({mb:.0f} MB)")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"| archive_mb | {mb:.1f} |\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "test", "archive"])
    parser.add_argument("--out", type=Path, help="write the test report (JSON) here")
    args = parser.parse_args()
    if args.command == "build":
        return build()
    return test(args.out) if args.command == "test" else archive()


if __name__ == "__main__":
    sys.exit(main())
