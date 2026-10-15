"""CI checks of a built bundle (any OS): self-test with wall time and size, cold start.

    python prototypes/gui_web/ci_check.py selftest  OUT_DIR
    python prototypes/gui_web/ci_check.py coldstart OUT_DIR   (needs a display: xvfb-run)

Finds the bundle in prototypes/gui_web/dist, runs it, prints what it printed, writes
OUT_DIR/metrics-<check>.json and a line to the GitHub step summary. Standard library only.
The self-test's exit code is this script's; a cold start that never shows a window
exits 1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "FORSTER-web"


def bundle() -> tuple[Path, Path]:
    dist = HERE / "dist"
    if sys.platform == "darwin":
        app = dist / f"{NAME}.app"
        return app, app / "Contents" / "MacOS" / NAME
    folder = dist / NAME
    return folder, folder / (f"{NAME}.exe" if sys.platform == "win32" else NAME)


def size_mb(path: Path) -> float:
    return (
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 1e6
    )


def summary(line: str) -> None:
    print(line)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n\n")


def selftest(out: Path) -> int:
    folder, exe = bundle()
    tmp = out / "selftest"
    start = time.perf_counter()
    done = subprocess.run(
        [str(exe), "--self-test", str(tmp)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    wall = time.perf_counter() - start
    log = tmp / "selftest.log"
    print(done.stdout or (log.read_text(encoding="utf-8") if log.exists() else "(no output)"))
    print(done.stderr, file=sys.stderr)
    metrics = {
        "os": sys.platform,
        "bundle_mb": round(size_mb(folder), 1),
        "selftest_wall_s": round(wall, 2),
        "selftest_exit": done.returncode,
    }
    (out / "metrics-selftest.json").write_text(json.dumps(metrics, indent=1))
    summary(
        f"**{sys.platform}**: bundle {metrics['bundle_mb']} MB; self-test exit "
        f"{done.returncode} in {wall:.1f} s wall"
    )
    return done.returncode


def coldstart(out: Path, runs: int = 3) -> int:
    _, exe = bundle()
    metrics_file = out / "coldstart.jsonl"
    metrics_file.unlink(missing_ok=True)
    times = []
    for _ in range(runs):
        env = {**os.environ, "FORSTER_GUI_T0": repr(time.time())}
        env["FORSTER_GUI_METRICS"] = str(metrics_file)
        try:
            subprocess.run([str(exe), "--exit-when-ready"], env=env, timeout=120)
        except subprocess.TimeoutExpired:
            print("no window within 120 s")
    if metrics_file.exists():
        for line in metrics_file.read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") == "ready":
                times.append(record["seconds"])
    (out / "metrics-coldstart.json").write_text(json.dumps({"ready_s": times}, indent=1))
    summary(f"**{sys.platform}**: cold start to the first window {times} s")
    return 0 if times else 1


if __name__ == "__main__":
    check, out_dir = sys.argv[1], Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.exit({"selftest": selftest, "coldstart": coldstart}[check](out_dir))
