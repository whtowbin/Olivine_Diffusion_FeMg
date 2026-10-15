"""Drive the browser version with real mouse drags: latency, scan, cancel, screenshots.

Start the app first (`main.py --browser --no-show --port 8765 --open <project>`, with
FORSTER_GUI_METRICS=<file> for the server-side records), then:

    uv run --with playwright python prototypes/gui_web/measure_browser.py \
        --url http://127.0.0.1:8765 --shots prototypes/gui_web/screenshots --drags 20

Any installed Chromium works (--browser-path); Playwright's own needs
`playwright install chromium`.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BRAVE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"

HANDLE_XY = """([index, axis]) => {
  const gd = document.querySelector('.js-plotly-plot');
  const L = gd._fullLayout, s = L.shapes[index], box = gd.getBoundingClientRect();
  const xpx = v => box.left + L._size.l + L.xaxis.l2p(v);
  const ypx = v => box.top + L._size.t + L.yaxis.l2p(v);
  if (axis === 'x') return [xpx(s.x0), box.top + L._size.t + L._size.h / 2];
  return [(xpx(s.x0) + xpx(s.x1)) / 2, ypx(s.y0)];
}"""


def wait_fresh(page: Page, timeout: float = 30_000) -> None:
    page.wait_for_function(
        "() => { const gd = document.querySelector('.js-plotly-plot');"
        " return gd && gd.layout && gd.layout.meta && gd.layout.meta.fresh; }",
        timeout=timeout,
    )


def drag(page: Page, index: int, axis: str, pixels: float) -> float:
    """Drag a handle by ``pixels``; returns the client-side release-to-redraw ms."""
    before = page.evaluate("(window.__fg.dragMs || []).length")
    x, y = page.evaluate(HANDLE_XY, [index, axis])
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 11):
        dx, dy = (pixels * i / 10, 0) if axis == "x" else (0, pixels * i / 10)
        page.mouse.move(x + dx, y + dy)
    page.mouse.up()
    page.wait_for_function(f"(window.__fg.dragMs || []).length > {before}", timeout=30_000)
    return page.evaluate("window.__fg.dragMs[window.__fg.dragMs.length - 1]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--shots", type=Path, default=Path("prototypes/gui_web/screenshots"))
    parser.add_argument("--drags", type=int, default=20)
    parser.add_argument("--browser-path", default=BRAVE)
    parser.add_argument("--mode", choices=["drag", "scan", "cancel"], default="drag")
    args = parser.parse_args()
    args.shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser_path or None, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 880})
        page.goto(args.url)
        wait_fresh(page)
        page.wait_for_timeout(500)
        out: dict = {}
        if args.mode == "drag":
            page.screenshot(path=args.shots / "browser-1-opened.png")
            times = []
            for i in range(args.drags):
                sign = 1 if i % 2 == 0 else -1
                index, axis, px = [(0, "x", 25 * sign), (1, "y", -12 * sign)][i // 2 % 2]
                times.append(drag(page, index, axis, px))
            page.wait_for_timeout(300)
            page.screenshot(path=args.shots / "browser-2-after-drags.png")
            out["client_drag_ms"] = {
                "n": len(times),
                "median": round(statistics.median(times), 1),
                "min": round(min(times), 1),
                "max": round(max(times), 1),
                "all": [round(t, 1) for t in times],
            }
            edge = page.get_by_label("Edge Fo (fraction)")
            edge.fill("1.5")
            edge.press("Enter")
            page.wait_for_timeout(600)
            page.screenshot(path=args.shots / "browser-3-refused-edit.png")
            out["error_shown"] = page.get_by_text("Fo values must be fractions").count() > 0
        elif args.mode == "scan":
            page.get_by_role("button", name="Fit initial condition").click()
            page.wait_for_timeout(1500)
            page.screenshot(path=args.shots / "browser-4-scan-running.png")
            page.get_by_text("best:").wait_for(timeout=120_000)
            wait_fresh(page)
            page.wait_for_timeout(500)
            page.screenshot(path=args.shots / "browser-5-scan-done.png")
            out["scan_text"] = page.locator(".whitespace-pre-line").inner_text()
        else:
            step = page.get_by_label("Step position (µm)").input_value()
            page.get_by_role("button", name="Fit initial condition").click()
            page.wait_for_timeout(500)
            page.get_by_role("button", name="Cancel").click()
            page.wait_for_timeout(8000)  # longer than the scan: its result must be dropped
            page.screenshot(path=args.shots / "browser-6-cancelled.png")
            after = page.get_by_label("Step position (µm)").input_value()
            out["cancel"] = {"step_before": step, "step_after": after, "unchanged": step == after}
        print(json.dumps(out, indent=1))
        browser.close()


if __name__ == "__main__":
    main()
