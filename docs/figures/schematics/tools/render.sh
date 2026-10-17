#!/bin/sh
# Render one FORSTER schematic: vector PDF and SVG from the schematic-figure skill's
# render.mjs, the PNG from png_settled.mjs (render.mjs's own PNG can come out tiled),
# then the tiling check and the skill's gate.
#
# Usage, from anywhere:  docs/figures/schematics/tools/render.sh <figure> [dpi]
#   e.g.  docs/figures/schematics/tools/render.sh workflow        (600 dpi)
#         docs/figures/schematics/tools/render.sh workflow 150    (quick look)
# Needs Node >= 22, uv, and a Chromium browser (CHROME, else Chrome, Chromium or Brave).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
FIG=$(cd "$HERE/../$1" && pwd)
DPI=${2:-600}
FOUNDATION=${SCHEMATIC_FOUNDATION:-$HOME/.claude/skills/schematic-figure/foundation}
if [ -z "${CHROME:-}" ]; then
  for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
           "/Applications/Chromium.app/Contents/MacOS/Chromium" \
           "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"; do
    if [ -x "$c" ]; then CHROME=$c; break; fi
  done
fi
export CHROME SCHEMATIC_FOUNDATION="$FOUNDATION"

cd "$FIG"
node "$FOUNDATION/tools/render.mjs" --root "$FIG" --fig "$FIG/figure/figure.js"
set -- figure/out/*.pdf
[ "$#" -eq 1 ] || { echo "expected one PDF in figure/out, found: $*" >&2; exit 1; }
PDF=$1
PNG=${PDF%.pdf}.png
node "$HERE/png_settled.mjs" "$FIG" "$DPI" "$PNG"
uv run -q --no-project --with pillow --with numpy python "$HERE/tiled_check.py" "$PNG"
node "$FOUNDATION/tools/check.mjs" --root "$FIG" --fig "$FIG/figure/figure.js" --final --out "$FIG/figure/out"
