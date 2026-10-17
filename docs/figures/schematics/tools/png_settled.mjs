// Write a figure's PNG with the schematic-figure skill's own page loader, but wait for the
// page to settle first. The skill's render.mjs screenshots straight after resizing the
// viewport, and under Brave (and possibly other Chromium builds) that capture comes out
// tiled: the top-left of the page repeated across the image. PDF and SVG are unaffected.
// Usage: node png_settled.mjs <figure-folder> <dpi> <out.png>
// The skill's foundation is found at $SCHEMATIC_FOUNDATION, else ~/.claude/skills/schematic-figure/foundation.
import { writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const foundation = process.env.SCHEMATIC_FOUNDATION || join(homedir(), '.claude/skills/schematic-figure/foundation');
const { serve, launch, loadFigure, screenshot } = await import(pathToFileURL(join(foundation, 'tools/lib/cdp.mjs')).href);

const [root, dpi, outPng] = process.argv.slice(2);
if (!root || !dpi || !outPng) throw new Error('usage: node png_settled.mjs <figure-folder> <dpi> <out.png>');
const { srv, port } = await serve(foundation, resolve(root));
const cdp = await launch();
try {
  const { error } = await loadFigure(cdp, port, 'figure/figure.js', { dpi: +dpi });
  if (error) throw new Error(error);
  await cdp.evaluate('new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))');
  await new Promise((r) => setTimeout(r, 400));
  writeFileSync(outPng, await screenshot(cdp));
  console.log('wrote', outPng);
} finally {
  cdp.close();
  srv.close();
}
