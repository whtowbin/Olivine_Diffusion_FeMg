// Zoned-olivine concept figure for the FORSTER methods paper.
// Hero: zoned olivine section in melt, drawn flat with the illustrated finish (route flat).
export const figure = { width: 183, height: 72, name: 'olivine-concept', format: 'journal', illustrate: true };

const COL = {
  a: '#0072B2',        // a [100] direction key (Wong blue)
  c: '#D55E00',        // c [001] direction key (Wong vermillion)
  b: '#6b6b6b',        // b [010]: neutral
  rimZone: '#EDC874',  // lower-Fo (Fe-richer) rim zone: light amber, bright as in BSE images
  core: '#6E8B3D',     // higher-Fo core: olive (the Fo scale shared by all the schematics)
  melt: '#ece6dc',
  ink: '#2b2b2b',
  mid: '#7a7a7a',
};

// convex polygon inset by d (pts in any units, centroid-based inward normals)
function inset(pts, d) {
  const n = pts.length, cx = pts.reduce((s, p) => s + p[0], 0) / n, cy = pts.reduce((s, p) => s + p[1], 0) / n;
  const lines = pts.map((p, i) => {
    const q = pts[(i + 1) % n], ex = q[0] - p[0], ey = q[1] - p[1], L = Math.hypot(ex, ey);
    let nx = -ey / L, ny = ex / L;
    const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    if ((cx - mx) * nx + (cy - my) * ny < 0) { nx = -nx; ny = -ny; }
    return { p: [p[0] + nx * d, p[1] + ny * d], e: [ex, ey] };
  });
  return lines.map((l1, i) => {
    const l0 = lines[(i - 1 + n) % n];
    const den = l0.e[0] * l1.e[1] - l0.e[1] * l1.e[0];
    const t = ((l1.p[0] - l0.p[0]) * l1.e[1] - (l1.p[1] - l0.p[1]) * l1.e[0]) / den;
    return [l0.p[0] + l0.e[0] * t, l0.p[1] + l0.e[1] * t];
  });
}
// first crossing of the ray o + s*dir (s > 0) with a closed polygon
function rayHit(pts, o, dir) {
  let best = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length], ex = q[0] - p[0], ey = q[1] - p[1];
    const den = dir[0] * ey - dir[1] * ex; if (Math.abs(den) < 1e-9) continue;
    const s = ((p[0] - o[0]) * ey - (p[1] - o[1]) * ex) / den, u = ((p[0] - o[0]) * dir[1] - (p[1] - o[1]) * dir[0]) / den;
    if (s > 0 && u >= 0 && u <= 1) best = Math.min(best, s);
  }
  return [o[0] + dir[0] * best, o[1] + dir[1] * best];
}

// a label in runs, each set after the previous one; a run {s, math: true} uses the maths font (Arial's ∝ is tiny)
function runs(L, p, x, y, parts, o = {}) {
  let cx = x;
  for (const r of parts) {
    const q = typeof r === 'string' ? { s: r } : r;
    const t = L.text(p, cx, y, q.s, { ...o, ...(q.math ? { family: L.STYLE.font.math, size: (o.size ?? 6) * 1.15 } : {}) });
    cx += t.getBBox().width + (q.gap ?? 0.9);
  }
  return cx;
}

export default async function draw(F, L) {
  const m = L.mm;
  const data = await (await fetch('/figure/data/profiles.json')).json();
  const head = (p, str) => L.text(p, m(4.2), m(2.9), str, { size: 7, weight: 'bold', valign: 'baseline' });

  // ---------------------------------------------------------------- panel a: zoned olivine in melt
  const a = F.panel('a', { x: 1, y: 1, w: 61, h: 70 });
  head(a, 'Zoned olivine in melt');
  L.rect(a, m(0.5), m(5.5), m(60), m(64), { r: m(2.5), fill: L.linearGradient(a, [[0, '#f1ece3'], [1, '#e6dfd2']], 90) });
  L.text(a, m(3), m(10), 'melt', { color: COL.mid, italic: true });

  const cx = 29, cy = 35, ZW = 3.8;            // crystal centre and rim-zone width (mm)
  const outerMM = [[-19, -12], [-10, -21], [8, -22], [19, -13], [21, 9], [12, 21], [-8, 22], [-20, 12]].map(([x, y]) => [cx + x, cy + y]);
  const innerMM = inset(outerMM, ZW);
  const toPt = (pts) => pts.map(([x, y]) => [m(x), m(y)]);
  const outer = toPt(outerMM), inner = toPt(innerMM);
  L.finish(a, L.roundedPoly(outer, m(1.6)), COL.rimZone, { contrast: 0.6, shadow: { kind: 'drop', opacity: 0.14 }, highlight: 0.25, rim: false, core: 0.2, width: 0.6 });
  L.finish(a, L.roundedPoly(inner, m(1.2)), COL.core, { contrast: 0.55, highlight: 0.5, rim: false, core: 0.15, outline: L.shade(COL.rimZone, -0.25), width: 0.35 });

  // traverses: from the rim inward; length = ZW * xmax/x0, so each plot in b is this traverse to scale
  const span = ZW * data.xmax / data.x0;
  const trav = (startMM, dirMM, col, name) => {
    const s = rayHit(outerMM, [startMM[0], startMM[1]], [-dirMM[0], -dirMM[1]]);   // rim point behind the start
    const e = [s[0] + dirMM[0] * span, s[1] + dirMM[1] * span];
    L.line(a, [m(s[0]), m(s[1])], [m(e[0]), m(e[1])], { stroke: col, width: 0.5 });
    const n = 9, pts = [];
    for (let i = 0; i < n; i++) { const f = (i + 0.35) / n; pts.push([m(s[0] + dirMM[0] * span * f), m(s[1] + dirMM[1] * span * f)]); }
    L.particles(a, pts, 1.15, col, { seed: name === 'a' ? 4 : 9, vary: { size: 0.08, jitter: 0 }, shadow: false, highlight: 0.5 });
    return { s, e };
  };
  const tA = trav([cx - 5, cy + 4], [1, 0], COL.a, 'a');
  const tC = trav([cx + 5, cy - 6], [0, 1], COL.c, 'c');
  L.text(a, m(tA.e[0] + 1.2), m(tA.e[1]), 'along a [100]', { color: COL.a, valign: 'middle', halo: true, size: 6 });
  L.text(a, m(tC.e[0]), m(tC.e[1] + 2.4), 'along c [001]', { color: COL.c, anchor: 'middle', valign: 'middle', halo: true, size: 6 });

  // labels on the section
  L.text(a, m(cx - 4), m(cy + 13), 'core', { color: L.shade(COL.core, 0.75), anchor: 'middle', valign: 'middle' });
  const rimPt = [m(innerMM[6][0] + (outerMM[6][0] - innerMM[6][0]) * 0.5 + 4), m((innerMM[6][1] + outerMM[6][1]) / 2 + 0.2)];
  L.callout(a, rimPt, [m(3), m(64)], 'rim zone', { anchor: 'start' });
  L.callout(a, [m(tC.s[0]), m(tC.s[1] + 0.4)], [m(41), m(9.6)], 'microprobe spots', { anchor: 'start' });

  // axis triad: a right, c up, b into the page (right-handed: a × b = c)
  const o = [m(52), m(64)], La = m(7.5);
  L.arrow(a, o, [o[0] + La, o[1]], { stroke: COL.a, width: 0.6, head: 2.6 });
  L.arrow(a, o, [o[0], o[1] - La], { stroke: COL.c, width: 0.6, head: 2.6 });
  L.circle(a, o[0], o[1], 1.6, { fill: '#fff', stroke: COL.b, width: 0.5 });
  { const k = 1.6 * 0.62; L.line(a, [o[0] - k, o[1] - k], [o[0] + k, o[1] + k], { stroke: COL.b, width: 0.45 }); L.line(a, [o[0] - k, o[1] + k], [o[0] + k, o[1] - k], { stroke: COL.b, width: 0.45 }); }
  L.text(a, o[0] + La * 0.55, o[1] + m(2.6), 'a [100]', { color: COL.a, anchor: 'middle', valign: 'middle', size: 5.5 });
  L.text(a, o[0] - m(1.2), o[1] - La - m(0.4), 'c [001]', { color: COL.c, anchor: 'end', valign: 'middle', size: 5.5 });
  L.text(a, o[0] - m(1.6), o[1] + m(2.6), 'b [010]', { color: COL.b, anchor: 'end', valign: 'middle', size: 5.5 });

  // ---------------------------------------------------------------- panel b: the step relaxes with time
  const b = F.panel('b', { x: 65, y: 1, w: 80, h: 70 });
  head(b, 'Step relaxes with time');
  const PW = m(33), PH = m(44), PY = m(12.5);
  const xs = data.x;
  const plot = (x0, axis, title, col, opts) => {
    const g = L.group(b, { x: x0, y: PY });
    const X = (x) => (x / data.xmax) * PW, Y = (u) => PH - m(5) - u * (PH - m(12));
    L.text(g, 0, -m(2.2), title, { color: col, size: 6, weight: 'bold' });
    // axes
    L.line(g, [0, 0], [0, PH], { stroke: COL.ink, width: 0.4 });
    L.line(g, [0, PH], [PW, PH], { stroke: COL.ink, width: 0.4 });
    // t0: the starting step
    const xs0 = X(data.x0);
    L.polyline(g, [[0, Y(0)], [xs0, Y(0)], [xs0, Y(1)], [PW, Y(1)]], { stroke: COL.ink, width: 0.5, dash: '1.6 1.1' });
    // later times: sequential ramp of the direction's hue
    const shades = [0.62, 0.32, 0];
    data.curves[axis].forEach((u, k) => {
      const pts = xs.map((x, i) => [X(x), Y(u[i])]);
      L.polyline(g, pts, { stroke: L.shade(col, shades[k]), width: k === 2 ? 0.9 : 0.7 });
    });
    // width bracket at t3: between u = 0.16 and 0.84
    const u3 = data.curves[axis][2];
    const cross = (lev) => { for (let i = 1; i < xs.length; i++) if (u3[i - 1] < lev && u3[i] >= lev) { const f = (lev - u3[i - 1]) / (u3[i] - u3[i - 1]); return xs[i - 1] + f * (xs[i] - xs[i - 1]); } return null; };
    const xl = X(cross(0.16)), xr = X(cross(0.84)), yb = Y(1) - m(2.6);
    L.bracket(g, [xl, yb], [xr, yb], { side: 'bottom', depth: m(1), stroke: COL.ink, width: 0.4 });
    runs(L, g, xr + m(1), yb + m(0.4), opts.bracket, { size: 5.5, valign: 'middle' });
    // time arrow across the curves, high side
    const ya = Y(0.72), xEnd = X(cross(0.72)) + m(2.2);
    L.arrow(g, [xs0 + m(0.5), ya], [xEnd, ya], { stroke: COL.ink, width: 0.4, head: 2.2 });
    L.text(g, xEnd + m(0.8), ya, 'time', { size: 5.5, color: COL.ink, valign: 'middle', halo: true });
    L.text(g, xs0 + m(0.8), Y(0) - m(0.3), 't_{0}', { size: 5.5, valign: 'bottom', italic: true });
    if (opts.plateaus) {
      L.text(g, m(1), Y(0) + m(1.6), 'rim zone', { size: 5.5, color: COL.mid, valign: 'top' });
      L.text(g, PW - m(1), Y(1) + m(1.4), 'core', { size: 5.5, color: COL.mid, anchor: 'end', valign: 'top' });
    }
    return { g, X, Y };
  };
  plot(m(6), 'a', 'along a [100]: slow', COL.a, { bracket: ['width ~ √(*D t*)'], plateaus: true });
  plot(m(44), 'c', 'along c [001]: ~6× faster', COL.c, { bracket: ['√6 × wider'] });
  L.text(b, m(3), PY + PH / 2, 'Fo', { rotate: -90, anchor: 'middle', valign: 'middle', size: 6 });
  L.text(b, m(6) + (m(38) + PW) / 2, PY + PH + m(3.4), 'distance from rim →', { anchor: 'middle', valign: 'middle', size: 6 });
  L.text(b, m(79), PY + PH + m(3.4), 'illustrative', { anchor: 'end', valign: 'middle', size: 5, italic: true, color: COL.mid });

  // ---------------------------------------------------------------- panel c: direction sets D
  const c = F.panel('c', { x: 148, y: 1, w: 34, h: 70 });
  head(c, 'Direction sets *D*');
  const O = [m(5), m(36.5)], S = m(19);
  const pa = [1, 0], pc = [0, -1], pb = [0.55, -0.2];
  const Pj = (v) => [O[0] + S * (v[0] * pa[0] + v[1] * pb[0] + v[2] * pc[0]), O[1] + S * (v[0] * pa[1] + v[1] * pb[1] + v[2] * pc[1])];
  L.arrow(c, O, Pj([1.15, 0, 0]), { stroke: COL.a, width: 0.6, head: 2.6 });
  L.arrow(c, O, Pj([0, 0, 1.15]), { stroke: COL.c, width: 0.6, head: 2.6 });
  L.arrow(c, O, Pj([0, 1.25, 0]), { stroke: COL.b, width: 0.6, head: 2.6 });
  L.text(c, Pj([1.2, 0, 0])[0] + m(0.6), Pj([1.2, 0, 0])[1], 'a', { color: COL.a, valign: 'middle', weight: 'bold' });
  L.text(c, Pj([0, 0, 1.2])[0], Pj([0, 0, 1.2])[1] - m(0.6), 'c', { color: COL.c, anchor: 'middle', valign: 'bottom', weight: 'bold' });
  L.text(c, Pj([0, 1.3, 0])[0] + m(0.5), Pj([0, 1.3, 0])[1] - m(0.3), 'b', { color: COL.b, anchor: 'start', valign: 'bottom', weight: 'bold' });
  // a general profile direction and its angles
  const nrm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
  const u = nrm([0.3, 0.55, 0.78]);
  { const tip = u.map((x) => x * 1.25), foot = [tip[0], 0, tip[2]]; L.line(c, Pj(tip), Pj(foot), { stroke: COL.mid, width: 0.35, dash: '1 0.8' }); L.line(c, Pj(foot), O, { stroke: COL.mid, width: 0.35, dash: '1 0.8' }); }
  L.arrow(c, O, Pj(u.map((x) => x * 1.25)), { stroke: COL.ink, width: 0.9, head: 3 });
  L.text(c, Pj(u.map((x) => x * 1.3))[0] + m(0.5), Pj(u.map((x) => x * 1.3))[1] - m(0.3), 'profile', { size: 6, valign: 'bottom' });
  const arc = (v, r, col, lab, lr) => {
    const pts = [];
    for (let i = 0; i <= 24; i++) { const s = i / 24; const w = nrm(v.map((x, k) => x * (1 - s) + u[k] * s)); pts.push(Pj(w.map((x) => x * r))); }
    L.polyline(c, pts, { stroke: col, width: 0.45 });
    const mid = nrm(v.map((x, k) => x + u[k]));
    const q = Pj(mid.map((x) => x * lr));
    L.text(c, q[0], q[1], lab, { color: col, anchor: 'middle', valign: 'middle', size: 6 });
  };
  arc([1, 0, 0], 0.36, COL.a, 'α', 0.47);
  arc([0, 1, 0], 0.62, COL.b, 'β', 0.74);
  arc([0, 0, 1], 0.5, COL.c, 'γ', 0.63);
  // the relation and the clock
  L.text(c, m(1), m(48.5), '*D*_{profile} =', { size: 6 });
  L.text(c, m(1), m(52.8), '{#0072B2|*D*_{a}} cos²α + *D*_{b} cos²β + {#D55E00|*D*_{c}} cos²γ', { size: 5.5 });
  // clock glyph
  const kc = [m(3.4), m(62)], kr = m(2.3);
  L.circle(c, kc[0], kc[1], kr, { fill: '#fff', stroke: COL.ink, width: 0.5 });
  L.line(c, kc, [kc[0], kc[1] - kr * 0.7], { stroke: COL.ink, width: 0.5 });
  L.line(c, kc, [kc[0] + kr * 0.5, kc[1] + kr * 0.15], { stroke: COL.ink, width: 0.5 });
  runs(L, c, m(7), m(62), ['time ~ width² / *D*_{profile}'], { size: 6, valign: 'middle' });
}
