// Growth stages: a crystal grows in instantaneous steps; each zone diffuses from the time it grew.
// Every curve and colour map comes from figure/data/curves.json (FORSTER's own model; see compute_curves.py).
export const figure = { width: 183, height: 118, name: 'growth-stages', format: 'journal' };

export default async function draw(F, L) {
  const D = await (await fetch(new URL('./data/curves.json', import.meta.url))).json();
  const C = L.STYLE.color;
  const m = L.mm;
  const BLUE = C.blue;                       // the staged model (protagonist)
  const GREY = '#8a8a8a';
  const FO_STOPS = ['#F5D27F', '#B9B24F', '#56782B'];   // Fo 0.82 (light amber) -> 0.90 (olive green)
  const foColor = (fo) => L.ramp(FO_STOPS, (fo - 0.82) / 0.08);
  const X = D.x_um;

  // ---- shared parts -------------------------------------------------------------------------------
  // Fo at distance d (um) from the final rim in a snapshot (null = not yet grown).
  const foAt = (snap, d) => {
    if (d >= X[X.length - 1]) return snap[snap.length - 1];
    let i = 0; while (X[i + 1] < d) i++;
    const a = snap[i], b = snap[i + 1];
    if (a == null && b == null) return null;
    if (a == null) return d >= (X[i] + X[i + 1]) / 2 ? b : null;
    if (b == null) return a;
    const f = (d - X[i]) / (X[i + 1] - X[i]); return a + f * (b - a);
  };

  // Generic elongate eight-sided olivine section (habit unknown: kept generic). Crystal units um.
  const OCT = [[-450, -110], [-335, -225], [335, -225], [450, -110], [450, 110], [335, 225], [-335, 225], [-450, 110]];
  // inset a convex polygon by d (intersection of the offset edge lines)
  const inset = (poly, d) => {
    const n = poly.length, lines = [];
    let area = 0; for (let i = 0; i < n; i++) { const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % n]; area += x0 * y1 - x1 * y0; }
    const s = area > 0 ? 1 : -1;
    for (let i = 0; i < n; i++) {
      const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % n];
      const len = Math.hypot(x1 - x0, y1 - y0), nx = (-(y1 - y0) / len) * s, ny = ((x1 - x0) / len) * s;   // inward normal
      lines.push([[x0 + nx * d, y0 + ny * d], [x1 - x0, y1 - y0]]);
    }
    const out = [];
    for (let i = 0; i < n; i++) {
      const [p, r] = lines[(i + n - 1) % n], [q, u] = lines[i];
      const den = r[0] * u[1] - r[1] * u[0];
      const t = ((q[0] - p[0]) * u[1] - (q[1] - p[1]) * u[0]) / den;
      out.push([p[0] + t * r[0], p[1] + t * r[1]]);
    }
    return out;
  };
  const SC = 0.0372;                          // mm per um in the crystal sections
  const crystal = (P, cx, cy, snap, { ghostRim = false, oldRim = false } = {}) => {
    const g = L.group(P, { x: m(cx), y: m(cy) }).node;
    const toPt = (pts) => pts.map(([x, y]) => [m(x * SC), m(y * SC)]);
    const first = foAt(snap, 0) == null ? 40 : 0;
    if (ghostRim) L.polygon(g, toPt(OCT), { fill: '#f6f6f6', stroke: GREY, width: 0.4, dash: '1.4 1' });
    for (let d = first; d < 210; d += 2) {
      const fo = foAt(snap, Math.min(d + 1, 199));
      L.polygon(g, toPt(inset(OCT, d)), { fill: foColor(fo), stroke: 'none' });
    }
    L.polygon(g, toPt(inset(OCT, first)), { fill: 'none', stroke: L.shade(FO_STOPS[2], -0.35), width: 0.5 });
    if (oldRim) L.polygon(g, toPt(inset(OCT, 40)), { fill: 'none', stroke: L.shade(FO_STOPS[2], -0.35), width: 0.3, dash: '1 0.8' });
    // the traverse: from the rim (left face) 150 um inward, the span the plot below shows
    const a = [m(-450 * SC), 0], b = [m(-300 * SC), 0];
    L.line(g, a, b, { stroke: C.ink, width: 0.7 });
    L.line(g, [b[0], -m(0.6)], [b[0], m(0.6)], { stroke: C.ink, width: 0.5 });
    return { g, a, b, rimX: -450 * SC, halfH: 225 * SC, halfW: 450 * SC };
  };

  // Profile axes: Fo vs distance from rim, 0-150 um
  const PX0 = 9, PX1 = 56, PY0 = 31.5, PY1 = 58;          // panel mm
  const XR = [0, 150], YR = [0.812, 0.908];
  const px = (d) => m(PX0 + ((d - XR[0]) / (XR[1] - XR[0])) * (PX1 - PX0));
  const py = (fo) => m(PY1 - ((fo - YR[0]) / (YR[1] - YR[0])) * (PY1 - PY0));
  const axes = (P, ylab = true) => {
    L.line(P, [px(0), py(YR[0])], [px(150), py(YR[0])], { stroke: C.ink, width: 0.5 });
    L.line(P, [px(0), py(YR[0])], [px(0), py(YR[1])], { stroke: C.ink, width: 0.5 });
    for (const d of [0, 50, 100, 150]) {
      L.line(P, [px(d), py(YR[0])], [px(d), py(YR[0]) + 2], { stroke: C.ink, width: 0.5 });
      L.text(P, px(d), py(YR[0]) + 3, String(d), { size: 5.5, anchor: 'middle', valign: 'top' });
    }
    for (const fo of [0.82, 0.86, 0.90]) {
      L.line(P, [px(0) - 2, py(fo)], [px(0), py(fo)], { stroke: C.ink, width: 0.5 });
      L.text(P, px(0) - 3, py(fo), fo.toFixed(2), { size: 5.5, anchor: 'end', valign: 'middle' });
    }
    L.text(P, (px(0) + px(150)) / 2, py(YR[0]) + 9.5, 'distance from rim (µm)', { size: 6, anchor: 'middle', valign: 'top' });
    if (ylab) L.text(P, m(1.6), (py(YR[0]) + py(YR[1])) / 2, 'Fo', { size: 6, anchor: 'middle', valign: 'middle', rotate: -90 });
  };
  const curve = (P, xs, ys, style) => {
    const pts = [];
    for (let i = 0; i < xs.length; i++) if (ys[i] != null && xs[i] <= XR[1]) pts.push([px(xs[i]), py(ys[i])]);
    L.polyline(P, pts, { fill: 'none', width: 0.9, join: 'round', cap: 'round', ...style });
  };
  // zoom wedge from the crystal traverse to the plot's distance axis span
  const wedge = (P, cr, cx, cy) => {
    const ax = m(cx) + cr.a[0], bx = m(cx) + cr.b[0], y = m(cy);
    L.line(P, [ax, y + m(0.7)], [px(0), py(YR[1]) - m(0.6)], { stroke: GREY, width: 0.3, dash: '1.2 0.9' });
    L.line(P, [bx, y + m(0.7)], [px(150), py(YR[1]) - m(0.6)], { stroke: GREY, width: 0.3, dash: '1.2 0.9' });
  };
  const heading = (P, title, when) => {
    L.text(P, m(5), m(3.0), title, { size: 7, weight: 'bold', valign: 'baseline' });
    L.text(P, m(58.5), m(3.0), when, { size: 6, anchor: 'end', color: '#555', valign: 'baseline' });
  };
  const CX = 26, CY = 15.5;                   // crystal centre (panel mm)

  // ---- a: stage 0 -------------------------------------------------------------------------------
  const S = D.snapshots;
  const a = F.panel('a', { x: 0.5, y: 0.5, w: 59, h: 68 });
  heading(a, 'Stage 0: starting crystal', 't = 0 to 6 d');
  const ca = crystal(a, CX, CY, S.before_stage1, { ghostRim: true });
  wedge(a, ca, CX, CY);
  L.callout(a, [m(CX + 300 * SC), m(CY - 228 * SC)], [m(CX + 300 * SC + 3), m(CY - 228 * SC - 3.6)], 'final rim', { size: 5.5, anchor: 'start' });
  L.callout(a, [m(CX - 445 * SC), m(CY + 0.4)], [m(1.2), m(CY + 9.5)], 'profile', { size: 5.5, anchor: 'start', dot: false });
  // Fo colour key
  {
    const kx = m(50.5), ky0 = m(9), ky1 = m(22), stops = 24;
    for (let k = 0; k < stops; k++) {
      const fo = 0.90 - (0.08 * k) / (stops - 1);
      L.rect(a, kx, ky0 + ((ky1 - ky0) * k) / stops, m(1.8), (ky1 - ky0) / stops + 0.2, { fill: foColor(fo), stroke: 'none' });
    }
    L.rect(a, kx, ky0, m(1.8), ky1 - ky0, { fill: 'none', stroke: C.ink, width: 0.3 });
    for (const [fo, t] of [[0.90, 0], [0.86, 0.5], [0.82, 1]]) L.text(a, kx + m(2.5), ky0 + (ky1 - ky0) * t, fo.toFixed(2), { size: 5.5, valign: 'middle' });
    L.text(a, kx + m(0.9), ky0 - m(1.4), 'Fo', { size: 6, anchor: 'middle', valign: 'bottom' });
  }
  axes(a);
  L.rect(a, px(0) + 0.3, py(YR[1]), px(40) - px(0) - 0.3, py(YR[0]) - py(YR[1]) - 0.3, { fill: '#efefef', stroke: 'none' });
  L.text(a, (px(0) + px(40)) / 2, py(0.83), 'not yet\ngrown', { size: 5.5, anchor: 'middle', valign: 'middle', color: '#666' });
  curve(a, [40, 40, 150], [0.86, 0.90, 0.90], { stroke: GREY, dash: '1.6 1.1', width: 0.7 });
  curve(a, X, S.before_stage1, { stroke: BLUE });
  L.text(a, px(118), py(0.90) - m(1.2), 't = 0', { size: 6, anchor: 'middle', valign: 'bottom', color: '#555' });
  L.callout(a, [px(62), py(foAt(S.before_stage1, 62))], [px(80), py(0.872)], '6 d', { size: 6, anchor: 'start', textColor: BLUE, color: BLUE });

  // ---- b: stage 1 appears --------------------------------------------------------------------------
  const b = F.panel('b', { x: 62, y: 0.5, w: 59, h: 68 });
  heading(b, 'Stage 1: new rim zone', 'at 6 d');
  const cb = crystal(b, CX, CY, S.after_stage1, { oldRim: true });
  wedge(b, cb, CX, CY);
  L.callout(b, [m(CX + 250 * SC), m(CY - 205 * SC)], [m(CX + 250 * SC + 4), m(CY - 205 * SC - 4.4)], 'new zone', { size: 5.5, anchor: 'start' });
  L.callout(b, [m(CX + 420 * SC), m(CY + 60 * SC)], [m(CX + 450 * SC + 2.5), m(CY + 3.5)], 'old rim', { size: 5.5, anchor: 'start' });
  axes(b, false);
  L.rect(b, px(0) + 0.3, py(YR[1]), px(40) - px(0) - 0.3, py(YR[0]) - py(YR[1]) - 0.3, { fill: L.shade(FO_STOPS[0], 0.55), stroke: 'none' });
  L.text(b, (px(0) + px(40)) / 2, py(0.89), 'new\nzone', { size: 5.5, anchor: 'middle', valign: 'middle', color: '#7a5a10' });
  curve(b, X, S.after_stage1, { stroke: BLUE });
  L.callout(b, [px(40), py(0.842)], [px(52), py(0.838)], 'sharp step', { size: 6, anchor: 'start' });
  L.callout(b, [px(64), py(foAt(S.after_stage1, 64))], [px(82), py(0.872)], 'diffused 6 d', { size: 6, anchor: 'start' });

  // ---- c: measured profile and fit -----------------------------------------------------------------
  const c = F.panel('c', { x: 123.5, y: 0.5, w: 59, h: 68 });
  heading(c, 'Measured profile', 'at 10 d');
  const cc = crystal(c, CX, CY, S.end);
  wedge(c, cc, CX, CY);
  axes(c, false);
  const mx = D.measured.x_um, mf = D.measured.fo;
  for (let i = 0; i < mx.length; i++) if (mx[i] <= 150) L.circle(c, px(mx[i]), py(mf[i]), 0.85, { fill: '#fff', stroke: '#6f6f6f', width: 0.35 });
  curve(c, D.staged_fit.model_x_um, D.staged_fit.best_curve, { stroke: BLUE });
  L.callout(c, [px(mx[1]), py(mf[1])], [px(14), py(0.835) + m(5)], 'measured', { size: 6, anchor: 'start', halo: true });
  L.callout(c, [px(112), py(foAt(D.staged_fit.best_curve, 112))], [px(108), py(0.876)], 'staged model', { size: 6, anchor: 'start', textColor: BLUE, color: BLUE });
  // ages of the two gradients
  const yb = py(0.905);
  L.bracket(c, [px(12), yb], [px(44), yb], { side: 'bottom', depth: 1.6 });
  L.bracket(c, [px(47), yb], [px(108), yb], { side: 'bottom', depth: 1.6 });
  L.text(c, px(28), yb - m(0.8), 'outer step\n4 d', { size: 6, anchor: 'middle', valign: 'bottom' });
  L.text(c, px(77.5), yb - m(0.8), 'inner gradient\n10 d', { size: 6, anchor: 'middle', valign: 'bottom' });

  // ---- d: misfit against time ----------------------------------------------------------------------
  const d = F.panel('d', { x: 0.5, y: 71, w: 182, h: 46.5 });
  const TX0 = 14, TX1 = 176, TR = [0, 16];
  const tx = (t) => m(TX0 + ((t - TR[0]) / (TR[1] - TR[0])) * (TX1 - TX0));
  const MY0 = 13.5, MY1 = 33;                 // misfit plot band (mm)
  const LR = [Math.log10(38), Math.log10(1500)];
  const my = (chi) => m(MY1 - ((Math.log10(chi) - LR[0]) / (LR[1] - LR[0])) * (MY1 - MY0));
  const t1 = D.case.stage1_start_days, tb = D.staged_fit.time_days, ts = D.single_fit.time_days;
  L.text(d, m(5), m(3.0), 'Misfit against time', { size: 7, weight: 'bold', valign: 'baseline' });
  // stage bar
  const SB0 = 6.2, SB1 = 10.2;
  L.rect(d, tx(0), m(SB0), tx(t1) - tx(0), m(SB1 - SB0), { fill: L.shade(FO_STOPS[2], 0.72), stroke: 'none' });
  L.rect(d, tx(t1), m(SB0), tx(16) - tx(t1), m(SB1 - SB0), { fill: L.shade(FO_STOPS[0], 0.6), stroke: 'none' });
  L.text(d, tx(t1 / 2), m((SB0 + SB1) / 2), 'stage 0: starting crystal', { size: 6, anchor: 'middle', valign: 'middle' });
  L.text(d, tx(t1 + 0.5), m((SB0 + SB1) / 2), 'stage 1: new rim zone', { size: 6, anchor: 'start', valign: 'middle' });
  // no-fit interval
  L.rect(d, tx(0), m(MY0), tx(t1) - tx(0), m(MY1 - MY0), { fill: '#efefef', stroke: 'none' });
  L.text(d, tx(4.3), m(MY0 + 3.2), 'outer zone not grown:\nno staged fit', { size: 6, anchor: 'middle', valign: 'middle', color: '#666' });
  L.line(d, [tx(t1), m(SB0)], [tx(t1), m(MY1)], { stroke: '#555', width: 0.4, dash: '1.4 1' });
  // misfit curves (clipped to the band)
  const clip = L.clipPath(d.node ? d : d, `M${tx(0)},${m(MY0)} H${tx(16)} V${m(MY1)} H${tx(0)} Z`);
  const gc = L.group(d, { clip }).node;
  const mis = (fit, style) => {
    const pts = [];
    fit.misfit.t_days.forEach((t, i) => { const v = fit.misfit.chi2[i]; if (v != null && t <= 16) pts.push([tx(t), my(Math.min(v, 5000))]); });
    L.polyline(gc, pts, { fill: 'none', width: 0.9, join: 'round', ...style });
  };
  mis(D.single_fit, { stroke: GREY, dash: '2 1.2' });
  mis(D.staged_fit, { stroke: BLUE });
  // axes
  L.line(d, [tx(0), m(MY1)], [tx(16), m(MY1)], { stroke: C.ink, width: 0.5 });
  L.line(d, [tx(0), m(MY0)], [tx(0), m(MY1)], { stroke: C.ink, width: 0.5 });
  for (let t = 0; t <= 16; t += 2) {
    L.line(d, [tx(t), m(MY1)], [tx(t), m(MY1) + 2], { stroke: C.ink, width: 0.5 });
    L.text(d, tx(t), m(MY1) + 3, String(t), { size: 5.5, anchor: 'middle', valign: 'top' });
  }
  L.text(d, tx(16), m(MY1 + 4.6), 'time since stage 0 (days)', { size: 6, anchor: 'end', valign: 'top' });
  L.text(d, m(TX0 - 3.5), m((MY0 + MY1) / 2), 'misfit', { size: 6, anchor: 'middle', valign: 'middle', rotate: -90 });
  L.text(d, m(TX0 - 1.2), m(MY0 + 0.6), 'high', { size: 5.5, anchor: 'end', valign: 'top', color: '#555' });
  L.text(d, m(TX0 - 1.2), m(MY1 - 0.4), 'low', { size: 5.5, anchor: 'end', valign: 'bottom', color: '#555' });
  // minima
  const yMinS = my(D.staged_fit.chi2_min), yMinG = my(D.single_fit.chi2_min);
  L.circle(d, tx(ts), yMinG, 1.5, { fill: '#fff', stroke: GREY, width: 0.6 });
  L.circle(d, tx(tb), yMinS, 1.5, { fill: BLUE, stroke: '#fff', width: 0.4 });
  L.callout(d, [tx(tb), yMinS], [tx(10.75), my(420)], 'best fit', { size: 6, anchor: 'start', textColor: BLUE, color: BLUE, dot: false });
  // The single-stage fit reaches as low a misfit, but FORSTER's near-step check flags it.
  const warned = D.near_step.single_delta_aic < D.near_step.warn_below && D.near_step.staged_delta_aic >= D.near_step.warn_below;
  L.text(d, tx(ts), yMinG - m(1.4), `${ts.toFixed(1)} d`, { size: 6, anchor: 'middle', valign: 'bottom', color: '#666', halo: true });
  if (warned) L.callout(d, [tx(ts), yMinG], [tx(5.0), m(MY1 - 2.4)], 'near-step warning', { size: 6, anchor: 'end', textColor: '#666', color: GREY, dot: false });
  L.text(d, tx(13.6), my(420), 'staged model', { size: 6, anchor: 'end', valign: 'middle', color: BLUE });
  L.text(d, tx(15.9), my(86), 'both zones from t = 0', { size: 6, anchor: 'end', valign: 'top', color: '#666' });
  // the two reported times
  const R1 = MY1 + 8.4, R2 = MY1 + 11.8;
  const span = (t0, t1_, y, label) => {
    L.line(d, [tx(t0), m(y)], [tx(t1_), m(y)], { stroke: BLUE, width: 0.6 });
    for (const t of [t0, t1_]) L.line(d, [tx(t), m(y) - 1.6], [tx(t), m(y) + 1.6], { stroke: BLUE, width: 0.6 });
    L.text(d, tx(t1_) + m(1.2), m(y), label, { size: 6, anchor: 'start', valign: 'middle', color: BLUE });
  };
  L.line(d, [tx(tb), yMinS + m(1.5)], [tx(tb), m(MY1)], { stroke: BLUE, width: 0.3, dash: '1 0.8' });
  L.line(d, [tx(tb), m(MY1 + 6.6)], [tx(tb), m(R2)], { stroke: BLUE, width: 0.3, dash: '1 0.8' });
  span(0, tb, R1, 'since first stage: 10.1 d');
  span(t1, tb, R2, 'since last stage: 4.1 d');
  // snapshot markers on the stage bar
  const mark = (t, s) => {
    L.circle(d, tx(t), m(SB0) - m(1.6), m(1.25), { fill: '#fff', stroke: C.ink, width: 0.4 });
    L.text(d, tx(t), m(SB0) - m(1.6), s, { size: 6, weight: 'bold', anchor: 'middle', valign: 'middle' });
  };
  mark(0, 'a'); mark(t1, 'b'); mark(tb, 'c');
}
