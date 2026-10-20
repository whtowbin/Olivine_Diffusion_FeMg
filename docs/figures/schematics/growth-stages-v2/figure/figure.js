// Growth stages, version 2: two growth stages whose zones match joint-fit profile 1 (rim zone,
// mantle, core; steps and levels from joint-fit/figure/data/profiles.json). Each zone diffuses from
// the time it grew. Every curve and colour map comes from figure/data/curves.json (FORSTER's own
// model; see compute_curves.py).
export const figure = { width: 183, height: 116, name: 'growth-stages-v2', format: 'journal' };

export default async function draw(F, L) {
  const D = await (await fetch(new URL('./data/curves.json', import.meta.url))).json();
  const C = L.STYLE.color;
  const m = L.mm;
  const BLUE = C.blue;                       // the staged model (protagonist)
  const GREY = '#8a8a8a';
  const K = D.case;
  const FO_STOPS = ['#F5D27F', '#B9B24F', '#56782B'];   // Fo 0.80 (light amber) -> 0.90 (olive green)
  const FO_LO = K.edge_fo, FO_HI = K.core_fo;
  const foColor = (fo) => L.ramp(FO_STOPS, (fo - FO_LO) / (FO_HI - FO_LO));
  const X = D.x_um;
  const S = D.snapshots;
  const fmt = (v, n = 2) => String(+v.toFixed(n));          // 1.25 -> "1.25", 0.50 -> "0.5"
  const t1 = K.stage1_start_days, t2 = K.stage2_start_days, tObs = K.observed_days;
  const s1 = K.step_outer_um, s2 = K.step_inner_um, LEN = K.profile_um;

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
  const OCT = [[-225, -55], [-167.5, -112.5], [167.5, -112.5], [225, -55], [225, 55], [167.5, 112.5], [-167.5, 112.5], [-225, 55]];
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
  const SC = 0.058;                           // mm per um in the crystal sections
  const RIM_INK = L.shade(FO_STOPS[2], -0.35);
  const crystal = (P, cx, cy, snap, { ghostRim = false, oldRims = [] } = {}) => {
    const g = L.group(P, { x: m(cx), y: m(cy) }).node;
    const toPt = (pts) => pts.map(([x, y]) => [m(x * SC), m(y * SC)]);
    const first = X[snap.findIndex((v) => v != null)];       // the rim at this moment
    if (ghostRim) L.polygon(g, toPt(OCT), { fill: '#f6f6f6', stroke: GREY, width: 0.4, dash: '1.4 1' });
    for (let d = first; d < 111; d += 1) {
      const fo = foAt(snap, d + 0.5);
      L.polygon(g, toPt(inset(OCT, d)), { fill: foColor(fo), stroke: 'none' });
    }
    L.polygon(g, toPt(inset(OCT, first)), { fill: 'none', stroke: RIM_INK, width: 0.5 });
    for (const r of oldRims) L.polygon(g, toPt(inset(OCT, r)), { fill: 'none', stroke: RIM_INK, width: 0.3, dash: '1 0.8' });
    // the traverse: from the rim (left face) inward over the profile's length
    const a = [m(-225 * SC), 0], b = [m((-225 + LEN) * SC), 0];
    L.line(g, a, b, { stroke: C.ink, width: 0.7 });
    L.line(g, [b[0], -m(0.6)], [b[0], m(0.6)], { stroke: C.ink, width: 0.5 });
    return { g, a, b };
  };

  // Profile axes: Fo vs distance from rim, 0-80 um
  const PX0 = 9.5, PX1 = 42.5, PY0 = 33.5, PY1 = 59;          // panel mm
  const XR = [0, 80], YR = [0.795, 0.912];
  const px = (d) => m(PX0 + ((d - XR[0]) / (XR[1] - XR[0])) * (PX1 - PX0));
  const py = (fo) => m(PY1 - ((fo - YR[0]) / (YR[1] - YR[0])) * (PY1 - PY0));
  const axes = (P, ylab = false) => {
    L.line(P, [px(0), py(YR[0])], [px(XR[1]), py(YR[0])], { stroke: C.ink, width: 0.5 });
    L.line(P, [px(0), py(YR[0])], [px(0), py(YR[1])], { stroke: C.ink, width: 0.5 });
    for (const d of [0, 20, 40, 60, 80]) {
      L.line(P, [px(d), py(YR[0])], [px(d), py(YR[0]) + 2], { stroke: C.ink, width: 0.5 });
      L.text(P, px(d), py(YR[0]) + 3, String(d), { size: 5.5, anchor: 'middle', valign: 'top' });
    }
    for (const fo of [0.80, 0.85, 0.90]) {
      L.line(P, [px(0) - 2, py(fo)], [px(0), py(fo)], { stroke: C.ink, width: 0.5 });
      L.text(P, px(0) - 3, py(fo), fo.toFixed(2), { size: 5.5, anchor: 'end', valign: 'middle' });
    }
    L.text(P, (px(0) + px(XR[1])) / 2, py(YR[0]) + 9.5, 'distance from rim (µm)', { size: 6, anchor: 'middle', valign: 'top' });
    if (ylab) L.text(P, m(1.4), (py(YR[0]) + py(YR[1])) / 2, 'Fo', { size: 6, anchor: 'middle', valign: 'middle', rotate: -90 });
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
    L.line(P, [bx, y + m(0.7)], [px(LEN), py(YR[1]) - m(0.6)], { stroke: GREY, width: 0.3, dash: '1.2 0.9' });
  };
  const heading = (P, title, when) => {
    L.text(P, m(2.5), m(3.0), title, { size: 7, weight: 'bold', valign: 'baseline' });
    L.text(P, m(2.5), m(6.4), when, { size: 6, color: '#555', valign: 'baseline' });
  };
  // a shaded band over a distance span of the plot (not grown yet, or a new zone)
  const band = (P, d0, d1, fill) => L.rect(P, px(d0) + 0.3, py(YR[1]), px(d1) - px(d0) - 0.3, py(YR[0]) - py(YR[1]) - 0.3, { fill, stroke: 'none' });
  const NOT_GROWN = '#efefef';
  const zoneTint = (fo) => L.shade(foColor(fo), 0.55);
  const CX = 18.5, CY = 17.5;                 // crystal centre (panel mm)
  const PW = 44.5, PH = 66;

  // ---- a: stage 0, the core crystal -------------------------------------------------------------
  const a = F.panel('a', { x: 0.5, y: 0.5, w: PW, h: PH });
  heading(a, 'Stage 0: core crystal', `t = 0 to ${fmt(t1)} d`);
  const ca = crystal(a, CX, CY, S.before_stage1, { ghostRim: true });
  wedge(a, ca, CX, CY);
  L.callout(a, [m(CX + 150 * SC), m(CY + 114 * SC)], [m(CX + 150 * SC + 2.5), m(CY + 114 * SC + 2.6)], 'final rim', { size: 5.5, anchor: 'start' });
  L.callout(a, [m(CX - 222 * SC), m(CY + 0.4)], [m(1.2), m(CY + 8.5)], 'profile', { size: 5.5, anchor: 'start', dot: false });
  // Fo colour key
  {
    const kx = m(36.4), ky0 = m(10.5), ky1 = m(23.5), stops = 24;
    for (let k = 0; k < stops; k++) {
      const fo = FO_HI - ((FO_HI - FO_LO) * k) / (stops - 1);
      L.rect(a, kx, ky0 + ((ky1 - ky0) * k) / stops, m(1.8), (ky1 - ky0) / stops + 0.2, { fill: foColor(fo), stroke: 'none' });
    }
    L.rect(a, kx, ky0, m(1.8), ky1 - ky0, { fill: 'none', stroke: C.ink, width: 0.3 });
    for (const [fo, t] of [[0.90, 0], [0.85, 0.5], [0.80, 1]]) L.text(a, kx + m(2.5), ky0 + (ky1 - ky0) * t, fo.toFixed(2), { size: 5.5, valign: 'middle' });
    L.text(a, kx + m(0.9), ky0 - m(1.4), 'Fo', { size: 6, anchor: 'middle', valign: 'bottom' });
  }
  axes(a, true);
  band(a, 0, s2, NOT_GROWN);
  L.text(a, px(s2 / 2), py(0.835), 'not yet\ngrown', { size: 5.5, anchor: 'middle', valign: 'middle', color: '#666' });
  curve(a, [s2, s2, XR[1]], [K.mantle_fo, K.core_fo, K.core_fo], { stroke: GREY, dash: '1.6 1.1', width: 0.7 });
  curve(a, X, S.before_stage1, { stroke: BLUE });
  L.text(a, px(66), py(K.core_fo) - m(1.0), 't = 0', { size: 6, anchor: 'middle', valign: 'bottom', color: '#555' });
  L.callout(a, [px(38.6), py(foAt(S.before_stage1, 38.6))], [px(47), py(0.858)], `${fmt(t1)} d`, { size: 6, anchor: 'start', textColor: BLUE, color: BLUE });

  // ---- b: stage 1, the mantle grows ----------------------------------------------------------------
  const b = F.panel('b', { x: 46.5, y: 0.5, w: PW, h: PH });
  heading(b, 'Stage 1: mantle grows', `${fmt(t1)} to ${fmt(t2)} d`);
  const cb = crystal(b, CX, CY, S.before_stage2, { ghostRim: true, oldRims: [s2] });
  wedge(b, cb, CX, CY);
  L.callout(b, [m(CX + 120 * SC), m(CY - 88 * SC)], [m(CX + 150 * SC + 2.5), m(CY - 114 * SC - 2.6)], 'new zone', { size: 5.5, anchor: 'start' });
  L.callout(b, [m(CX + 189 * SC), m(CY + 30 * SC)], [m(CX + 225 * SC + 2.2), m(CY + 3.2)], 'old rim', { size: 5.5, anchor: 'start' });
  axes(b);
  band(b, 0, s1, NOT_GROWN);
  band(b, s1, s2, zoneTint(K.mantle_fo));
  L.text(b, px(s1 / 2), py(0.835), 'not\nyet\ngrown', { size: 5.5, anchor: 'middle', valign: 'middle', color: '#666' });
  L.text(b, px((s1 + s2) / 2), py(0.835), 'new\nzone', { size: 5.5, anchor: 'middle', valign: 'middle', color: '#5a5a10' });
  curve(b, X, S.after_stage1, { stroke: GREY, dash: '1.6 1.1', width: 0.7 });
  curve(b, X, S.before_stage2, { stroke: BLUE });
  L.callout(b, [px(41.5), py(foAt(S.after_stage1, 41.5))], [px(31), py(0.897)], `${fmt(t1)} d`, { size: 6, anchor: 'end', textColor: '#555', color: GREY, dot: false });
  L.callout(b, [px(40), py(foAt(S.before_stage2, 40))], [px(47), py(0.858)], `${fmt(t2)} d`, { size: 6, anchor: 'start', textColor: BLUE, color: BLUE });

  // ---- c: stage 2, the rim zone grows ----------------------------------------------------------------
  const c = F.panel('c', { x: 92.5, y: 0.5, w: PW, h: PH });
  heading(c, 'Stage 2: rim zone grows', `at ${fmt(t2)} d`);
  const cc = crystal(c, CX, CY, S.after_stage2, { oldRims: [s1, s2] });
  wedge(c, cc, CX, CY);
  L.callout(c, [m(CX + 140 * SC), m(CY - 106 * SC)], [m(CX + 150 * SC + 2.5), m(CY - 114 * SC - 2.6)], 'new zone', { size: 5.5, anchor: 'start' });
  L.callout(c, [m(CX + 210 * SC), m(CY + 30 * SC)], [m(CX + 225 * SC + 2.2), m(CY + 3.2)], 'old rims', { size: 5.5, anchor: 'start' });
  axes(c);
  band(c, 0, s1, zoneTint(K.rim_fo));
  L.text(c, px(s1 / 2), py(0.885), 'new\nzone', { size: 5.5, anchor: 'middle', valign: 'middle', color: '#7a5a10' });
  curve(c, X, S.after_stage2, { stroke: BLUE });
  L.callout(c, [px(s1), py(0.848)], [px(22), py(0.832)], 'sharp step', { size: 6, anchor: 'start' });
  L.callout(c, [px(0.3), py(0.812)], [px(17.5), py(0.806)], `edge held at ${K.edge_fo.toFixed(2)}`, { size: 6, anchor: 'start' });
  L.callout(c, [px(42), py(foAt(S.after_stage2, 42))], [px(49), py(0.862)], `diffused ${fmt(t2)} d`, { size: 6, anchor: 'start' });

  // ---- d: measured profile and fit -----------------------------------------------------------------
  const d = F.panel('d', { x: 138.5, y: 0.5, w: PW, h: PH });
  heading(d, 'Measured profile', `at ${fmt(tObs)} d`);
  const cd = crystal(d, CX, CY, S.end);
  wedge(d, cd, CX, CY);
  axes(d);
  const mx = D.measured.x_um, mf = D.measured.fo;
  for (let i = 0; i < mx.length; i++) if (mx[i] <= XR[1]) L.circle(d, px(mx[i]), py(mf[i]), 0.85, { fill: '#fff', stroke: '#6f6f6f', width: 0.35 });
  curve(d, D.staged_fit.model_x_um, D.staged_fit.best_curve, { stroke: BLUE });
  L.callout(d, [px(mx[2]), py(mf[2])], [px(14), py(0.812)], 'measured', { size: 6, anchor: 'start', halo: true });
  L.callout(d, [px(48), py(foAt(D.staged_fit.best_curve, 48))], [px(50), py(0.862)], 'staged model', { size: 6, anchor: 'start', textColor: BLUE, color: BLUE });
  // ages of the steps
  const yb = py(0.906);
  L.bracket(d, [px(0.5), yb], [px(20), yb], { side: 'bottom', depth: 1.6 });
  L.bracket(d, [px(24), yb], [px(60), yb], { side: 'bottom', depth: 1.6 });
  L.text(d, px(10.25), yb - m(0.8), `outer steps\n${fmt(tObs - t2)} d`, { size: 6, anchor: 'middle', valign: 'bottom' });
  L.text(d, px(42), yb - m(0.8), `inner step\n${fmt(tObs)} d`, { size: 6, anchor: 'middle', valign: 'bottom' });

  // ---- e: misfit against time ----------------------------------------------------------------------
  const e = F.panel('e', { x: 0.5, y: 67, w: 182, h: 48.5 });
  const TMAX = K.fit_max_days;
  const TX0 = 14, TX1 = 176, TR = [0, TMAX];
  const tx = (t) => m(TX0 + ((t - TR[0]) / (TR[1] - TR[0])) * (TX1 - TX0));
  const MY0 = 15, MY1 = 34.5;                 // misfit plot band (mm)
  const LR = [Math.log10(30), Math.log10(1100)];
  const my = (chi) => m(MY1 - ((Math.log10(chi) - LR[0]) / (LR[1] - LR[0])) * (MY1 - MY0));
  const tb = D.staged_fit.time_days, ts = D.single_fit.time_days;
  L.text(e, m(5), m(3.0), 'Misfit against time', { size: 7, weight: 'bold', valign: 'baseline' });
  // stage bars
  const SB0 = 7.7, SB1 = 11.7;
  const bars = [[0, t1, K.core_fo, 'stage 0: core crystal'], [t1, t2, K.mantle_fo, 'stage 1: mantle grows'], [t2, TMAX, K.rim_fo, 'stage 2: rim zone grows']];
  for (const [b0, b1, fo, lab] of bars) {
    L.rect(e, tx(b0), m(SB0), tx(b1) - tx(b0), m(SB1 - SB0), { fill: L.shade(foColor(fo), 0.6), stroke: 'none' });
    L.text(e, (tx(b0) + tx(b1)) / 2, m((SB0 + SB1) / 2), lab, { size: 6, anchor: 'middle', valign: 'middle' });
  }
  // no-fit interval: before the last stage
  L.rect(e, tx(0), m(MY0), tx(t2) - tx(0), m(MY1 - MY0), { fill: NOT_GROWN, stroke: 'none' });
  L.text(e, tx(0.9), m(MY0 + 3.2), 'rim zone not grown:\nno staged fit', { size: 6, anchor: 'middle', valign: 'middle', color: '#666' });
  for (const t of [t1, t2]) L.line(e, [tx(t), m(SB0)], [tx(t), m(MY1)], { stroke: '#555', width: 0.4, dash: '1.4 1' });
  // misfit curves (clipped to the band)
  const clip = L.clipPath(e, `M${tx(0)},${m(MY0)} H${tx(TMAX)} V${m(MY1)} H${tx(0)} Z`);
  const gc = L.group(e, { clip }).node;
  const mis = (fit, style) => {
    const pts = [];
    fit.misfit.t_days.forEach((t, i) => { const v = fit.misfit.chi2[i]; if (v != null && t <= TMAX) pts.push([tx(t), my(Math.min(v, 5000))]); });
    L.polyline(gc, pts, { fill: 'none', width: 0.9, join: 'round', ...style });
  };
  mis(D.single_fit, { stroke: GREY, dash: '2 1.2' });
  mis(D.staged_fit, { stroke: BLUE });
  // axes
  L.line(e, [tx(0), m(MY1)], [tx(TMAX), m(MY1)], { stroke: C.ink, width: 0.5 });
  L.line(e, [tx(0), m(MY0)], [tx(0), m(MY1)], { stroke: C.ink, width: 0.5 });
  for (let k = 0; k * 0.25 <= TMAX + 1e-9; k++) {
    const t = k * 0.25;
    L.line(e, [tx(t), m(MY1)], [tx(t), m(MY1) + 2], { stroke: C.ink, width: 0.5 });
    L.text(e, tx(t), m(MY1) + 3, fmt(t), { size: 5.5, anchor: 'middle', valign: 'top' });
  }
  L.text(e, tx(TMAX), m(MY1 + 4.6), 'time since stage 0 (days)', { size: 6, anchor: 'end', valign: 'top' });
  L.text(e, m(TX0 - 3.5), m((MY0 + MY1) / 2), 'misfit', { size: 6, anchor: 'middle', valign: 'middle', rotate: -90 });
  L.text(e, m(TX0 - 1.2), m(MY0 + 0.6), 'high', { size: 5.5, anchor: 'end', valign: 'top', color: '#555' });
  L.text(e, m(TX0 - 1.2), m(MY1 - 0.4), 'low', { size: 5.5, anchor: 'end', valign: 'bottom', color: '#555' });
  // minima
  const yMinS = my(D.staged_fit.chi2_min), yMinG = my(D.single_fit.chi2_min);
  L.circle(e, tx(ts), yMinG, 1.5, { fill: '#fff', stroke: GREY, width: 0.6 });
  L.circle(e, tx(tb), yMinS, 1.5, { fill: BLUE, stroke: '#fff', width: 0.4 });
  L.callout(e, [tx(tb), yMinS], [tx(1.47), my(300)], 'best fit', { size: 6, anchor: 'start', textColor: BLUE, color: BLUE, dot: false });
  L.text(e, tx(1.66), my(115), 'staged model', { size: 6, anchor: 'start', valign: 'middle', color: BLUE });
  // The single-stage fit reaches as low a misfit. FORSTER's near-step check may or may not flag it:
  // the label says what the check gave on this profile, and the residuals against the noise.
  const NS = D.near_step;
  const flagged = NS.single_delta_aic < NS.warn_below;
  const ratio = NS.single_rms / NS.noise_sd;
  L.callout(e, [tx(ts), yMinG - m(0.55)], [tx(0.2), my(300)],
    `${fmt(ts)} d\n${flagged ? 'near-step warning' : 'no near-step warning'}\nresiduals ${ratio.toFixed(1)} × noise`,
    { size: 6, anchor: 'start', textColor: '#666', color: GREY, dot: false });
  L.text(e, tx(0.9), my(48) - m(1.0), 'all zones from t = 0', { size: 6, anchor: 'middle', valign: 'bottom', color: '#666' });
  // the two reported times
  const R1 = MY1 + 8.4, R2 = MY1 + 11.8;
  const span = (t0, t1_, y, label) => {
    L.line(e, [tx(t0), m(y)], [tx(t1_), m(y)], { stroke: BLUE, width: 0.6 });
    for (const t of [t0, t1_]) L.line(e, [tx(t), m(y) - 1.6], [tx(t), m(y) + 1.6], { stroke: BLUE, width: 0.6 });
    L.text(e, tx(t1_) + m(1.2), m(y), label, { size: 6, anchor: 'start', valign: 'middle', color: BLUE });
  };
  L.line(e, [tx(tb), yMinS + m(1.5)], [tx(tb), m(MY1)], { stroke: BLUE, width: 0.3, dash: '1 0.8' });
  L.line(e, [tx(tb), m(MY1 + 6.6)], [tx(tb), m(R2)], { stroke: BLUE, width: 0.3, dash: '1 0.8' });
  span(0, tb, R1, `since first stage: ${fmt(tb)} d`);
  span(t2, tb, R2, `since last stage: ${fmt(D.staged_fit.since_last_days)} d`);
  // snapshot markers on the stage bar
  const mark = (t, s) => {
    L.circle(e, tx(t), m(SB0) - m(1.6), m(1.25), { fill: '#fff', stroke: C.ink, width: 0.4 });
    L.text(e, tx(t), m(SB0) - m(1.6), s, { size: 6, weight: 'bold', anchor: 'middle', valign: 'middle' });
  };
  mark(0, 'a'); mark(t1, 'b'); mark(t2, 'c'); mark(tb, 'd');
}
