// Joint fit of two profiles on one crystal (FORSTER methods paper). All curves and the Fo map are
// computed by figure/data/make_crystal.py and make_profiles.py; they are illustrative, not data.
export const figure = { width: 183, height: 66, name: 'joint-fit', format: 'journal', illustrate: false };

export default async function draw(F, L) {
  const C = L.STYLE.color, mm = L.mm, S = L.STYLE;
  const geo = await (await fetch('/figure/data/crystal.json')).json();
  const pr = await (await fetch('/figure/data/profiles.json')).json();
  const BLUE = C.blue, VERM = C.vermillion, SHARED = '#3a3a3a', GREY = '#8c8c8c';
  const COL = { p1: BLUE, p2: VERM };
  const LV = geo.levels;                       // shared Fo by label (relative), edge 0
  const heading = (P, s) => L.text(P, mm(4.2), mm(2.9), s, { size: S.text.title, valign: 'middle' });

  // =============================== panel a: the crystal ================================================
  const a = F.panel('a', { x: 0, y: 0, w: 57, h: 66 });
  heading(a, 'Zoned olivine section');
  const ka = mm(0.30), [EX0, EY0, EW, EH] = geo.map.extent;
  const mx = mm(10.3), my = mm(13);
  const U = ([x, y]) => [mx + (x - EX0) * ka, my + (y - EY0) * ka];
  // computed Fo map, clipped to the outline
  const outlineD = 'M' + geo.outline.map((q) => U(q).join(',')).join('L') + 'Z';
  const clip = L.clipPath(a, outlineD);
  const gmap = L.group(a, {});
  gmap.node.setAttribute('clip-path', clip);
  L.image(gmap, geo.map.png, mx, my, EW * ka, { h: EH * ka, pxw: geo.map.px[0], pxh: geo.map.px[1] });
  // starting zone boundaries (the steps) as faint dashed lines; the map shows them after diffusion
  for (const k of ['mantle', 'core']) L.polygon(a, geo.zones[k].map(U), { fill: 'none', stroke: '#ffffff', width: 0.35, dash: '1.2 1', opacity: 0.85 });
  L.path(a, outlineD, { fill: 'none', stroke: '#4d4f22', width: S.line.normal });

  // traverses: spots
  for (const k of ['p1', 'p2']) {
    for (const q of geo[k].spots) { const [x, y] = U(q); L.circle(a, x, y, 0.62, { fill: COL[k], stroke: '#ffffff', width: 0.25 }); }
  }
  const [cx, cy] = U(geo.cross);
  L.circle(a, cx, cy, 2.1, { fill: 'none', stroke: SHARED, width: 0.7 });
  // labels for the traverses at their rim ends
  const r1 = U(geo.p1.rim), r2 = U(geo.p2.rim);
  L.text(a, U([-73, 0])[0], U([0, 25])[1], 'Profile 1', { anchor: 'end', color: BLUE, weight: 'bold', size: S.text.label });
  L.text(a, r2[0] + 3, r2[1] - 3.2, 'Profile 2', { anchor: 'start', color: VERM, weight: 'bold', size: S.text.label });
  L.callout(a, [cx + 1.6, cy + 1.6], [U([40, 0])[0] + 2, U([0, 55])[1]], 'crossing point', { anchor: 'start', dot: false });

  // zone labels below the crystal, leaders up into each zone (bottom side of the section)
  const yl = U([0, 54])[1];
  L.callout(a, U([-38, 37]), [U([-60, 0])[0], yl], 'rim zone', { anchor: 'middle' });
  L.callout(a, U([-28, 27]), [U([-30, 0])[0], yl], 'mantle', { anchor: 'middle' });
  L.callout(a, U([-6, 12]), [U([-4, 0])[0], yl], 'core', { anchor: 'middle' });

  // axes glyph: c to the right, a down (section plane illustrative)
  const ox = mm(5.5), oy = mm(51.5);
  L.arrow(a, [ox, oy], [ox + mm(9), oy], { stroke: C.ink, width: 0.5, head: 2.4 });
  L.arrow(a, [ox, oy], [ox, oy + mm(8)], { stroke: C.ink, width: 0.5, head: 2.4 });
  L.text(a, ox + mm(9.8), oy, 'c [001], fast', { valign: 'middle', size: S.text.small });
  L.text(a, ox + mm(1.2), oy + mm(7.2), 'a [100], slow', { valign: 'middle', size: S.text.small });

  // Fo colour bar
  const bx = mm(33), by = mm(57.6), bw = mm(18), bh = mm(1.8);
  const grad = L.linearGradient(a, geo.ramp.map((c, i) => [i / (geo.ramp.length - 1), c]), 0);
  L.rect(a, bx, by, bw, bh, { fill: grad, stroke: '#4d4f22', width: 0.3 });
  L.text(a, bx - 1.2, by + bh / 2, 'Fo', { anchor: 'end', valign: 'middle', size: S.text.small, italic: false });
  L.text(a, bx, by + bh + 5.2, 'low', { anchor: 'start', size: S.text.small, color: C.mid });
  L.text(a, bx + bw, by + bh + 5.2, 'high', { anchor: 'end', size: S.text.small, color: C.mid });

  // =============================== panel b: the two profiles ============================================
  const b = F.panel('b', { x: 59, y: 0, w: 85, h: 66 });
  heading(b, 'Fo profiles from one crystal');
  const kb = mm(0.40), ptop = mm(18), ph = mm(35);
  const FO = [-0.4, 3.65];
  const Y = (v) => ptop + ph * (1 - (v - FO[0]) / (FO[1] - FO[0]));
  const plots = { p1: { x0: mm(7.5), span: 80 }, p2: { x0: mm(7.5) + 80 * kb + mm(5), span: 54 } };
  for (const k of ['p1', 'p2']) plots[k].X = (u) => plots[k].x0 + u * kb;
  const right = plots.p2.X(plots.p2.span);

  // shared levels across both plots (grey) + their labels at the right
  const lvl = [['core', LV.core], ['mantle', LV.mantle], ['rim zone', LV.rim]];
  for (const [, v] of lvl) L.line(b, [plots.p1.x0, Y(v)], [right + mm(1.5), Y(v)], { stroke: GREY, width: 0.4, dash: '1.6 1.1' });
  L.line(b, [plots.p1.x0, Y(LV.edge)], [right + mm(1.5), Y(LV.edge)], { stroke: SHARED, width: 0.4, dash: '0.5 0.9', cap: 'round' });
  for (const [s, v] of lvl) L.text(b, right + mm(2.2), Y(v), s, { valign: 'middle', size: S.text.label, color: SHARED });
  L.text(b, right + mm(2.2), Y(LV.edge), 'edge Fo', { valign: 'middle', size: S.text.label, color: SHARED });

  for (const k of ['p1', 'p2']) {
    const P = plots[k], d = pr[k], col = COL[k], X = P.X, x1 = X(P.span);
    // axes
    L.line(b, [P.x0, ptop + ph], [x1, ptop + ph], { stroke: C.ink, width: 0.5 });
    L.line(b, [P.x0, ptop], [P.x0, ptop + ph], { stroke: C.ink, width: 0.5 });
    L.text(b, (P.x0 + x1) / 2, ptop + ph + mm(5.6), 'distance from rim', { anchor: 'middle', size: S.text.small, color: C.mid });
    // starting profile (light), model curve, spots
    const xsM = d.curve_x.map((xm) => xm + d.edge_off);           // model distance from the true rim
    const keep = xsM.map((u) => u >= 0 && u <= P.span);
    const ptsS = [], ptsC = [];
    xsM.forEach((u, i) => { if (keep[i]) { ptsS.push([X(u), Y(d.start_fo[i])]); ptsC.push([X(u), Y(d.curve_fo[i])]); } });
    // the starting profile begins at the edge value at the rim: a step from edge Fo to the rim zone
    ptsS.unshift([X(0), Y(LV.edge + d.drift)]);
    L.polyline(b, ptsS, { stroke: L.shade(col, 0.5), width: 0.55, join: 'miter' });
    L.polyline(b, [[X(0), Y(LV.edge + d.drift)], ...ptsC], { stroke: col, width: 1.0, join: 'round' });
    d.x_meas.forEach((xm, i) => { const u = xm + d.edge_off; if (u <= P.span) L.circle(b, X(u), Y(d.fo[i]), 0.85, { fill: col, stroke: '#ffffff', width: 0.25 }); });
    // shared edge Fo at the rim: black dot
    L.circle(b, X(0), Y(LV.edge), 1.25, { fill: SHARED, stroke: '#ffffff', width: 0.3 });
    // steps: own to each profile (triangles under the axis)
    for (const s of d.steps) {
      const tx = X(s), ty = ptop + ph + 0.4;
      L.polygon(b, [[tx, ty], [tx - 1.6, ty + 2.6], [tx + 1.6, ty + 2.6]], { fill: col });
    }
    // edge offset: rim to first spot
    const fy = ptop + ph - 3.2;
    L.line(b, [X(0), fy], [X(d.edge_off), fy], { stroke: col, width: 0.5 });
    L.line(b, [X(0), fy - 1.3], [X(0), fy + 1.3], { stroke: col, width: 0.5 });
    L.line(b, [X(d.edge_off), fy - 1.3], [X(d.edge_off), fy + 1.3], { stroke: col, width: 0.5 });
    // crossing point on the curve
    const ux = d.cross_meas + d.edge_off;
    let ci = 0; xsM.forEach((u, i) => { if (Math.abs(u - ux) < Math.abs(xsM[ci] - ux)) ci = i; });
    L.circle(b, X(ux), Y(d.curve_fo[ci]), 2.1, { fill: 'none', stroke: SHARED, width: 0.7 });
    P.cross = [X(ux), Y(d.curve_fo[ci])];
    // title
    L.text(b, P.x0 + 1, ptop - mm(1.6), k === 'p1' ? 'Profile 1: near c, high D' : 'Profile 2: near a, low D',
      { size: S.text.label, color: col, weight: 'bold' });
  }
  // y axis label with the Fo colour strip (same ramp as panel a)
  const g2 = L.linearGradient(b, geo.ramp.map((c, i) => [1 - i / (geo.ramp.length - 1), c]).reverse(), 90);
  const sx = plots.p1.x0 - mm(2.2);
  L.rect(b, sx, Y(LV.core), mm(1.1), Y(LV.edge) - Y(LV.core), { fill: g2, stroke: 'none' });
  L.text(b, sx - mm(1.6), ptop + ph / 2, 'Fo', { anchor: 'middle', valign: 'middle', size: S.text.label, rotate: -90 });

  // direct labels on profile 1 (the template, learned once)
  const p1 = plots.p1, d1 = pr.p1, X1 = p1.X;
  L.callout(b, d1.steps.map((u) => [X1(u) + 1.2, ptop + ph + 2.0]), [X1(40), ptop + ph + mm(1.9)], 'steps', { anchor: 'start', dot: false });
  L.callout(b, [X1(d1.edge_off) + 0.6, ptop + ph - 3.2], [X1(9), ptop + ph - 3.2], 'edge offset', { anchor: 'start', dot: false });
  L.callout(b, [X1(46), Y(LV.core - 0.04)], [X1(54), Y(2.45)], 'model at time *t*', { anchor: 'start', dot: false });
  L.callout(b, [X1(d1.steps[0]) + 0.3, Y(1.02)], [X1(27), Y(0.55)], 'starting profile', { anchor: 'start', dot: false });
  L.text(b, p1.cross[0] - 3, p1.cross[1] - 4.2, 'crossing point', { anchor: 'end', size: S.text.small, color: SHARED });

  // drift offset on profile 2: its plateaus sit above the shared levels
  const p2 = plots.p2, d2 = pr.p2, bxp = p2.X(48);
  L.line(b, [bxp, Y(LV.core)], [bxp, Y(LV.core + d2.drift)], { stroke: VERM, width: 0.6 });
  L.line(b, [bxp - 1.2, Y(LV.core)], [bxp + 1.2, Y(LV.core)], { stroke: VERM, width: 0.6 });
  L.line(b, [bxp - 1.2, Y(LV.core + d2.drift)], [bxp + 1.2, Y(LV.core + d2.drift)], { stroke: VERM, width: 0.6 });
  L.callout(b, [bxp, Y(LV.core + d2.drift / 2)], [p2.X(36), Y(2.55)], 'drift offset', { anchor: 'middle', dot: false });

  // shared conditions: one bracket over both plots
  L.bracket(b, [plots.p1.x0, mm(10.6)], [right, mm(10.6)], { side: 'top', depth: 2.2, stroke: SHARED, width: 0.5 });
  L.text(b, (plots.p1.x0 + right) / 2, mm(8.2), 'one time *t*; same *T*, *P*, *f*O_{2}', { anchor: 'middle', size: S.text.label, color: SHARED });

  // colour key: what the colours mean
  const ky = mm(63.2), kx = plots.p1.x0;
  L.line(b, [kx, ky], [kx + mm(4), ky], { stroke: SHARED, width: 1.0 });
  L.text(b, kx + mm(5), ky, 'shared by the crystal', { valign: 'middle', size: S.text.label, color: SHARED });
  const kx2 = kx + mm(30);
  L.line(b, [kx2, ky], [kx2 + mm(2), ky], { stroke: BLUE, width: 1.0 });
  L.line(b, [kx2 + mm(2), ky], [kx2 + mm(4), ky], { stroke: VERM, width: 1.0 });
  L.text(b, kx2 + mm(5), ky, 'own to each profile', { valign: 'middle', size: S.text.label });

  // =============================== panel c: the joint fit ===============================================
  const c = F.panel('c', { x: 146, y: 0, w: 37, h: 66 });
  heading(c, 'Joint fit (planned)');
  const m = pr.misfit, tau = m.tau;
  const cx0 = mm(5), cw = mm(30.5), ctop = mm(9.5), chh = mm(24);
  const TR = [0.2, 2.0], DM = [0, 7];
  const XT = (v) => cx0 + cw * (v - TR[0]) / (TR[1] - TR[0]);
  const YM = (v) => ctop + chh * (1 - (Math.min(v, DM[1]) - DM[0]) / (DM[1] - DM[0]));
  L.line(c, [cx0, ctop + chh], [cx0 + cw, ctop + chh], { stroke: C.ink, width: 0.5 });
  L.line(c, [cx0, ctop], [cx0, ctop + chh], { stroke: C.ink, width: 0.5 });
  L.text(c, cx0 - mm(1.4), ctop + chh / 2, 'misfit', { anchor: 'middle', valign: 'middle', size: S.text.label, rotate: -90 });
  // true time (synthetic) and the 1-sigma threshold
  L.line(c, [XT(1), ctop], [XT(1), ctop + chh], { stroke: GREY, width: 0.4, dash: '0.5 0.9', cap: 'round' });
  L.line(c, [cx0, YM(1)], [cx0 + cw, YM(1)], { stroke: C.light, width: 0.4, dash: '1.6 1.1' });
  L.text(c, cx0 + 1, YM(1) - 1.6, '1σ', { size: S.text.small, color: C.mid });
  for (const [k, col, w] of [['p2', VERM, 0.8], ['p1', BLUE, 0.8], ['joint', SHARED, 1.1]]) {
    const pts = [];
    tau.forEach((t, i) => { if (t >= TR[0] && t <= TR[1] && m[k].vs_time[i] <= DM[1]) pts.push([XT(t), YM(m[k].vs_time[i])]); });
    L.polyline(c, pts, { stroke: col, width: w, join: 'round' });
  }
  // 1-sigma ranges of time under the axis
  const bars = (yy, key, Xf, lo, hi, arrowLo) => {
    const rows = [['p1', BLUE], ['p2', VERM], ['joint', SHARED]];
    rows.forEach(([k, col], i) => {
      const [a0, a1, best] = m[k][key];
      const y = yy + i * mm(1.7);
      const x0 = Xf(Math.max(a0, lo)), x1 = Xf(Math.min(a1, hi));
      L.line(c, [x0, y], [x1, y], { stroke: col, width: k === 'joint' ? 1.3 : 0.9, cap: 'butt' });
      if (a0 < lo && arrowLo) L.chevron(c, [x0 - 0.6, y], [-1, 0], { len: 2.0, stroke: col, width: 0.7 });
      L.circle(c, Xf(best), y, 0.9, { fill: col, stroke: '#ffffff', width: 0.25 });
    });
  };
  const tb = ctop + chh + mm(2.2);
  bars(tb, 'time_1sigma', XT, TR[0], TR[1], true);
  L.line(c, [XT(1), tb - mm(1)], [XT(1), tb + mm(4.4)], { stroke: GREY, width: 0.4, dash: '0.5 0.9', cap: 'round' });
  [['profile 1', BLUE], ['profile 2', VERM], ['joint (sum)', SHARED]].forEach(([s2, col], i) => L.text(c, XT(0.6) - 1.5, tb + i * mm(1.7), s2, { anchor: 'end', valign: 'middle', size: S.text.small, color: col, weight: i === 2 ? 'bold' : 'normal' }));
  L.text(c, cx0 + cw / 2, tb + mm(8.0), 'time', { anchor: 'middle', size: S.text.label });
  L.text(c, XT(1) + 1, ctop + 3, 'true value', { size: S.text.small, color: C.mid });
  // 1-sigma ranges of the edge Fo
  const ER = [-1.6, 0.8];
  const XE = (v) => cx0 + cw * (v - ER[0]) / (ER[1] - ER[0]);
  const eb = mm(50.5);
  bars(eb, 'edge_1sigma', XE, ER[0], ER[1], true);
  L.text(c, cx0, eb - mm(2.6), '1σ ranges', { size: S.text.small, color: C.mid });
  L.line(c, [XE(0), eb - mm(1)], [XE(0), eb + mm(4.4)], { stroke: GREY, width: 0.4, dash: '0.5 0.9', cap: 'round' });
  L.line(c, [cx0, eb + mm(5.2)], [cx0 + cw, eb + mm(5.2)], { stroke: C.ink, width: 0.5 });
  
  L.text(c, cx0 + cw / 2, eb + mm(9.0), 'edge Fo', { anchor: 'middle', size: S.text.label });
}
