// FORSTER workflow schematic. Every curve is illustrative, computed by figure/data/make_curves.py.
export const figure = { width: 183, height: 112, name: 'forster-workflow', format: 'journal' };

export default async function draw(F, L) {
  const D = await (await fetch('/figure/data/curves.json')).json();
  const C = L.STYLE.color, LW = L.STYLE.line;
  const S = { ...L.STYLE.text, small: 6, label: 6.5, title: 7 };
  const INK = '#2b2b2b', GREY = '#8a8a8a';
  const ORANGE = '#E69F00', ORANGE_D = '#B57A00';
  const BLUE = '#0072B2', T_COL = ['#7fbfe6', BLUE, '#0b2f57'];
  const GREEN = '#009E73', GREEN_L = '#bfe6d8';
  const BAND = '#d6e8f5';
  const sub = (t) => t; // labels are plain strings with library markup

  const heading = (P, str) => L.text(P, 10.5, 7.4, str, { size: S.title, weight: 'bold', role: 'heading' });

  // small schematic axes (no ticks): returns mappers
  const axes = (P, x, y, w, h, xr, yr, xlabel, ylabel) => {
    const pl = L.plot(P, x, y, w, h, { xr, yr, xlabel, ylabel, series: [], labelSize: S.small });
    return pl;
  };
  const line = (P, M, xs, ys, s) => L.polyline(P, xs.map((v, i) => [M.X(v), M.Y(ys[i])]), { cap: 'round', join: 'round', ...s });
  const dots = (P, M, xs, ys, r = 1.15, fill = INK) => xs.forEach((v, i) => L.circle(P, M.X(v), M.Y(ys[i]), r, { fill, stroke: '#fff', width: 0.3 }));
  const clipX = (xmax) => {
    const keep = D.x.map((v) => v <= xmax);
    return (arr) => arr.filter((_, i) => keep[i]);
  };

  const XR = [0, 34], YR = [0.786, 0.894];
  const cx = clipX(XR[1]);
  const xs = cx(D.x);
  const step = (x0, fa, fb, M, P, s) => L.polyline(P, [[M.X(0), M.Y(fa)], [M.X(x0), M.Y(fa)], [M.X(x0), M.Y(fb)], [M.X(XR[1]), M.Y(fb)]], { join: 'miter', ...s });
  const { fo_a: FA, fo_b: FB, x0: X0 } = D.params;

  // ===================================================================== a: inputs
  const a = F.panel('a', { x: 1, y: 1, w: 62, h: 52 });
  heading(a, 'Inputs');
  {
    // generic olivine section (no habit claimed): convex polygon, darker rim band = zoning
    const cxs = 40, cys = 58;
    const outer = [[-31, -6], [-20, -24], [16, -25], [31, -9], [29, 10], [14, 24], [-18, 23], [-31, 8]].map(([u, v]) => [cxs + u, cys + v]);
    const inner = outer.map(([x, y]) => [cxs + (x - cxs) * 0.72, cys + (y - cys) * 0.68]);
    L.polygon(a, outer, { fill: '#e2d79a', stroke: '#8f8a4e', width: LW.thin, join: 'round' });
    L.polygon(a, inner, { fill: '#c9dca0', stroke: 'none' });
    L.text(a, cxs + 6, cys - 14, 'olivine', { size: S.small, anchor: 'middle', color: '#5d6a33', italic: true });
    // traverse: from the left edge (rim) inward
    const ty = cys + 1, tx0 = cxs - 30.2, tx1 = cxs + 2;
    L.line(a, [tx0, ty], [tx1, ty], { stroke: INK, width: LW.hair });
    const ns = D.spots.length; // same count as the measured points in the plot
    for (let k = 0; k < ns; k++) L.circle(a, tx0 + 1.2 + k * ((tx1 - tx0 - 1.2) / (ns - 1)), ty, 0.75, { fill: INK });
    L.callout(a, [tx0 + 7.5, ty - 1.2], [4, 22], 'microprobe spots', { size: S.small, anchor: 'start', dot: false });
    // crystal axes triad (generic, not tied to faces)
    const ox = 10, oy = 104;
    L.arrow(a, [ox, oy], [ox + 13, oy], { width: LW.thin, head: 2.2, stroke: GREY });
    L.arrow(a, [ox, oy], [ox, oy - 13], { width: LW.thin, head: 2.2, stroke: GREY });
    L.arrow(a, [ox, oy], [ox + 8, oy + 7], { width: LW.thin, head: 2.2, stroke: GREY });
    L.text(a, ox + 14.5, oy, '[100]', { size: S.small, valign: 'middle', color: '#555' });
    L.text(a, ox, oy - 14.5, '[001]', { size: S.small, anchor: 'middle', valign: 'bottom', color: '#555' });
    L.text(a, ox + 9, oy + 8.5, '[010]', { size: S.small, valign: 'top', color: '#555' });
    L.text(a, ox - 2, oy + 25, 'orientation', { size: S.small, color: '#555' });
    // conditions chips
    const chips = ['*T*', '*P*', '*f*O_{2}'];
    chips.forEach((t, i) => {
      const bx = 45 + i * 13.5, by = 101;
      L.rect(a, bx, by - 5.5, 12, 11, { r: 2, fill: '#e6f5ef', stroke: GREEN, width: LW.thin });
      L.text(a, bx + 6, by, t, { size: S.label, anchor: 'middle', valign: 'middle', color: '#006b4f' });
    });

    // measured profile with the starting profile
    const M = axes(a, 96, 24, 74, 82, XR, YR, 'distance from rim', 'Fo');
    step(X0, FA, FB, M, a, { stroke: ORANGE, width: LW.bold });
    dots(a, M, D.spots, D.data);
    L.text(a, M.X(X0 * 0.5), M.Y(FA) - 3, 'segment a', { size: S.small, anchor: 'middle', valign: 'bottom', color: ORANGE_D });
    L.text(a, M.X(X0 + 7.5), M.Y(FB) - 3, 'segment b', { size: S.small, anchor: 'middle', valign: 'bottom', color: ORANGE_D });
    L.callout(a, [M.X(X0), M.Y((FA + FB) / 2) + 6], [M.X(X0) + 12, M.Y(FA) + 6], 'step', { size: S.small, anchor: 'start', color: ORANGE_D, textColor: ORANGE_D, dot: false });
    L.text(a, M.X(XR[1]), 130, '{#B57A00|starting profile}, measured Fo', { size: S.small, anchor: 'end' });
    // traverse → profile
    L.arrow(a, [cxs + 33, ty], [93, ty], { width: LW.thin, stroke: GREY, head: 2.4 });
    F.anchor('a.out', a.node, a.w, 66);
  }

  // ===================================================================== b: forward model
  const b = F.panel('b', { x: 67, y: 1, w: 57, h: 52 });
  heading(b, 'Forward model in time');
  let MB;
  {
    MB = axes(b, 22, 24, 128, 92, XR, YR, 'distance from rim', 'Fo');
    step(X0, FA, FB, MB, b, { stroke: ORANGE, width: LW.normal, dash: '2 1.2' });
    L.text(b, MB.X(X0) - 2, MB.Y(FB), 'starting profile', { size: S.small, anchor: 'end', valign: 'middle', color: ORANGE_D });
    D.t_show.forEach((t, i) => line(b, MB, xs, cx(D.curves[String(t)]), { stroke: T_COL[i], width: i === 1 ? LW.bold * 1.15 : LW.normal }));
    dots(b, MB, D.spots, D.data);
    // labels on the curves at the core side
    // time arrow across the core side of the front at one Fo level; a dot where it meets each curve (as in c)
    const FoA = 0.869, xa0 = X0 + 0.6, xa1 = X0 + 13.5, ya = MB.Y(FoA);
    L.arrow(b, [MB.X(xa0), ya], [MB.X(xa1), ya], { width: LW.thin, stroke: INK, head: 2.4 });
    L.text(b, MB.X(xa1) + 1.5, ya, 'time', { size: S.small, valign: 'middle' });
    D.t_show.forEach((t, i) => {
      const cv = D.curves[String(t)];
      const j = D.x.findIndex((v, k) => v > X0 && cv[k] >= FoA);
      L.circle(b, MB.X(D.x[j]), ya, 1.5, { fill: T_COL[i], stroke: '#fff', width: 0.4 });
      L.text(b, MB.X(D.x[j]) + 1.8, ya + 3.6, `*t*_{${i + 1}}`, { size: S.small, color: T_COL[i], valign: 'top', anchor: 'start', halo: true });
    });
    // diffusion length at t2: from the step to step + sqrt(D t)
    const s2 = Math.sqrt(D.t_show[1]), yb = MB.Y(0.80) - 6;
    L.arrow(b, [MB.X(X0), yb], [MB.X(X0 + s2), yb], { width: LW.thin, stroke: BLUE, head: 2, both: true });
    L.text(b, MB.X(X0 + s2) + 2, yb, '√(*D* *t*)', { size: S.small, anchor: 'start', valign: 'middle', color: BLUE });
    F.anchor('b.in', b.node, 0, 66);
    F.anchor('b.out', b.node, b.w, 66);
  }

  // ===================================================================== c: misfit against time
  const c = F.panel('c', { x: 127, y: 1, w: 55, h: 52 });
  heading(c, 'Misfit against time');
  {
    const TR = [0, 36], MR = [0, 30];
    const M = axes(c, 20, 22, 128, 58, TR, MR, '', 'misfit χ^{2}');
    L.rect(c, M.X(D.t_lo), M.Y(MR[1]), M.X(D.t_hi) - M.X(D.t_lo), M.Y(0) - M.Y(MR[1]), { fill: BAND });
    const keep = D.times.map((t, i) => t <= TR[1] && D.misfit[i] <= MR[1]);
    const tt = D.times.filter((_, i) => keep[i]), mm = D.misfit.filter((_, i) => keep[i]);
    line(c, M, tt, mm, { stroke: INK, width: LW.normal });
    L.line(c, [M.X(D.t_best), M.Y(0)], [M.X(D.t_best), M.Y(MR[1])], { stroke: BLUE, width: LW.thin, dash: '1.6 1' });
    D.t_show.forEach((t, i) => { if (t <= TR[1]) L.circle(c, M.X(t), M.Y(Math.min(D.misfit_at_show[i], MR[1])), 1.7, { fill: T_COL[i], stroke: '#fff', width: 0.4 }); });
    L.text(c, M.X(D.t_show[0]) + 2.6, M.Y(D.misfit_at_show[0]), '*t*_{1}', { size: S.small, color: T_COL[0], valign: 'middle' });
    L.text(c, M.X(D.t_show[2]), M.Y(D.misfit_at_show[2]) + 3, '*t*_{3}', { size: S.small, color: T_COL[2], valign: 'top', anchor: 'middle', halo: true });
    L.text(c, M.X(TR[1]) + 2, M.Y(0) + 2, 'time', { size: S.small, anchor: 'end', valign: 'top' });
    L.callout(c, [M.X(D.t_best), M.Y(0) - 1], [M.X(D.t_best) + 18, M.Y(8)], 'best-fit time', { size: S.small, anchor: 'start', color: BLUE, textColor: BLUE, dot: false });
    L.text(c, (M.X(D.t_lo) + M.X(D.t_hi)) / 2, M.Y(MR[1]) - 1.5, '1σ range', { size: S.small, anchor: 'middle', valign: 'bottom', color: BLUE });

    // noise model: residuals at the best fit = white scatter + smooth correlated part
    const RY = [-0.012, 0.012];
    const R = axes(c, 20, 104, 128, 24, XR, RY, 'distance from rim', '');
    L.line(c, [R.X(0), R.Y(0)], [R.X(XR[1]), R.Y(0)], { stroke: C.light, width: LW.hair });
    line(c, R, D.spots, D.smooth, { stroke: '#4d7ea8', width: LW.normal });
    dots(c, R, D.spots, D.residuals, 1.0, INK);
    L.text(c, 21, 95, 'noise model', { size: S.small });
    const iw = D.residuals.indexOf(Math.min(...D.residuals));
    L.callout(c, [R.X(D.spots[iw]) + 1.2, R.Y(D.residuals[iw])], [R.X(D.spots[iw]) + 10, R.Y(-0.011)], 'white', { size: S.small, anchor: 'start', dot: false, halo: true });
    const ic = D.spots.length - 4;
    L.callout(c, [R.X(D.spots[ic]), R.Y(D.smooth[ic]) - 0.8], [R.X(D.spots[ic]) + 2, R.Y(0.0115)], '{#4d7ea8|correlated}', { size: S.small, anchor: 'start', dot: false, color: '#4d7ea8' });
    F.anchor('c.in', c.node, 0, 66);
    F.anchor('c.out', c.node, 74, c.h + 1);
  }

  // ===================================================================== d: uncertain starting profile
  const d = F.panel('d', { x: 1, y: 58, w: 62, h: 53 });
  heading(d, 'Uncertain starting profile');
  {
    L.text(d, 10.5, 15.5, '(optional; fitted with the time)', { size: S.small, color: '#555' });
    const M = axes(d, 24, 26, 136, 88, XR, YR, 'distance from rim', 'Fo');
    for (const [x0, fb] of D.ic_alternatives) step(x0, FA, fb, M, d, { stroke: ORANGE, width: LW.thin, opacity: 0.45 });
    step(X0, FA, FB, M, d, { stroke: ORANGE, width: LW.bold });
    line(d, M, xs, cx(D.curves[String(D.t_show[1])]), { stroke: BLUE, width: LW.normal });
    dots(d, M, D.spots, D.data);
    // priors as double arrows
    const ya = M.Y(FA) + 6;
    L.arrow(d, [M.X(X0 - 4.2), ya], [M.X(X0 + 4.2), ya], { width: LW.thin, stroke: ORANGE_D, head: 2, both: true });
    L.text(d, M.X(X0 + 4.2) + 2, ya, 'step position', { size: S.small, valign: 'middle', color: ORANGE_D });
    const xb = M.X(29.5);
    L.arrow(d, [xb, M.Y(FB + 0.0075)], [xb, M.Y(FB - 0.0075)], { width: LW.thin, stroke: ORANGE_D, head: 2, both: true });
    L.text(d, xb - 2.5, M.Y(FB - 0.016), 'segment Fo', { size: S.small, anchor: 'middle', valign: 'top', color: ORANGE_D });
    F.anchor('d.in', d.node, 74, -1);
    F.anchor('d.out', d.node, d.w, 70);
  }

  // ===================================================================== e: Monte Carlo
  const e = F.panel('e', { x: 67, y: 58, w: 57, h: 53 });
  heading(e, 'Monte Carlo');
  {
    const gs = [['*T*', 0], ['*P*', 1], ['*f*O_{2}', 2], ['law parameters', 3]];
    const gx = 6, gw = 24;
    gs.forEach(([t, i]) => {
      const gy = 34 + i * 22;
      const pts = L.sample((u) => gy - 11 * Math.exp(-0.5 * ((u - 0.5) / 0.16) ** 2), 0, 1, 50).map(([u, v]) => [gx + u * gw, v]);
      L.polygon(e, [[gx, gy], ...pts, [gx + gw, gy]], { fill: GREEN_L, stroke: 'none' });
      L.polyline(e, pts, { stroke: GREEN, width: LW.normal });
      L.line(e, [gx, gy], [gx + gw, gy], { stroke: GREY, width: LW.hair });
      L.text(e, gx + gw + 2, gy - 2, t, { size: S.small, color: '#006b4f', valign: 'bottom' });
    });
    // spread in log D maps to spread in log t (t = (D t)_fit / D)
    const ax0 = 97, ax1 = 145, yD = 40, yT = 96;
    L.line(e, [ax0, yD], [ax1, yD], { stroke: INK, width: LW.thin });
    L.line(e, [ax0, yT], [ax1, yT], { stroke: INK, width: LW.thin });
    L.text(e, ax1 + 1.5, yD, 'log *D*', { size: S.small, valign: 'middle' });
    L.text(e, ax1 + 1.5, yT, 'log *t*', { size: S.small, valign: 'middle' });
    const mid = (ax0 + ax1) / 2, k = 64; // pt per log10 unit
    for (const g of D.draws_logD) {
      const xd = mid + g * k, xt = mid - g * k;
      L.line(e, [xd, yD], [xt, yT], { stroke: GREY, width: LW.hair, opacity: 0.8 });
      L.line(e, [xd, yD - 3], [xd, yD + 0.0], { stroke: GREEN, width: LW.normal });
      L.line(e, [xt, yT], [xt, yT + 3], { stroke: BLUE, width: LW.normal });
    }
    L.brace(e, [80, 22], [80, 100], { side: -1, depth: 4, stroke: GREY, width: LW.thin });
    L.arrow(e, [86, 61], [ax0 - 2, yD + 2], { width: LW.thin, stroke: GREY, head: 2.4 });
    L.text(e, mid, yT + 12, '*t* = (*D* *t*)_{fit} / *D*', { size: S.label, anchor: 'middle', valign: 'top' });
    L.text(e, mid, (yD + yT) / 2, 'faster *D*,\nshorter *t*', { size: S.small, anchor: 'middle', valign: 'middle', halo: true });
    F.anchor('e.in', e.node, 0, 70);
    F.anchor('e.out', e.node, e.w, 70);
  }

  // ===================================================================== f: time distribution
  const f = F.panel('f', { x: 127, y: 58, w: 55, h: 53 });
  heading(f, 'Time distribution');
  {
    const LR = [0.6, 1.72];
    const M = axes(f, 14, 26, 134, 86, LR, [0, 1.12], 'time (log scale)', 'probability');
    const keep = D.logt.map((v) => v >= LR[0] && v <= LR[1]);
    const lt = D.logt.filter((_, i) => keep[i]);
    const sel = (arr) => arr.filter((_, i) => keep[i]);
    const mc = sel(D.dist_mc);
    L.polygon(f, [[M.X(lt[0]), M.Y(0)], ...lt.map((v, i) => [M.X(v), M.Y(mc[i])]), [M.X(lt[lt.length - 1]), M.Y(0)]], { fill: GREEN_L, stroke: 'none' });
    line(f, M, lt, mc, { stroke: GREEN, width: LW.normal });
    line(f, M, lt, sel(D.dist_ic), { stroke: ORANGE, width: LW.normal * 1.3 });
    line(f, M, lt, sel(D.dist_fit), { stroke: BLUE, width: LW.normal * 1.3 });
    const [p16, p50, p84] = D.pct_mc;
    L.line(f, [M.X(p50), M.Y(0)], [M.X(p50), M.Y(1.06)], { stroke: INK, width: LW.thin, dash: '1.6 1' });
    L.text(f, M.X(p50), M.Y(1.07), 'median', { size: S.small, anchor: 'middle', valign: 'bottom' });
    const yb = M.Y(0) - 9;
    L.arrow(f, [M.X(p16), yb], [M.X(p84), yb], { width: LW.thin, stroke: '#006b4f', head: 2, both: true });
    L.text(f, (M.X(p16) + M.X(p84)) / 2, yb + 1.6, '68% range', { size: S.small, anchor: 'middle', valign: 'top', color: '#006b4f', halo: true });
    // direct labels
    const pk = (arr) => { const a = sel(arr); const i = a.indexOf(Math.max(...a)); return [lt[i], a[i]]; };
    const [xf] = pk(D.dist_fit);
    L.callout(f, [M.X(xf - 0.07), M.Y(0.86)], [M.X(LR[0]) + 2, M.Y(1.0)], '{#0072B2|fit alone}', { size: S.small, anchor: 'start', dot: false, color: BLUE });
    L.callout(f, [M.X(xf + 0.12), M.Y(0.62)], [M.X(1.36), M.Y(0.86)], '{#B57A00|+ starting profile}', { size: S.small, anchor: 'start', dot: false, color: ORANGE_D });
    L.callout(f, [M.X(xf + 0.2), M.Y(0.42)], [M.X(1.42), M.Y(0.56)], '{#006b4f|+ *T*, *P*, *f*O_{2}, law}', { size: S.small, anchor: 'start', dot: false, color: GREEN });
    F.anchor('f.in', f.node, 0, 70);
  }

  // ===================================================================== flow between panels
  const flow = { stroke: GREY, width: LW.normal, head: 3, kind: 'flow' };
  F.relate('a.out', 'b.in', 'flow'); L.connect(F, 'a.out', 'b.in', flow);
  F.relate('b.out', 'c.in', 'flow'); L.connect(F, 'b.out', 'c.in', flow);
  F.relate('c.out', 'd.in', 'flow');
  { const s = F.at('c.out'), t = F.at('d.in'), ym = (s.y + t.y) / 2; L.connect(F, 'c.out', 'd.in', { ...flow, via: [[s.x, ym], [t.x, ym]], radius: 3 }); }
  F.relate('d.out', 'e.in', 'flow'); L.connect(F, 'd.out', 'e.in', flow);
  F.relate('e.out', 'f.in', 'flow'); L.connect(F, 'e.out', 'f.in', flow);
}
