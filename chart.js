/* 지표 계산 + 캔버스 차트 — scan/indi.py를 그대로 옮긴 것입니다 */
(function (g) {
'use strict';

/* ── 지표 ─────────────────────────────────────── */
function sma(v, n) {
  var o = new Array(v.length).fill(null), s = 0, i;
  if (v.length < n) return o;
  for (i = 0; i < n; i++) s += v[i];
  o[n - 1] = s / n;
  for (i = n; i < v.length; i++) { s += v[i] - v[i - n]; o[i] = s / n; }
  return o;
}
function ema(v, n) {
  var o = new Array(v.length).fill(null), i, s = 0;
  if (v.length < n) return o;
  for (i = 0; i < n; i++) s += v[i];
  var k = 2 / (n + 1), p = s / n;
  o[n - 1] = p;
  for (i = n; i < v.length; i++) { p = v[i] * k + p * (1 - k); o[i] = p; }
  return o;
}
function compact(a) { return a.filter(function (x) { return x !== null; }); }
function pad(len, arr) { return new Array(len - arr.length).fill(null).concat(arr); }

function tsi(c, r, s, sig) {
  r = r || 10; s = s || 30; sig = sig || 8;
  var mom = [0], am = [0], i;
  for (i = 1; i < c.length; i++) { mom.push(c[i] - c[i - 1]); am.push(Math.abs(c[i] - c[i - 1])); }
  var e2 = ema(compact(ema(mom, r)), s), a2 = ema(compact(ema(am, r)), s);
  var out = new Array(c.length - e2.length).fill(null);
  for (i = 0; i < e2.length; i++)
    out.push((e2[i] === null || a2[i] === null || a2[i] === 0) ? null : 100 * e2[i] / a2[i]);
  return [out, pad(out.length, ema(compact(out), sig))];
}
function rsi(c, n) {
  n = n || 35;
  var up = [0], dn = [0], i;
  for (i = 1; i < c.length; i++) {
    var d = c[i] - c[i - 1];
    up.push(Math.max(d, 0)); dn.push(Math.max(-d, 0));
  }
  var au = ema(up, n), ad = ema(dn, n), out = [];
  for (i = 0; i < c.length; i++)
    out.push((au[i] === null || ad[i] === null || au[i] + ad[i] === 0) ? null
             : 100 * au[i] / (au[i] + ad[i]));
  return out;
}
function ultimate(h, l, c, p1, p2, p3, sig) {
  p1 = p1 || 7; p2 = p2 || 14; p3 = p3 || 28; sig = sig || 8;
  var n = c.length, bp = [], tr = [], i;
  for (i = 0; i < n; i++) {
    var pc = i ? c[i - 1] : c[0];
    bp.push(c[i] - Math.min(l[i], pc));
    tr.push(Math.max(h[i], pc) - Math.min(l[i], pc));
  }
  function rs(p) {
    var o = new Array(n).fill(null), j;
    for (j = p - 1; j < n; j++) {
      var st = 0, sb = 0, k;
      for (k = j - p + 1; k <= j; k++) { st += tr[k]; sb += bp[k]; }
      o[j] = st ? sb / st : null;
    }
    return o;
  }
  var r1 = rs(p1), r2 = rs(p2), r3 = rs(p3), out = [];
  for (i = 0; i < n; i++)
    out.push((r1[i] === null || r2[i] === null || r3[i] === null) ? null
             : 100 * (4 * r1[i] + 2 * r2[i] + r3[i]) / 7);
  return [out, pad(out.length, sma(compact(out), sig))];
}
function golden(c, n) {
  n = n || 32;
  var buy = [], sell = [], i;
  for (i = 0; i < c.length; i++) {
    var w = c.slice(Math.max(0, i - n + 1), i + 1);
    var lo = Math.min.apply(null, w), hi = Math.max.apply(null, w);
    buy.push(lo ? (c[i] / lo - 1) * 100 : 0);
    sell.push(c[i] ? (hi / c[i] - 1) * 100 : 0);
  }
  return [buy, sell];
}
function mesh(c, start, step, cnt) {
  start = start || 10; step = step || 5; cnt = cnt || 6;
  var mas = [], i;
  for (i = 0; i < cnt; i++) mas.push(sma(c, start + step * i));
  var top = [], bot = [], red = [];
  for (i = 0; i < c.length; i++) {
    var vs = mas.map(function (m) { return m[i]; }).filter(function (x) { return x !== null; });
    if (vs.length === cnt) {
      top.push(Math.max.apply(null, vs)); bot.push(Math.min.apply(null, vs));
      red.push(mas[0][i] > mas[cnt - 1][i]);
    } else { top.push(null); bot.push(null); red.push(false); }
  }
  return [top, bot, red, mas];
}

g.IND = { sma: sma, ema: ema, tsi: tsi, rsi: rsi, ultimate: ultimate,
          golden: golden, mesh: mesh };

/* ── 차트 ─────────────────────────────────────── */
var C = {
  bg: '#0f1115', grid: '#1e222a', axis: '#39404d', text: '#8b93a3',
  up: '#e8544f', dn: '#3f7fd6', vol: '#39404d',
  gb: '#e8544f', gs: '#4a8fe7', t: '#e0b341', b: '#a06fd8', u: '#42b883',
  meshUp: 'rgba(232,84,79,.16)', meshDn: 'rgba(63,127,214,.16)',
  ma: ['#e8b04a', '#5aa9e6', '#c77dd6', '#5fce9e', '#dd6a6a']
};

function ext(arrs) {
  var lo = Infinity, hi = -Infinity;
  arrs.forEach(function (a) {
    a.forEach(function (x) {
      if (x === null || x === undefined || isNaN(x)) return;
      if (x < lo) lo = x; if (x > hi) hi = x;
    });
  });
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  return [lo, hi];
}

function drawChart(cv, d, title) {
  var n = d.c.length;
  if (!n) return;
  var W = cv.width, H = cv.height, x = cv.getContext('2d');
  x.fillStyle = C.bg; x.fillRect(0, 0, W, H);

  var G = golden(d.c, 32), T = tsi(d.c), B = rsi(d.c, 35),
      U = ultimate(d.h, d.l, d.c), M = mesh(d.c);
  var L = 52, R = 10, TOP = 20, GAP = 7;
  // 6단: 가격 / 거래량 / U / T / B / G
  var hs = [0.40, 0.10, 0.115, 0.115, 0.115, 0.155];
  var avail = H - TOP - 16 - GAP * 5, ys = [], acc = TOP;
  hs.forEach(function (r) { ys.push([acc, avail * r]); acc += avail * r + GAP; });
  var PW = W - L - R, bw = PW / n, cw = Math.max(1, Math.min(bw * 0.7, 9));

  function panel(i) { return { y: ys[i][0], h: ys[i][1] }; }
  function px(i) { return L + bw * (i + 0.5); }

  function frame(p, label) {
    x.strokeStyle = C.axis; x.lineWidth = 1;
    x.strokeRect(L + .5, p.y + .5, PW, p.h);
    x.fillStyle = C.text; x.font = '10px sans-serif'; x.textAlign = 'left';
    x.fillText(label, L + 5, p.y + 11);
  }
  function scaler(p, lo, hi) {
    return function (v) { return p.y + p.h - (v - lo) / (hi - lo) * p.h; };
  }
  function gridline(p, lo, hi, vals) {
    var Y = scaler(p, lo, hi);
    x.strokeStyle = C.grid; x.setLineDash([2, 3]); x.lineWidth = 1;
    x.fillStyle = C.text; x.font = '9px sans-serif'; x.textAlign = 'right';
    vals.forEach(function (v) {
      if (v < lo || v > hi) return;
      var yy = Math.round(Y(v)) + .5;
      x.beginPath(); x.moveTo(L, yy); x.lineTo(L + PW, yy); x.stroke();
      x.fillText(String(v), L - 4, yy + 3);
    });
    x.setLineDash([]);
  }
  function line(p, arr, lo, hi, col, wd) {
    var Y = scaler(p, lo, hi), started = false;
    x.strokeStyle = col; x.lineWidth = wd || 1.2; x.beginPath();
    for (var i = 0; i < n; i++) {
      var v = arr[i];
      if (v === null || v === undefined || isNaN(v)) { started = false; continue; }
      var xx = px(i), yy = Y(v);
      if (!started) { x.moveTo(xx, yy); started = true; } else x.lineTo(xx, yy);
    }
    x.stroke();
  }

  /* 1단 — 가격 + 그물망 */
  var p0 = panel(0);
  var pr = ext([d.h, d.l, M[0], M[1]]);
  var lo = pr[0] - (pr[1] - pr[0]) * 0.04, hi = pr[1] + (pr[1] - pr[0]) * 0.06;
  var Y0 = scaler(p0, lo, hi);
  frame(p0, title);
  // 그물망 띠
  for (var i = 1; i < n; i++) {
    if (M[0][i] === null || M[0][i - 1] === null) continue;
    x.fillStyle = M[2][i] ? C.meshUp : C.meshDn;
    x.beginPath();
    x.moveTo(px(i - 1), Y0(M[0][i - 1])); x.lineTo(px(i), Y0(M[0][i]));
    x.lineTo(px(i), Y0(M[1][i])); x.lineTo(px(i - 1), Y0(M[1][i - 1]));
    x.closePath(); x.fill();
  }
  M[3].forEach(function (m, k) { line(p0, m, lo, hi, C.ma[k % C.ma.length], 0.9); });
  // 캔들
  for (i = 0; i < n; i++) {
    var up = d.c[i] >= d.o[i], col = up ? C.up : C.dn, cx = px(i);
    x.strokeStyle = col; x.fillStyle = col; x.lineWidth = 1;
    x.beginPath(); x.moveTo(cx, Y0(d.h[i])); x.lineTo(cx, Y0(d.l[i])); x.stroke();
    var a = Y0(d.o[i]), b2 = Y0(d.c[i]);
    var top = Math.min(a, b2), hgt = Math.max(Math.abs(a - b2), 1);
    if (up) x.fillRect(cx - cw / 2, top, cw, hgt);
    else x.strokeRect(cx - cw / 2, top, cw, hgt);
  }
  // 현재가
  x.fillStyle = C.text; x.font = '10px sans-serif'; x.textAlign = 'right';
  x.fillText(d.c[n - 1].toLocaleString(), L - 4, Y0(d.c[n - 1]) + 3);

  /* 2단 — 거래량 */
  var p1 = panel(1), vh = ext([d.v])[1];
  frame(p1, '거래량');
  for (i = 0; i < n; i++) {
    var hh = (d.v[i] / vh) * (p1.h - 14);
    x.fillStyle = d.c[i] >= d.o[i] ? C.up : C.dn;
    x.globalAlpha = .65;
    x.fillRect(px(i) - cw / 2, p1.y + p1.h - hh, cw, hh);
    x.globalAlpha = 1;
  }

  /* 3단 — U */
  var p2 = panel(2);
  frame(p2, 'U (Ultimate 7,14,28)');
  gridline(p2, 0, 100, [30, 50, 70]);
  line(p2, U[0], 0, 100, C.u, 1.3); line(p2, U[1], 0, 100, '#6b7686', 1);

  /* 4단 — T */
  var p3 = panel(3), te = ext([T[0], T[1]]);
  frame(p3, 'T (TSI 10,30)  기준 10');
  gridline(p3, te[0], te[1], [0, 10]);
  line(p3, T[0], te[0], te[1], C.t, 1.3); line(p3, T[1], te[0], te[1], '#6b7686', 1);

  /* 5단 — B */
  var p4 = panel(4), be = ext([B]);
  frame(p4, 'B (BPDI Hilo)  기준 55');
  gridline(p4, be[0], be[1], [50, 55]);
  line(p4, B, be[0], be[1], C.b, 1.3);

  /* 6단 — G */
  var p5 = panel(5), ge = ext([G[0], G[1]]);
  frame(p5, 'G  적선=Buy · 청선=Sell  기준 5');
  gridline(p5, ge[0], ge[1], [0, 5]);
  line(p5, G[0], ge[0], ge[1], C.gb, 1.3);
  line(p5, G[1], ge[0], ge[1], C.gs, 1.3);

  /* 날짜축 */
  x.fillStyle = C.text; x.font = '9px sans-serif'; x.textAlign = 'center';
  var step = Math.max(1, Math.floor(n / 8));
  for (i = 0; i < n; i += step) x.fillText(d.t[i], px(i), H - 4);
}

g.drawChart = drawChart;
})(window);
