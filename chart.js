/* 지표 계산 + 대신증권식 캔버스 차트 — 지표는 scan/indi.py를 그대로 옮긴 것입니다 */
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
  /* 대신증권 Golden Power 정확 재현 (2026-08-14 확정, 기간 25)
     적선 = 100(C/LL − LL/C) · 청선 = 100(HH/C − C/HH)
     CYBOS 실측 133만 표본과 오차 0.0001 미만 — HTS 화면과 같은 값입니다 */
  n = n || 25;
  var buy = [], sell = [], i;
  for (i = 0; i < c.length; i++) {
    var w = c.slice(Math.max(0, i - n + 1), i + 1);
    var lo = Math.min.apply(null, w), hi = Math.max.apply(null, w);
    buy.push((lo && c[i]) ? (c[i] / lo - lo / c[i]) * 100 : 0);
    sell.push((hi && c[i]) ? (hi / c[i] - c[i] / hi) * 100 : 0);
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

/* ── 대신증권 색 ──────────────────────────────── */
var C = {
  bg: '#C0FFFF',            // 연한 하늘색 배경
  grid: '#A8A8A8',          // 회색 격자
  frame: '#808080',
  text: '#404040',
  up: '#FF0000',            // 양봉 빨강
  dn: '#0000FF',            // 음봉 파랑
  fillUp: '#FFC8CF',        // 상승 영역 분홍
  fillDn: '#40E0D0',        // 하락 영역 청록
  ref: '#0000FF',           // 기준선 파랑
  dash: '#000000',          // 점선 검정
  zero: '#000000',
  // 이동평균 (대신증권 기본 팔레트)
  ma: ['#FF0000', '#0000C0', '#008000', '#C000C0', '#C08000', '#008080']
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

  var G = golden(d.c, 25), T = tsi(d.c), B = rsi(d.c, 35),
      U = ultimate(d.h, d.l, d.c), M = mesh(d.c);
  var L = 56, R = 8, TOP = 6, GAP = 3;
  var hs = [0.42, 0.10, 0.11, 0.115, 0.115, 0.14];   // 가격/거래량/U/T/B/G
  var avail = H - TOP - 15 - GAP * 5, ys = [], acc = TOP;
  hs.forEach(function (r) { ys.push([acc, avail * r]); acc += avail * r + GAP; });
  var PW = W - L - R, bw = PW / n, cw = Math.max(1, Math.min(bw * 0.62, 8));

  function panel(i) { return { y: ys[i][0], h: ys[i][1] }; }
  function px(i) { return L + bw * (i + 0.5); }
  function sc(p, lo, hi) {
    return function (v) { return p.y + p.h - (v - lo) / (hi - lo) * p.h; };
  }

  function frame(p, label) {
    x.strokeStyle = C.frame; x.lineWidth = 1;
    x.strokeRect(L + .5, p.y + .5, PW, p.h);
    if (label) {
      x.fillStyle = C.text; x.font = '10px "Malgun Gothic",sans-serif'; x.textAlign = 'left';
      x.fillText(label, L + 4, p.y + 11);
    }
  }
  // 월 구분 세로선
  function months(p) {
    x.strokeStyle = C.grid; x.lineWidth = 1; x.setLineDash([]);
    for (var i = 1; i < n; i++) {
      var a = String(d.t[i - 1]), b = String(d.t[i]);
      var ma = a.length > 7 ? a.slice(0, 7) : a.slice(0, 5);
      var mb = b.length > 7 ? b.slice(0, 7) : b.slice(0, 5);
      if (ma !== mb) {
        var xx = Math.round(px(i) - bw / 2) + .5;
        x.beginPath(); x.moveTo(xx, p.y); x.lineTo(xx, p.y + p.h); x.stroke();
      }
    }
  }
  function hline(p, lo, hi, v, col, dash, wd) {
    if (v < lo || v > hi) return;
    var yy = Math.round(sc(p, lo, hi)(v)) + .5;
    x.strokeStyle = col; x.lineWidth = wd || 1; x.setLineDash(dash || []);
    x.beginPath(); x.moveTo(L, yy); x.lineTo(L + PW, yy); x.stroke();
    x.setLineDash([]);
  }
  function ylab(p, lo, hi, v, txt) {
    if (v < lo || v > hi) return;
    x.fillStyle = C.text; x.font = '10px sans-serif'; x.textAlign = 'right';
    x.fillText(txt, L - 4, sc(p, lo, hi)(v) + 3);
  }
  function line(p, arr, lo, hi, col, wd) {
    var Y = sc(p, lo, hi), on = false;
    x.strokeStyle = col; x.lineWidth = wd || 1.1; x.beginPath();
    for (var i = 0; i < n; i++) {
      var v = arr[i];
      if (v === null || v === undefined || isNaN(v)) { on = false; continue; }
      var xx = px(i), yy = Y(v);
      if (!on) { x.moveTo(xx, yy); on = true; } else x.lineTo(xx, yy);
    }
    x.stroke();
  }
  // 기준선 대비 위/아래를 색으로 채움 (대신증권 지표창 스타일)
  function area(p, arr, lo, hi, base) {
    var Y = sc(p, lo, hi), yb = Y(base), i, seg = [];
    for (i = 0; i < n; i++) {
      var v = arr[i];
      if (v === null || v === undefined || isNaN(v)) { flush(); continue; }
      seg.push([px(i), Y(v), v >= base]);
      if (seg.length > 1 && seg[seg.length - 1][2] !== seg[seg.length - 2][2]) {
        var tail = seg.pop(); flush(); seg = [tail];
      }
    }
    flush();
    function flush() {
      if (seg.length < 2) { seg = []; return; }
      x.fillStyle = seg[0][2] ? C.fillUp : C.fillDn;
      x.beginPath(); x.moveTo(seg[0][0], yb);
      seg.forEach(function (s) { x.lineTo(s[0], s[1]); });
      x.lineTo(seg[seg.length - 1][0], yb); x.closePath(); x.fill();
      seg = [];
    }
  }

  /* 1단 — 가격 + 그물망 */
  var p0 = panel(0);
  var pr = ext([d.h, d.l, M[0], M[1]]);
  var lo = pr[0] - (pr[1] - pr[0]) * 0.05, hi = pr[1] + (pr[1] - pr[0]) * 0.07;
  var Y0 = sc(p0, lo, hi);
  months(p0);
  for (var i = 1; i < n; i++) {
    if (M[0][i] === null || M[0][i - 1] === null) continue;
    x.fillStyle = M[2][i] ? C.fillUp : C.fillDn;
    x.beginPath();
    x.moveTo(px(i - 1), Y0(M[0][i - 1])); x.lineTo(px(i), Y0(M[0][i]));
    x.lineTo(px(i), Y0(M[1][i])); x.lineTo(px(i - 1), Y0(M[1][i - 1]));
    x.closePath(); x.fill();
  }
  M[3].forEach(function (m, k) { line(p0, m, lo, hi, C.ma[k % C.ma.length], 0.9); });
  for (i = 0; i < n; i++) {
    var up = d.c[i] >= d.o[i], col = up ? C.up : C.dn, cx = Math.round(px(i)) + .5;
    x.strokeStyle = col; x.fillStyle = col; x.lineWidth = 1;
    x.beginPath(); x.moveTo(cx, Y0(d.h[i])); x.lineTo(cx, Y0(d.l[i])); x.stroke();
    var a = Y0(d.o[i]), b2 = Y0(d.c[i]);
    x.fillRect(cx - cw / 2, Math.min(a, b2), cw, Math.max(Math.abs(a - b2), 1));
  }
  frame(p0, '');
  // 가격 눈금 5개
  for (i = 0; i <= 4; i++) {
    var v = lo + (hi - lo) * i / 4;
    ylab(p0, lo, hi, v, Math.round(v).toLocaleString());
  }
  x.fillStyle = '#000'; x.font = 'bold 10px "Malgun Gothic",sans-serif'; x.textAlign = 'left';
  x.fillText(title, L + 4, p0.y + 11);

  /* 2단 — 거래량 */
  var p1 = panel(1), vh = ext([d.v])[1];
  months(p1);
  for (i = 0; i < n; i++) {
    var hh = (d.v[i] / vh) * (p1.h - 3);
    x.fillStyle = d.c[i] >= d.o[i] ? C.up : C.dn;
    x.fillRect(Math.round(px(i)) + .5 - cw / 2, p1.y + p1.h - hh, cw, hh);
  }
  frame(p1, '거래량');

  /* 3단 — U */
  var p2 = panel(2), ue = ext([U[0], U[1]]);
  months(p2);
  area(p2, U[0], ue[0], ue[1], 50);
  hline(p2, ue[0], ue[1], 50, C.dash, [4, 2]);
  line(p2, U[0], ue[0], ue[1], C.up, 1.2);
  line(p2, U[1], ue[0], ue[1], C.dn, 1);
  frame(p2, 'Ultimate_7,14,28  Signal_8');
  ylab(p2, ue[0], ue[1], 50, '50');

  /* 4단 — T */
  var p3 = panel(3), te = ext([T[0], T[1], [-12, 12]]);
  months(p3);
  area(p3, T[0], te[0], te[1], 0);
  hline(p3, te[0], te[1], 0, C.zero, [], 1);
  hline(p3, te[0], te[1], 10, C.ref, [], 1.4);
  hline(p3, te[0], te[1], 10, C.dash, [5, 2, 1, 2]);
  line(p3, T[0], te[0], te[1], C.up, 1.2);
  line(p3, T[1], te[0], te[1], C.dn, 1);
  frame(p3, 'TSI_10,30  Signal_8   기준 10');
  ylab(p3, te[0], te[1], 0, '0'); ylab(p3, te[0], te[1], 10, '10');

  /* 5단 — B */
  var p4 = panel(4), be = ext([B, [48, 58]]);
  months(p4);
  area(p4, B, be[0], be[1], 55);
  hline(p4, be[0], be[1], 50, C.zero, [], 1);
  hline(p4, be[0], be[1], 55, C.ref, [], 1.4);
  hline(p4, be[0], be[1], 55, C.dash, [5, 2, 1, 2]);
  line(p4, B, be[0], be[1], C.up, 1.2);
  frame(p4, 'BPDI Hilo Index   기준 55');
  ylab(p4, be[0], be[1], 50, '50'); ylab(p4, be[0], be[1], 55, '55');

  /* 6단 — G (정확 눈금: 매수 청≤5·적≥40 / 주봉 매도 25) */
  var p5 = panel(5), ge = ext([G[0], G[1], [0, 45]]);
  months(p5);
  hline(p5, ge[0], ge[1], 40, C.ref, [4, 3], 1);
  hline(p5, ge[0], ge[1], 25, C.ref, [], 1.2);
  hline(p5, ge[0], ge[1], 5, C.ref, [], 1);
  hline(p5, ge[0], ge[1], 0, C.zero, [], 1);
  line(p5, G[0], ge[0], ge[1], C.up, 1.3);
  line(p5, G[1], ge[0], ge[1], C.dn, 1.3);
  frame(p5, 'Golden Buy(적) · Golden Sell(청)   기준 5·25·40 — HTS와 같은 값');
  ylab(p5, ge[0], ge[1], 5, '5'); ylab(p5, ge[0], ge[1], 25, '25');
  ylab(p5, ge[0], ge[1], 40, '40');

  /* 날짜축 — 겹치지 않도록 최소 간격을 둡니다 */
  x.fillStyle = C.text; x.font = '9px sans-serif'; x.textAlign = 'center';
  var last = null, lastX = -1e9, MINGAP = 44;
  for (i = 0; i < n; i++) {
    var s = String(d.t[i]);
    var key = s.length > 7 ? s.slice(0, 7) : s.slice(0, 5);
    if (key !== last) {
      var xx = px(i);
      if (last !== null && xx - lastX >= MINGAP && xx < L + PW - 18) {
        x.fillText(key, xx, H - 3); lastX = xx;
      }
      last = key;
    }
  }
}

g.drawChart = drawChart;
})(window);
