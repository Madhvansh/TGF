/*
 * charts.js — hand-rolled SVG chart primitives. No canvas, no dependencies.
 *
 * Exposes Charts.sparkline(), Charts.lineChart(), Charts.gauge().
 * Everything is drawn as SVG so it stays crisp, themeable via CSS custom
 * properties (colors come in as var(--...) strings), and screen-reader legible
 * (role="img" + aria-label + a visually-hidden data summary).
 *
 * Responsiveness: charts render into a measured width and re-render on resize
 * via a shared ResizeObserver; tick density thins on narrow viewports.
 * A single reusable tooltip node is positioned over the page for crosshairs.
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, kids) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (kids) kids.forEach(function (c) { if (c) n.appendChild(c); });
    return n;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function toMs(t) { return typeof t === "number" ? t : Date.parse(t); }
  function fmtNum(v, d) { return (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d == null ? 1 : d); }

  // ---- shared tooltip -----------------------------------------------------
  var tip = null;
  function getTip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tip";
      tip.setAttribute("role", "status");
      tip.hidden = true;
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(html, x, y) {
    var t = getTip();
    t.innerHTML = html;
    t.hidden = false;
    var pad = 12, w = t.offsetWidth, h = t.offsetHeight;
    var left = x + pad, top = y - h - pad;
    if (left + w > window.innerWidth - 4) left = x - w - pad;
    if (top < 4) top = y + pad;
    t.style.left = Math.max(4, left) + "px";
    t.style.top = Math.max(4, top) + "px";
  }
  function hideTip() { if (tip) tip.hidden = true; }

  // ---- shared resize handling --------------------------------------------
  var ro = null, roMap = new WeakMap();
  function observe(node, render) {
    roMap.set(node, render);
    if (!ro && "ResizeObserver" in global) {
      ro = new ResizeObserver(function (entries) {
        entries.forEach(function (e) {
          var fn = roMap.get(e.target);
          if (fn) fn(Math.max(160, Math.round(e.contentRect.width)));
        });
      });
    }
    if (ro) ro.observe(node);
  }

  // Nice-ish domain padding
  function padDomain(min, max) {
    if (min === max) { min -= 1; max += 1; }
    var p = (max - min) * 0.08;
    return [min - p, max + p];
  }

  function ticks(min, max, count) {
    var span = max - min, step = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = (span / count) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], start = Math.ceil(min / step) * step;
    for (var v = start; v <= max + step * 1e-6; v += step) out.push(v);
    return out;
  }

  // =========================================================================
  // Sparkline — tiny inline trend, no axes.
  // =========================================================================
  function sparkline(values, opts) {
    opts = opts || {};
    var w = opts.width || 96, h = opts.height || 28, pad = 2;
    var svg = el("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h,
      "class": "sparkline", preserveAspectRatio: "none", "aria-hidden": "true", focusable: "false" });
    var vals = values.filter(function (v) { return v != null && !isNaN(v); });
    if (!vals.length) return svg;
    var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
    if (min === max) { min -= 1; max += 1; }
    var n = values.length;
    var x = function (i) { return pad + (i / (n - 1)) * (w - 2 * pad); };
    var y = function (v) { return h - pad - ((v - min) / (max - min)) * (h - 2 * pad); };
    if (opts.band) {
      var lo = Math.max(opts.band[0], min), hi = Math.min(opts.band[1], max);
      if (hi > lo) svg.appendChild(el("rect", { x: pad, y: y(hi), width: w - 2 * pad,
        height: Math.max(0, y(lo) - y(hi)), fill: "var(--band-fill)" }));
    }
    var d = "";
    values.forEach(function (v, i) { if (v != null && !isNaN(v)) d += (d ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); });
    svg.appendChild(el("path", { d: d, fill: "none", stroke: opts.color || "var(--accent)",
      "stroke-width": 1.5, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    var li = n - 1;
    svg.appendChild(el("circle", { cx: x(li), cy: y(values[li]), r: 1.8, fill: opts.color || "var(--accent)" }));
    return svg;
  }

  // =========================================================================
  // Gauge — arc with colored risk zones and a value needle.
  // cfg: {value, min, max, zones:[{to,color}], label, unit, decimals, verdict}
  // =========================================================================
  function polar(cx, cy, r, deg) {
    var rad = (deg - 180) * Math.PI / 180; // 180deg = left, 0deg = right (top half)
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  }
  function arcPath(cx, cy, r, a0, a1) {
    var p0 = polar(cx, cy, r, a0), p1 = polar(cx, cy, r, a1);
    var large = (a1 - a0) > 180 ? 1 : 0;
    return "M" + p0[0].toFixed(2) + " " + p0[1].toFixed(2) +
      "A" + r + " " + r + " 0 " + large + " 1 " + p1[0].toFixed(2) + " " + p1[1].toFixed(2);
  }
  function gauge(container, cfg) {
    var W = 220, H = 132, cx = W / 2, cy = 116, r = 92, sw = 13;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, "class": "gauge",
      role: "img", "aria-label": cfg.label + " gauge, value " + fmtNum(cfg.value, cfg.decimals) +
      ", " + (cfg.verdict || "") });
    var min = cfg.min, max = cfg.max, span = max - min;
    var toDeg = function (v) { return Math.max(0, Math.min(180, ((v - min) / span) * 180)); };
    // track
    svg.appendChild(el("path", { d: arcPath(cx, cy, r, 0, 180), fill: "none",
      stroke: "var(--hairline)", "stroke-width": sw, "stroke-linecap": "butt" }));
    // zones
    var prev = min;
    (cfg.zones || []).forEach(function (z) {
      var a0 = toDeg(prev), a1 = toDeg(z.to);
      if (a1 > a0) svg.appendChild(el("path", { d: arcPath(cx, cy, r, a0, a1), fill: "none",
        stroke: z.color, "stroke-width": sw, "stroke-linecap": "butt", opacity: 0.9 }));
      prev = z.to;
    });
    // needle
    var vClamped = Math.max(min, Math.min(max, cfg.value));
    var np = polar(cx, cy, r - sw - 4, toDeg(vClamped));
    svg.appendChild(el("line", { x1: cx, y1: cy, x2: np[0].toFixed(2), y2: np[1].toFixed(2),
      stroke: "var(--ink)", "stroke-width": 2.5, "stroke-linecap": "round", "class": "gauge-needle" }));
    svg.appendChild(el("circle", { cx: cx, cy: cy, r: 4.5, fill: "var(--ink)" }));
    // value + unit + verdict as SVG text (kept inside the graphic for export fidelity)
    var val = el("text", { x: cx, y: cy - 26, "text-anchor": "middle", "class": "gauge-val" });
    val.textContent = fmtNum(cfg.value, cfg.decimals);
    svg.appendChild(val);
    if (cfg.verdict) {
      var vd = el("text", { x: cx, y: cy - 8, "text-anchor": "middle", "class": "gauge-verdict" });
      vd.textContent = cfg.verdict;
      svg.appendChild(vd);
    }
    // min/max end labels
    var lo = el("text", { x: cx - r, y: cy + 18, "text-anchor": "middle", "class": "gauge-end" });
    lo.textContent = fmtNum(min, 0);
    var hi = el("text", { x: cx + r, y: cy + 18, "text-anchor": "middle", "class": "gauge-end" });
    hi.textContent = fmtNum(max, 0);
    svg.appendChild(lo); svg.appendChild(hi);

    clear(container); container.appendChild(svg);
    return { update: function (ns) { gauge(container, Object.assign({}, cfg, ns)); } };
  }

  // =========================================================================
  // lineChart — history line(s), guideline band, now-divider, forecast cone,
  // anomaly markers, optional overlay trajectory, crosshair + tooltip.
  // =========================================================================
  function lineChart(container, cfg) {
    var height = cfg.height || 220;
    var render = function (W) {
      W = W || container.clientWidth || 640;
      clear(container);
      var H = height, padL = 44, padR = 14, padT = 14, padB = 26;
      var innerW = W - padL - padR, innerH = H - padT - padB;

      // ---- collect x domain across history + forecast + overlay
      var histT = (cfg.t || []).map(toMs);
      var allT = histT.slice();
      var fc = cfg.forecast;
      if (fc && fc.t) fc.t.forEach(function (t) { allT.push(toMs(t)); });
      if (cfg.overlayT) cfg.overlayT.forEach(function (t) { allT.push(toMs(t)); });
      if (!allT.length) { return; }
      var tMin = Math.min.apply(null, allT), tMax = Math.max.apply(null, allT);

      // ---- y domain across everything drawn
      var yVals = [];
      (cfg.series || []).forEach(function (s) { s.values.forEach(function (v) { if (v != null && !isNaN(v)) yVals.push(v); }); });
      if (fc) { (fc.q10 || []).forEach(pushY); (fc.q90 || []).forEach(pushY); (fc.q50 || []).forEach(pushY); }
      if (cfg.overlays) cfg.overlays.forEach(function (o) { o.values.forEach(pushY); });
      if (cfg.band) { yVals.push(cfg.band[0]); yVals.push(cfg.band[1]); }
      function pushY(v) { if (v != null && !isNaN(v)) yVals.push(v); }
      if (!yVals.length) return;
      var yd = cfg.yDomain || padDomain(Math.min.apply(null, yVals), Math.max.apply(null, yVals));
      var y0 = yd[0], y1 = yd[1];

      var xS = function (t) { return padL + ((toMs(t) - tMin) / (tMax - tMin || 1)) * innerW; };
      var yS = function (v) { return padT + (1 - (v - y0) / (y1 - y0 || 1)) * innerH; };

      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
        "class": "linechart", role: "img", "aria-label": cfg.ariaLabel || "chart",
        preserveAspectRatio: "none" });
      if (cfg.summary) {
        var t = el("title"); t.textContent = cfg.summary; svg.appendChild(t);
      }

      // ---- guideline band
      if (cfg.band) {
        var by0 = yS(cfg.band[1]), by1 = yS(cfg.band[0]);
        svg.appendChild(el("rect", { x: padL, y: Math.min(by0, by1), width: innerW,
          height: Math.abs(by1 - by0), fill: "var(--band-fill)", "class": "band" }));
        [cfg.band[0], cfg.band[1]].forEach(function (bv) {
          svg.appendChild(el("line", { x1: padL, y1: yS(bv), x2: padL + innerW, y2: yS(bv),
            stroke: "var(--accent)", "stroke-width": 1, "stroke-dasharray": "1 4", opacity: 0.5 }));
        });
      }

      // ---- y grid + ticks
      var narrow = W < 430;
      ticks(y0, y1, narrow ? 3 : 4).forEach(function (tv) {
        var yy = yS(tv);
        if (yy < padT - 1 || yy > padT + innerH + 1) return;
        svg.appendChild(el("line", { x1: padL, y1: yy, x2: padL + innerW, y2: yy,
          stroke: "var(--hairline)", "stroke-width": 1, opacity: 0.7 }));
        var lb = el("text", { x: padL - 6, y: yy + 3, "text-anchor": "end", "class": "axis-lbl" });
        lb.textContent = fmtNum(tv, cfg.yDecimals != null ? cfg.yDecimals : 0);
        svg.appendChild(lb);
      });

      // ---- x ticks (day labels)
      var xtCount = narrow ? 3 : 6;
      for (var i = 0; i <= xtCount; i++) {
        var tv2 = tMin + (i / xtCount) * (tMax - tMin);
        var xx = xS(tv2);
        var dd = new Date(tv2);
        var lab = (dd.getUTCMonth() + 1) + "/" + dd.getUTCDate();
        var lb2 = el("text", { x: xx, y: H - 8, "text-anchor": "middle", "class": "axis-lbl" });
        lb2.textContent = lab;
        svg.appendChild(lb2);
      }

      // ---- forecast cone (drawn before lines so lines sit on top)
      if (fc && fc.t && fc.q10 && fc.q90) {
        var up = "", dn = "";
        fc.t.forEach(function (tt, i) {
          up += (i ? "L" : "M") + xS(tt).toFixed(1) + " " + yS(fc.q90[i]).toFixed(1);
        });
        for (var j = fc.t.length - 1; j >= 0; j--) {
          dn += "L" + xS(fc.t[j]).toFixed(1) + " " + yS(fc.q10[j]).toFixed(1);
        }
        // bridge from last history point for visual continuity
        var lastX = xS(histT[histT.length - 1]);
        svg.appendChild(el("path", { d: up + dn + "Z", fill: "var(--accent)", opacity: 0.12,
          stroke: "none", "class": "cone" }));
        var med = "";
        fc.t.forEach(function (tt, i) { med += (i ? "L" : "M") + xS(tt).toFixed(1) + " " + yS(fc.q50[i]).toFixed(1); });
        svg.appendChild(el("path", { d: med, fill: "none", stroke: "var(--accent)",
          "stroke-width": 1.6, "stroke-dasharray": "5 3", "class": "forecast-median" }));
      }

      // ---- now divider
      if (cfg.nowT != null) {
        var nx = xS(cfg.nowT);
        svg.appendChild(el("line", { x1: nx, y1: padT, x2: nx, y2: padT + innerH,
          stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "2 3", "class": "now-divider" }));
        var nl = el("text", { x: nx, y: padT + 2, "text-anchor": "middle", "class": "now-lbl" });
        nl.textContent = "now";
        svg.appendChild(nl);
      }

      // ---- overlays (approve/decline trajectories)
      if (cfg.overlays) cfg.overlays.forEach(function (o) {
        var xt = cfg.overlayT || cfg.t, d = "";
        o.values.forEach(function (v, i) { if (v != null) d += (d ? "L" : "M") + xS(xt[i]).toFixed(1) + " " + yS(v).toFixed(1); });
        svg.appendChild(el("path", { d: d, fill: "none", stroke: o.color || "var(--accent)",
          "stroke-width": 2, "stroke-dasharray": o.dash || null, "stroke-linejoin": "round" }));
      });

      // ---- history series
      (cfg.series || []).forEach(function (s) {
        var d = "";
        s.values.forEach(function (v, i) { if (v != null && !isNaN(v)) d += (d ? "L" : "M") + xS(histT[i]).toFixed(1) + " " + yS(v).toFixed(1); });
        svg.appendChild(el("path", { d: d, fill: "none", stroke: s.color || "var(--ink)",
          "stroke-width": s.width || 1.7, "stroke-linejoin": "round", "stroke-linecap": "round",
          "stroke-dasharray": s.dash || null }));
      });

      // ---- anomaly markers (diamonds)
      var markerHit = [];
      if (cfg.markers) cfg.markers.forEach(function (m) {
        var mx = xS(m.t), my = yS(m.value);
        var sz = m.severity === "high" ? 6 : m.severity === "medium" ? 5 : 4;
        var col = m.severity === "high" ? "var(--breach)" : "var(--watch)";
        var dia = el("path", { d: "M" + mx + " " + (my - sz) + "L" + (mx + sz) + " " + my +
          "L" + mx + " " + (my + sz) + "L" + (mx - sz) + " " + my + "Z", fill: col,
          stroke: "var(--surface)", "stroke-width": 1, "class": "marker", tabindex: "0",
          role: "button", "aria-label": "anomaly on " + m.param + ": " + m.note });
        svg.appendChild(dia);
        markerHit.push({ x: mx, y: my, m: m });
        function mtip(ev) {
          var r = svg.getBoundingClientRect();
          showTip("<b>Anomaly · " + m.severity + "</b><br>" + m.param + " · score " + m.score +
            "<br>" + m.note, r.left + mx * (r.width / W), r.top + my * (r.height / H));
        }
        dia.addEventListener("mouseenter", mtip);
        dia.addEventListener("focus", mtip);
        dia.addEventListener("mouseleave", hideTip);
        dia.addEventListener("blur", hideTip);
      });

      // ---- crosshair (history region)
      var focus = el("g", { "class": "crosshair", opacity: 0 });
      var vline = el("line", { y1: padT, y2: padT + innerH, stroke: "var(--muted)", "stroke-width": 1 });
      focus.appendChild(vline);
      var dots = (cfg.series || []).map(function (s) {
        var c = el("circle", { r: 3, fill: s.color || "var(--ink)", stroke: "var(--surface)", "stroke-width": 1 });
        focus.appendChild(c); return c;
      });
      svg.appendChild(focus);

      var overlay = el("rect", { x: padL, y: padT, width: innerW, height: innerH, fill: "transparent",
        "class": "hit" });
      svg.appendChild(overlay);
      function move(ev) {
        var r = svg.getBoundingClientRect();
        var clientX = (ev.touches ? ev.touches[0].clientX : ev.clientX);
        var clientY = (ev.touches ? ev.touches[0].clientY : ev.clientY);
        var px = (clientX - r.left) * (W / r.width);
        // nearest history index
        var idx = 0, best = Infinity;
        histT.forEach(function (tt, i) { var dpx = Math.abs(xS(tt) - px); if (dpx < best) { best = dpx; idx = i; } });
        var cx = xS(histT[idx]);
        vline.setAttribute("x1", cx); vline.setAttribute("x2", cx);
        var rows = "";
        (cfg.series || []).forEach(function (s, k) {
          var v = s.values[idx];
          if (v == null || isNaN(v)) { dots[k].setAttribute("opacity", 0); return; }
          dots[k].setAttribute("opacity", 1);
          dots[k].setAttribute("cx", cx); dots[k].setAttribute("cy", yS(v));
          rows += "<span class='k' style='color:" + (s.color || "var(--ink)") + "'>■</span> " +
            s.label + ": <b>" + fmtNum(v, cfg.yDecimals != null ? cfg.yDecimals : 1) + "</b><br>";
        });
        focus.setAttribute("opacity", 1);
        var dd = new Date(histT[idx]);
        var when = (dd.getUTCMonth() + 1) + "/" + dd.getUTCDate() + " " +
          String(dd.getUTCHours()).padStart(2, "0") + ":00";
        showTip("<b>" + when + "</b><br>" + rows, clientX, clientY);
      }
      function leave() { focus.setAttribute("opacity", 0); hideTip(); }
      overlay.addEventListener("mousemove", move);
      overlay.addEventListener("mouseleave", leave);
      overlay.addEventListener("touchstart", function (e) { move(e); }, { passive: true });
      overlay.addEventListener("touchmove", function (e) { move(e); }, { passive: true });
      overlay.addEventListener("touchend", leave);

      // ---- visually-hidden data summary for screen readers
      if (cfg.summary) {
        var sr = document.createElement("span");
        sr.className = "visually-hidden";
        sr.textContent = cfg.summary;
        // attach as sibling caption via aria-describedby is overkill; title above covers it.
      }

      container.appendChild(svg);
    };

    render(container.clientWidth);
    observe(container, render);
    return { render: render };
  }

  var Charts = { sparkline: sparkline, gauge: gauge, lineChart: lineChart,
    _showTip: showTip, _hideTip: hideTip };
  if (typeof module !== "undefined" && module.exports) module.exports = Charts;
  global.Charts = Charts;
})(typeof window !== "undefined" ? window : this);
