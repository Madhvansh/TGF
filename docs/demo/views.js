/* TGF Console — view renderers. Everything drawn here comes from the scenario
   JSON produced offline by generate_data.py; chemistry indices are additionally
   recomputed live in the browser by chem.js. */
"use strict";

window.Views = {};

/* ============================================================ shared bits */
function provChip(text, title, extraClass) {
  return el("span", { class: "prov" + (extraClass ? " " + extraClass : ""),
    text: text, "data-pop": "<div class='p-title'>Provenance</div>" + escapeHtml(title || text) });
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function kindChip(kind) {
  var map = {
    live: ["live sensor", "badge-live", "Simulated live-sensor channel (hourly)."],
    estimated: ["estimated", "", "Derived by the TGF pipeline from the simulated sensors (e.g. via cycles of concentration or the residual tracker)."],
    lab: ["lab (manual)", "", "Simulated lab panel result, sampled every 12 h — presented as a manually entered value."],
    fault: ["FAULT", "badge-fault", "The simulated probe has failed sanity checks — readings after the fault are untrustworthy by design."],
  };
  var m = map[kind] || map.estimated;
  return provChip(m[0], m[2], m[1]);
}
function sparkOpt(vals, band, color) {
  var lo = Math.min.apply(null, vals.filter(isFinite));
  var hi = Math.max.apply(null, vals.filter(isFinite));
  var pad = (hi - lo || 1) * 0.15;
  return {
    animation: false,
    grid: { left: 2, right: 2, top: 4, bottom: 2 },
    xAxis: { type: "category", show: false, boundaryGap: false, data: vals.map(function (_, i) { return i; }) },
    yAxis: { type: "value", show: false, min: lo - pad, max: hi + pad },
    series: [{
      type: "line", data: vals, symbol: "none", lineStyle: { width: 1.6, color: color },
      areaStyle: { opacity: 0.12, color: color },
      markArea: band ? { silent: true, itemStyle: { color: cssv("--accent-soft"), opacity: 0.4 },
        data: [[{ yAxis: band[0] }, { yAxis: band[1] }]] } : undefined,
    }],
  };
}
function stateColor(st) {
  return st === "bad" ? cssv("--bad") : st === "warn" ? cssv("--warn") : cssv("--accent");
}
function gaugeOpt(value, min, max, zones, label) {
  // zones: array of [fromFrac, toFrac, color]
  return {
    animationDuration: 300,
    series: [{
      type: "gauge", min: min, max: max, startAngle: 205, endAngle: -25,
      radius: "88%", center: ["50%", "56%"],
      progress: { show: false },
      axisLine: { lineStyle: { width: 9, color: zones } },
      pointer: { length: "58%", width: 4, itemStyle: { color: cssv("--ink") } },
      axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { show: false },
      anchor: { show: true, size: 7, itemStyle: { color: cssv("--ink") } },
      title: { show: true, offsetCenter: [0, "34%"], fontSize: 11, color: cssv("--muted") },
      detail: { valueAnimation: false, offsetCenter: [0, "76%"], fontSize: 17,
        fontWeight: 800, color: cssv("--ink"), formatter: function (v) { return fmt(v, 2); } },
      data: [{ value: value, name: label }],
    }],
  };
}
function idxZones(kind) {
  var ok = cssv("--ok"), warn = cssv("--warn"), bad = cssv("--bad");
  if (kind === "lsi") {
    // range -2..2 : corrosive < -0.5 | balanced | > 0.5 scaling watch | > 1 scaling
    return [[0.375, bad], [0.625, ok], [0.75, warn], [1, bad]];
  }
  // rsi/psi range 4..10 : < 6 scaling | 6-7 balanced | > 7 corrosive
  return [[1 / 3, warn], [0.5, ok], [0.75, warn], [1, bad]];
}
function idxVerdict(kind, v) {
  return kind === "lsi" ? Chem.lsiVerdict(v) : kind === "rsi" ? Chem.rsiVerdict(v) : Chem.psiVerdict(v);
}

/* ============================================================== OVERVIEW */
var KPI_KEYS = ["ph", "conductivity", "inhibitor", "coc", "orp", "temperature"];

Views.overview = {
  render: function (root, d, i) {
    var head = el("div", { class: "view-head" }, [
      el("h2", { text: "Overview" }),
      el("span", { class: "note", text: d.meta.title + " — " + d.meta.subtitle }),
    ]);
    root.appendChild(head);

    /* hero: verdict + radar */
    var hero = el("div", { class: "ov-hero" });
    var verdict = el("div", { class: "card verdict", id: "ovVerdict" });
    hero.appendChild(verdict);
    var radarCard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Tower health radar" }),
        el("div", { class: "right" }, [
          el("button", { class: "chip-btn", text: "How is this scored?", onclick: function (e) {
            var html = "<div class='p-title'>Radar axis formulas</div>" + d.radar.axes.map(function (ax) {
              return "<b>" + escapeHtml(ax.label) + "</b><br><code>" + escapeHtml(ax.formula) + "</code>";
            }).join("<br>");
            showPop(html, e.clientX, e.clientY);
            e.stopPropagation();
          } }),
          provChip("computed offline", d.radar.provenance),
        ]),
      ]),
      el("div", { class: "chart", id: "ovRadar", style: "height:250px" }),
    ]);
    hero.appendChild(radarCard);
    root.appendChild(hero);

    /* KPI cards */
    var kpis = el("div", { class: "kpis", id: "ovKpis" });
    root.appendChild(kpis);

    /* mid: indices gauges + alerts summary */
    var mid = el("div", { class: "ov-mid" });
    var gcard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Water chemistry indices" }),
        el("div", { class: "right" }, [provChip("computed in-browser",
          "LSI and RSI recomputed live by chem.js from the simulated chemistry at the sim clock — the same formulas TGF's physics engine uses. PSI uses the Puckorius equilibrium pH. See Data & Methods for the formulas and citations.")]),
      ]),
      el("div", { class: "gauges3", id: "ovGauges" }),
      el("p", { style: "font-size:.74rem;color:var(--muted);margin:.4rem 0 0",
        text: "These indices are computed live in your browser from the simulated water chemistry — the same formulas TGF's physics engine uses." }),
    ]);
    mid.appendChild(gcard);
    var acard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Active alerts" }),
        el("div", { class: "right" }, [
          el("button", { class: "chip-btn", text: "Open alert feed", onclick: openDrawer }),
        ]),
      ]),
      el("div", { id: "ovAlerts" }),
    ]);
    mid.appendChild(acard);
    root.appendChild(mid);

    this.update(root, d, i);
    this._renderRadar(d, i);
  },

  _renderRadar: function (d, i) {
    var dom = $("#ovRadar"); if (!dom) return;
    var anc = anchorFor(i);
    var idxNow = d.radar.anchors_h.indexOf(anc.h);
    var idxPrev = Math.max(0, idxNow - 2);   // ~24 h earlier (12 h anchor step)
    var axes = d.radar.axes.map(function (ax) { return { name: ax.label, max: 100 }; });
    var nowVals = d.radar.axes.map(function (ax) { return d.radar.scores[ax.key][idxNow]; });
    var prevVals = d.radar.axes.map(function (ax) { return d.radar.scores[ax.key][idxPrev]; });
    mkChart(dom, {
      animationDuration: 350,
      legend: { bottom: 0, textStyle: { color: cssv("--muted"), fontSize: 10.5 },
        itemWidth: 12, itemHeight: 8 },
      tooltip: { backgroundColor: cssv("--card-hi"), borderColor: cssv("--line"),
        textStyle: { color: cssv("--ink"), fontSize: 11 } },
      radar: {
        indicator: axes, radius: "62%", center: ["50%", "48%"],
        axisName: { color: cssv("--muted"), fontSize: 10 },
        splitArea: { areaStyle: { color: ["transparent", cssv("--line-soft")] } },
        splitLine: { lineStyle: { color: cssv("--line-soft") } },
        axisLine: { lineStyle: { color: cssv("--line") } },
      },
      series: [{
        type: "radar",
        data: [
          { value: prevVals, name: "24 h earlier", lineStyle: { color: cssv("--faint"), width: 1.4 },
            itemStyle: { color: cssv("--faint") }, areaStyle: { opacity: 0.06 }, symbolSize: 3 },
          { value: nowVals, name: "at sim clock", lineStyle: { color: cssv("--accent"), width: 2 },
            itemStyle: { color: cssv("--accent") }, areaStyle: { opacity: 0.16 }, symbolSize: 3 },
        ],
      }],
    });
  },

  update: function (root, d, i) {
    /* verdict */
    var v = verdictAt(i);
    var vd = $("#ovVerdict");
    if (vd) {
      vd.className = "card verdict " + v.cls;
      vd.innerHTML = "";
      vd.appendChild(el("span", { class: "v-dot" }));
      vd.appendChild(el("div", { class: "label", text: "Tower state · at sim clock" }));
      vd.appendChild(el("div", { class: "v-state", text: v.text }));
      vd.appendChild(el("div", { class: "v-sub", text: v.sub }));
      var hints = {
        baseline: ["Open the Digital twin to watch the loop at steady state.",
                   "Scrub or play the sim clock to replay the whole week."],
        scaling: ["In Forecast & anomalies, the cone crosses the guideline before the excursion peaks.",
                  "Watch the radar's scaling margin collapse as cycles of concentration climb."],
        makeup: ["Watch LSI go negative after the day-2 makeup shift.",
                 "The Dosing advisor raises the inhibitor dose with a corrosion-risk rationale."],
        fault: ["Open Safety & interlocks — the sensor-sanity lamp trips on day 6.",
                "The Dosing advisor shows the dose held; forecasts on the bad probe are suppressed."],
      };
      var ul = el("ul", { class: "v-hints" });
      (hints[d.meta.id] || []).forEach(function (t) { ul.appendChild(el("li", { text: t })); });
      vd.appendChild(ul);
      vd.appendChild(el("div", { style: "margin-top:.4rem" }, [
        provChip("simulated", d.meta.provenance_page),
      ]));
    }

    /* KPI cards — rebuilt each tick (cheap: 6 small sparklines) */
    var box = $("#ovKpis");
    if (box) {
      disposeChartsIn(box);
      box.innerHTML = "";
      KPI_KEYS.forEach(function (key) {
        var p = d.params[key];
        var val = valueAt(key, i);
        var kind = kindOf(key, i);
        var isFault = kind === "fault" && dropoutActiveAt(i);
        var st = isFault ? "bad" : bandState(val, p.band);
        var card = el("div", { class: "card kpi state-" + st, role: "button", tabindex: "0",
          "aria-label": p.label + " details" });
        card.appendChild(el("div", { class: "k-top" }, [
          el("span", { class: "k-name", text: p.label }),
          el("span", { style: "margin-left:auto" }, [kindChip(kind)]),
        ]));
        card.appendChild(el("div", { class: "k-val", html: (isFault ? "—" : fmt(val, p.decimals)) +
          (p.unit ? "<span class='u'>" + p.unit + "</span>" : "") }));
        var band = el("div", { class: "k-band" }, [el("div", { class: "fill" })]);
        var frac = clamp(((val - p.band[0]) / (p.band[1] - p.band[0])) || 0, 0, 1);
        if (!isFault) band.appendChild(el("div", { class: "mark", style: "left:calc(" + (frac * 100) + "% - 1.5px)" }));
        card.appendChild(band);
        var d24 = val - valueAt(key, Math.max(0, i - 24));
        var flatEps = 0.5 * Math.pow(10, -p.decimals);
        var cls = Math.abs(d24) < flatEps ? "flat" : d24 > 0 ? "up" : "down";
        card.appendChild(el("div", { class: "k-meta" }, [
          el("span", { text: "band " + fmt(p.band[0], p.decimals) + "–" + fmt(p.band[1], p.decimals) + (p.unit ? " " + p.unit : "") }),
          el("span", { class: "delta " + cls, text: (d24 >= 0 ? "▲ +" : "▼ ") + fmt(d24, p.decimals) + " /24h" }),
        ]));
        var spark = el("div", { class: "spark" });
        card.appendChild(spark);
        box.appendChild(card);
        var seg = d.series[key].slice(Math.max(0, i - 47), i + 1);
        mkChart(spark, sparkOpt(seg, p.band, stateColor(st)));
        card.addEventListener("click", function () { openParamPanel(key, d, i); });
        card.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openParamPanel(key, d, i); } });
      });
    }

    /* gauges */
    var g = $("#ovGauges");
    if (g) {
      disposeChartsIn(g);
      g.innerHTML = "";
      var suppressed = dropoutActiveAt(i);
      [["lsi", "LSI", -2, 2], ["rsi", "RSI", 4, 10], ["psi", "PSI", 4, 10]].forEach(function (def) {
        var cell = el("div", { class: "gauge-cell" });
        g.appendChild(cell);
        if (suppressed) {
          cell.appendChild(el("div", { class: "suppressed-note",
            text: def[1] + " suppressed — pH probe faulted; no index is computed on bad data" }));
        } else {
          var val = d.indices[def[0]][i];
          var ch = el("div", { class: "g-chart" });
          cell.appendChild(ch);
          cell.appendChild(el("div", { class: "g-verdict", text: idxVerdict(def[0], val) }));
          mkChart(ch, gaugeOpt(val, def[2], def[3], idxZones(def[0]), def[1]));
        }
      });
    }

    /* alerts summary */
    var ab = $("#ovAlerts");
    if (ab) {
      ab.innerHTML = "";
      var list = visibleAlerts(i).filter(function (a) { return a.state === "active"; }).slice(0, 4);
      if (!list.length) {
        ab.appendChild(el("div", { class: "empty-note", text: "No active alerts at the sim clock — a quiet loop is a good loop." }));
      } else {
        list.forEach(function (a) {
          var card = el("div", { class: "alert-card sev-" + a.severity });
          card.appendChild(el("div", { class: "a-row" }, [
            el("span", { class: "a-time", text: tShort(a.t) }),
            el("span", { class: "a-state active", text: "active" }),
            el("span", { class: "prov", text: a.source, title: "alert source (simulated run)" }),
          ]));
          card.appendChild(el("div", { class: "a-text", text: a.text }));
          ab.appendChild(card);
        });
      }
    }

    /* radar follows the anchor as the clock moves */
    if ($("#ovRadar") && anchorFor(i).h !== this._lastAnchor) {
      this._lastAnchor = anchorFor(i).h;
      disposeChartsIn($("#ovRadar"));
      this._renderRadar(d, i);
    }
  },
};

/* dispose only the chart instances whose DOM lives inside `node` */
function disposeChartsIn(node) {
  S.charts = S.charts.filter(function (c) {
    var dom = c.getDom && c.getDom();
    if (dom && node.contains(dom)) { try { c.dispose(); } catch (e) {} return false; }
    return true;
  });
}

/* parameter slide-over: trend + distribution strip + provenance */
function openParamPanel(key, d, i) {
  var p = d.params[key] || d.lab.params[key];
  var isLab = !d.params[key];
  openSlideover(p.label, function (body) {
    var kind = kindOf(key, i);
    body.appendChild(el("div", { style: "display:flex;gap:.4rem;margin-bottom:.6rem;flex-wrap:wrap" }, [
      kindChip(isLab ? "lab" : kind),
      provChip("simulated", isLab ? d.lab.provenance : d.series_provenance),
    ]));
    var ch = el("div", { class: "chart chart-md" });
    body.appendChild(ch);
    var tArr, vals;
    if (isLab) { tArr = d.lab.t; vals = d.lab.values[key]; }
    else { tArr = d.series.t; vals = d.series[key]; }
    var nowT = d.series.t[i];
    var opt = {
      animation: false, grid: baseGrid(), tooltip: tooltipStyle(),
      xAxis: timeAxis(tArr),
      yAxis: Object.assign({ type: "value", scale: true,
        axisLabel: { color: cssv("--muted"), fontSize: 10.5 } }, axisStyle()),
      series: [{
        name: p.label, type: "line", data: vals,
        symbol: isLab ? "circle" : "none", symbolSize: 5,
        lineStyle: { width: 1.8, color: cssv("--accent") },
        markArea: bandMarkArea(p.band),
        markLine: { silent: true, symbol: "none",
          label: { formatter: "sim clock", color: cssv("--muted"), fontSize: 9 },
          lineStyle: { color: cssv("--faint"), type: "dashed" },
          data: [{ xAxis: tShort(nowT) }] },
      }],
    };
    mkChart(ch, opt);
    if (!isLab && d.distributions[key]) {
      var st = d.distributions[key];
      body.appendChild(el("div", { class: "label", text: "Distribution over the window", style: "margin:1rem 0 .3rem" }));
      var tbl = el("table", { class: "stat-table" });
      [["min", st.min], ["q1", st.q1], ["median", st.median], ["mean", st.mean],
       ["q3", st.q3], ["max", st.max], ["IQR fences", st.fence_lo + " / " + st.fence_hi]].forEach(function (r) {
        tbl.appendChild(el("tr", null, [el("td", { text: r[0] }), el("td", { text: String(r[1]) })]));
      });
      body.appendChild(tbl);
    }
    body.appendChild(el("p", { style: "font-size:.7rem;color:var(--faint);margin-top:.9rem",
      text: "Simulated data — guideline band shown as the shaded area." }));
  });
}

/* ================================================================= TWIN */
Views.twin = {
  render: function (root, d, i) {
    root.appendChild(el("div", { class: "view-head" }, [
      el("h2", { text: "Digital twin" }),
      el("span", { class: "note", text: "Animated schematic of the simulated tower — click any tag for its trend. Dot colors mark how each value is produced." }),
    ]));
    var wrap = el("div", { class: "twin-wrap" });

    /* SVG flowsheet */
    var svgCard = el("div", { class: "card twin-svg-card" });
    svgCard.innerHTML = this._svg();
    wrap.appendChild(svgCard);

    /* tag panels */
    var panels = el("div", { class: "tagpanels" });
    var recirc = el("div", { class: "card tagpanel" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Recirculating water" }),
        el("div", { class: "right" }, [provChip("simulated", d.series_provenance)]),
      ]),
      el("div", { class: "tp-grid", id: "tagsRecirc" }),
    ]);
    var makeup = el("div", { class: "card tagpanel" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Makeup water" }),
        el("div", { class: "right" }, [provChip("simulated", d.series_provenance)]),
      ]),
      el("div", { class: "tp-grid", id: "tagsMakeup" }),
      el("div", { class: "kindlegend" }, [
        el("span", { html: "<i style='background:var(--ok)'></i>live sensor" }),
        el("span", { html: "<i style='background:var(--accent)'></i>estimated" }),
        el("span", { html: "<i style='background:var(--warn)'></i>lab (manual)" }),
        el("span", { html: "<i style='background:var(--bad)'></i>fault" }),
      ]),
    ]);
    panels.appendChild(recirc);
    panels.appendChild(makeup);
    wrap.appendChild(panels);
    root.appendChild(wrap);

    /* pumps + valve row */
    var prow = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Dosing & blowdown equipment" }),
        el("div", { class: "right" }, [provChip("simulated",
          "Equipment states derive from the offline dosing timeline and a seeded biocide schedule — simulated for the demo.")]),
      ]),
      el("div", { class: "pumps", id: "twinPumps" }),
    ]);
    root.appendChild(prow);

    this.update(root, d, i);
  },

  _svg: function () {
    /* hand-drawn flowsheet, 1200x520 */
    return '<svg class="twin-svg" viewBox="0 190 1180 340" role="img" aria-label="Cooling tower process schematic: makeup water and recirculating loop through a heat exchanger, tower fill and basin, with blowdown and dosing points">' +
      /* flows */
      '<path class="flow hot animated" d="M330 330 L330 240 Q330 225 345 225 L760 225 Q775 225 775 240 L775 300"/>' +  /* hot return riser to tower */
      '<path class="flow cold animated" d="M775 430 L775 460 Q775 470 765 470 L360 470 Q345 470 345 455 L345 415 Q345 400 330 400 L250 400"/>' + /* cold from basin to pump/exchanger */
      '<path class="flow cold animated" d="M250 400 L215 400 Q200 400 200 385 L200 345 Q200 330 215 330 L330 330"/>' + /* pump loop to exchanger */
      '<path class="flow makeup animated" d="M60 470 L340 470"/>' +               /* makeup line into basin path */
      '<path class="flow blowdown animated" d="M900 470 L1140 470"/>' +           /* blowdown out */
      '<path class="flow dose animated" d="M560 505 L640 505 L640 470"/>' +       /* dosing into basin return */
      /* heat exchanger */
      '<rect class="equip" x="255" y="300" width="150" height="60" rx="9"/>' +
      '<path d="M265 330 h22 l10 -14 l14 28 l14 -28 l14 28 l10 -14 h26" fill="none" stroke="var(--accent)" stroke-width="2"/>' +
      '<text class="equip-label" x="330" y="290" text-anchor="middle">Heat exchanger</text>' +
      '<text class="svg-tag" x="330" y="375" text-anchor="middle">process heat load</text>' +
      /* recirc pump */
      '<circle class="equip" cx="250" cy="400" r="17"/>' +
      '<path d="M242 400 a8 8 0 1 0 16 0 a8 8 0 1 0 -16 0 M250 392 l6 -7" stroke="var(--muted)" stroke-width="1.6" fill="none"/>' +
      '<text class="equip-label" x="250" y="438" text-anchor="middle">Recirc pump</text>' +
      /* tower */
      '<path class="equip" d="M700 300 L850 300 L838 430 L712 430 Z"/>' +
      '<g id="twinFan"><circle cx="775" cy="278" r="24" fill="var(--card-hi)" stroke="var(--line)" stroke-width="1.4"/>' +
      '<g class="fanblade" id="fanBlades" style="transform-box: fill-box">' +
      '<path d="M775 278 L775 258 M775 278 L792 288 M775 278 L758 288" stroke="var(--accent)" stroke-width="3" stroke-linecap="round"/></g></g>' +
      '<text class="equip-label" x="775" y="248" text-anchor="middle">Cooling tower cell</text>' +
      '<path d="M718 320 h114 M714 345 h122 M710 370 h130" stroke="var(--line)" stroke-width="1.2" opacity=".8"/>' +
      '<text class="svg-tag" x="775" y="400" text-anchor="middle">fill / drift eliminators</text>' +
      /* basin */
      '<rect class="equip" x="660" y="440" width="240" height="46" rx="8"/>' +
      '<rect class="basin-water" x="666" y="452" width="228" height="28" rx="5"/>' +
      '<text class="equip-label" x="780" y="505" text-anchor="middle">Collection basin</text>' +
      /* makeup */
      '<rect class="equip" x="18" y="448" width="46" height="44" rx="7"/>' +
      '<text class="equip-label" x="41" y="440" text-anchor="middle">Makeup</text>' +
      /* blowdown valve */
      '<path class="equip" d="M900 460 l20 10 l-20 10 Z M940 460 l-20 10 l20 10 Z"/>' +
      '<text class="equip-label" x="920" y="440" text-anchor="middle">Blowdown valve</text>' +
      '<text class="svg-tag" x="1132" y="460" text-anchor="end">to discharge</text>' +
      /* dosing skid */
      '<rect class="equip" x="470" y="488" width="90" height="34" rx="7"/>' +
      '<text class="equip-label" x="515" y="480" text-anchor="middle">Dosing skid</text>' +
      '<circle id="doseDot" cx="640" cy="470" r="5" fill="var(--warn)"/>' +
      /* line labels */
      '<text class="svg-tag" x="552" y="215" text-anchor="middle">hot return</text>' +
      '<text class="svg-tag" x="560" y="462" text-anchor="middle">cold supply</text>' +
      '<text class="svg-tag" x="150" y="462" text-anchor="middle">makeup in</text>' +
      '<text class="svg-tag" x="1010" y="490" text-anchor="middle">blowdown</text>' +
      /* inline sensor tags (values injected on update) */
      '<g id="svgTags" font-family="var(--mono)">' +
      '<text class="svg-tag" x="470" y="245" id="svgTemp">—</text>' +
      '<text class="svg-tag" x="380" y="462" id="svgCond">—</text>' +
      '<text class="svg-tag" x="648" y="432" id="svgPh" text-anchor="end">—</text>' +
      '<text class="svg-tag" x="1010" y="512" id="svgBd" text-anchor="middle">—</text>' +
      '</g></svg>';
  },

  update: function (root, d, i) {
    /* svg live tags */
    var set = function (id, txt, bad) {
      var n = $("#" + id); if (n) { n.textContent = txt; n.style.fill = bad ? "var(--bad)" : ""; }
    };
    set("svgTemp", "T " + fmt(valueAt("temperature", i), 1) + " °C");
    set("svgCond", "cond " + fmt(valueAt("conductivity", i), 0) + " µS/cm");
    var phFault = kindOf("ph", i) === "fault";
    set("svgPh", "pH " + (dropoutActiveAt(i) ? "FAULT" : fmt(valueAt("ph", i), 2)), phFault);
    var bo = d.ops.blowdown_open[Math.min(i, d.ops.blowdown_open.length - 1)];
    set("svgBd", "open " + fmt(bo * 100, 0) + "%");
    var fan = $("#fanBlades");
    if (fan) fan.classList.toggle("paused", !S.playing);

    /* tag chips */
    var recircKeys = ["ph", "conductivity", "temperature", "orp", "inhibitor", "coc",
                      "calcium", "alkalinity", "chlorides"];
    var labKeys = Object.keys(d.lab.params);
    var makeupKeys = ["makeup_conductivity", "makeup_hardness"];
    var rbox = $("#tagsRecirc");
    if (rbox) {
      rbox.innerHTML = "";
      recircKeys.forEach(function (key) { appendTag(rbox, d, i, key, false); });
      labKeys.forEach(function (key) { appendTag(rbox, d, i, key, true); });
    }
    var mbox = $("#tagsMakeup");
    if (mbox) {
      mbox.innerHTML = "";
      makeupKeys.forEach(function (key) { appendTag(mbox, d, i, key, false); });
    }

    /* pumps */
    var pbox = $("#twinPumps");
    if (pbox) {
      pbox.innerHTML = "";
      pumpStates(d, i).forEach(function (ps) {
        pbox.appendChild(el("div", { class: "pump " + ps.cls }, [
          el("span", { class: "p-dot" }),
          el("div", null, [
            el("div", { class: "p-name", text: ps.label }),
            el("div", { class: "p-state", text: ps.state }),
          ]),
        ]));
      });
    }
  },
};

function labIndexAt(d, i) {
  var tNow = d.series.t[i];
  var li = 0;
  d.lab.t.forEach(function (t, k) { if (t <= tNow) li = k; });
  return li;
}
function appendTag(box, d, i, key, fromLab) {
  var p = fromLab ? d.lab.params[key] : d.params[key];
  if (!p) return;
  var val, kind;
  if (fromLab) { val = d.lab.values[key][labIndexAt(d, i)]; kind = "lab"; }
  else { val = valueAt(key, i); kind = kindOf(key, i); }
  var isFault = kind === "fault" && dropoutActiveAt(i);
  var st = isFault ? "bad" : bandState(val, p.band);
  var tag = el("button", { class: "tag state-" + (st === "ok" ? "ok" : st),
    "aria-label": p.label + " " + fmt(val, p.decimals) + " " + (p.unit || "") });
  tag.appendChild(el("span", { class: "t-kind " + kind }));
  tag.appendChild(el("span", { class: "t-name", text: p.label }));
  tag.appendChild(el("span", { class: "t-val", text: (isFault ? "FAULT" : fmt(val, p.decimals)) + (p.unit ? " " + p.unit : "") }));
  tag.addEventListener("click", function () { openParamPanel(key, d, i); });
  box.appendChild(tag);
}
function pumpStates(d, i) {
  var out = [];
  var dose = d.timeline.dose_ml_min[i];
  var blocked = blockedAt(i);
  out.push({
    label: "Inhibitor metering pump",
    cls: blocked ? "held" : dose > 0 ? "running" : "",
    state: blocked ? "HELD by safety layer" : dose > 0 ? "running · " + fmt(dose, 1) + " mL/min" : "idle",
  });
  var inSlug = (d.ops.biocide_slugs || []).some(function (s) {
    return i >= s.start_h && i < s.start_h + s.hours;
  });
  out.push({
    label: "Biocide slug pump",
    cls: inSlug ? "running" : "",
    state: inSlug ? "running · slug window" : "idle · scheduled twice weekly",
  });
  var bo = d.ops.blowdown_open[Math.min(i, d.ops.blowdown_open.length - 1)];
  out.push({
    label: "Blowdown valve",
    cls: bo > 0.05 ? "running" : "",
    state: "open " + fmt(bo * 100, 0) + "%",
  });
  return out;
}

/* ============================================================ ANALYTICS */
Views.analytics = {
  _sel: ["conductivity", "ph"],
  _focus: "conductivity",

  render: function (root, d, i) {
    var self = this;
    root.appendChild(el("div", { class: "view-head" }, [
      el("h2", { text: "Trends & analytics" }),
      el("span", { class: "note", text: "Pick parameters to trend; the shaded areas are guideline bands. Statistics cover the full 7-day window." }),
    ]));

    var controls = el("div", { class: "an-controls" });
    var groups = [
      ["Sensors", ["ph", "conductivity", "temperature", "orp"]],
      ["Estimated", ["inhibitor", "coc", "calcium", "alkalinity", "chlorides"]],
    ];
    groups.forEach(function (g) {
      controls.appendChild(el("span", { class: "label", text: g[0], style: "margin-right:.2rem" }));
      g[1].forEach(function (key) {
        controls.appendChild(el("button", {
          class: "chip-btn", text: d.params[key].label,
          "aria-pressed": String(self._sel.indexOf(key) >= 0),
          onclick: function () {
            var ix = self._sel.indexOf(key);
            if (ix >= 0) { if (self._sel.length > 1) self._sel.splice(ix, 1); }
            else { self._sel.push(key); }
            self._focus = key;
            renderView();
          },
        }));
      });
    });
    var anomToggle = el("button", {
      class: "chip-btn", text: "anomaly flags",
      "aria-pressed": String(!!this._anoms),
      style: "margin-left:auto",
      onclick: function () { self._anoms = !self._anoms; renderView(); },
    });
    controls.appendChild(anomToggle);
    root.appendChild(controls);

    var grid = el("div", { class: "an-grid" });
    var left = el("div", null);
    var chartsCard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Synchronized trends" }),
        el("div", { class: "right" }, [provChip("simulated", d.series_provenance)]),
      ]),
    ]);
    var group = "an-" + Math.random().toString(36).slice(2, 8);
    var trendSpecs = [];
    this._sel.forEach(function (key) {
      var p = d.params[key];
      var dom = el("div", { class: "chart", style: "height:185px" });
      chartsCard.appendChild(el("div", { class: "label", text: p.label + (p.unit ? " · " + p.unit : ""), style: "margin:.4rem 0 .1rem" }));
      chartsCard.appendChild(dom);
      var markPoint;
      if (self._anoms) {
        var pts = (d.anomalies || []).filter(function (a) { return a.param === key; }).map(function (a) {
          var ix = d.series.t.indexOf(a.t);
          return { coord: [tShort(a.t), d.series[key][ix]], value: "◆",
            itemStyle: { color: a.severity === "critical" ? cssv("--bad") : cssv("--warn") } };
        });
        if (pts.length) markPoint = { symbol: "diamond", symbolSize: 11, data: pts,
          label: { show: false },
          tooltip: { formatter: function (pp) { return "anomaly flag"; } } };
      }
      trendSpecs.push({ dom: dom, opt: {
        animation: false, grid: baseGrid(), tooltip: tooltipStyle(),
        xAxis: timeAxis(d.series.t),
        yAxis: Object.assign({ type: "value", scale: true }, axisStyle()),
        series: [{
          name: p.label, type: "line", data: d.series[key], symbol: "none",
          lineStyle: { width: 1.7, color: cssv("--accent") },
          markArea: bandMarkArea(p.band),
          markPoint: markPoint,
          markLine: { silent: true, symbol: "none",
            label: { formatter: "sim clock", color: cssv("--muted"), fontSize: 9 },
            lineStyle: { color: cssv("--faint"), type: "dashed" },
            data: [{ xAxis: tShort(d.series.t[i]) }] },
        }],
      } });
    });
    left.appendChild(chartsCard);

    /* correlation heatmap */
    var corr = d.correlation;
    var hm = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Correlation matrix" }),
        el("div", { class: "right" }, [
          provChip("computed offline", "Pearson correlation over " + corr.n + " hourly points (" + corr.window + "), computed by generate_data.py. " + corr.note),
        ]),
      ]),
      el("div", { class: "chart", id: "anHeat", style: "height:330px" }),
    ]);
    left.appendChild(hm);
    grid.appendChild(left);

    /* distribution card */
    var st = d.distributions[this._focus];
    var p = d.params[this._focus];
    var dist = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Distribution — " + p.label }),
        el("div", { class: "right" }, [provChip("computed offline",
          "Quantiles and a 24-bin histogram over the window, computed by generate_data.py from the simulated series.")]),
      ]),
      el("div", { class: "chart", id: "anHist", style: "height:190px" }),
    ]);
    var tbl = el("table", { class: "stat-table" });
    [["min", st.min], ["q1", st.q1], ["median", st.median], ["mean", st.mean], ["q3", st.q3],
     ["max", st.max], ["fence lo", st.fence_lo], ["fence hi", st.fence_hi]].forEach(function (r) {
      tbl.appendChild(el("tr", null, [el("td", { text: r[0] }), el("td", { text: String(r[1]) })]));
    });
    dist.appendChild(tbl);
    dist.appendChild(el("p", { style: "font-size:.7rem;color:var(--muted);margin-top:.5rem",
      text: "Click a parameter chip to focus its distribution." }));
    grid.appendChild(dist);
    root.appendChild(grid);

    /* trend charts — created after attach so they measure real widths */
    trendSpecs.forEach(function (ts) { var c = mkChart(ts.dom, ts.opt); c.group = group; });
    echarts.connect(group);

    /* histogram */
    var edges = st.hist.edges, counts = st.hist.counts;
    var labels = counts.map(function (_, k) { return fmt((edges[k] + edges[k + 1]) / 2, p.decimals); });
    mkChart($("#anHist"), {
      animation: false,
      grid: { left: 40, right: 8, top: 10, bottom: 22 },
      tooltip: Object.assign(tooltipStyle(), { trigger: "item" }),
      xAxis: (function () { var ax = axisStyle();
        Object.assign(ax.axisLabel, { fontSize: 9, interval: Math.floor(counts.length / 5) });
        return Object.assign(ax, { type: "category", data: labels }); })(),
      yAxis: Object.assign({ type: "value" }, axisStyle()),
      series: [{ type: "bar", data: counts, barWidth: "82%",
        itemStyle: { color: cssv("--accent"), opacity: 0.75, borderRadius: [3, 3, 0, 0] } }],
    });

    /* heatmap */
    var names = corr.params.map(function (k) { return d.params[k].label; });
    var cells = [];
    corr.matrix.forEach(function (row, r) { row.forEach(function (v, c) { cells.push([c, r, v]); }); });
    mkChart($("#anHeat"), {
      animation: false,
      grid: { left: 110, right: 60, top: 10, bottom: 74 },
      tooltip: Object.assign(tooltipStyle(), { trigger: "item",
        formatter: function (pp) { return names[pp.value[1]] + " × " + names[pp.value[0]] + " : <b>" + fmt(pp.value[2], 2) + "</b>"; } }),
      xAxis: (function () { var ax = axisStyle();
        Object.assign(ax.axisLabel, { fontSize: 9.5, rotate: 38, interval: 0 });
        return Object.assign(ax, { type: "category", data: names }); })(),
      yAxis: (function () { var ax = axisStyle();
        Object.assign(ax.axisLabel, { fontSize: 9.5, interval: 0 });
        return Object.assign(ax, { type: "category", data: names }); })(),
      visualMap: {
        min: -1, max: 1, calculable: false, orient: "vertical", right: 4, top: "center",
        itemHeight: 110, textStyle: { color: cssv("--muted"), fontSize: 9 },
        inRange: { color: ["#3B6FD4", cssv("--panel"), "#D45B4E"] },
      },
      series: [{ type: "heatmap", data: cells,
        label: { show: true, fontSize: 8.5, color: cssv("--muted"),
          formatter: function (pp) { return fmt(pp.value[2], 1); } },
        itemStyle: { borderColor: cssv("--bg"), borderWidth: 1 } }],
    });
  },
};

/* ============================================================= FORECAST */
Views.forecast = {
  _param: "conductivity",
  _anchorIdx: null,

  render: function (root, d, i) {
    var self = this;
    var anchors = d.forecast.anchors;
    if (this._anchorIdx === null || this._anchorIdx >= anchors.length) {
      this._anchorIdx = d.radar.anchors_h.indexOf(anchorFor(i).h);
      if (this._anchorIdx < 0) this._anchorIdx = anchors.length - 1;
    }
    root.appendChild(el("div", { class: "view-head" }, [
      el("h2", { text: "Forecast & anomalies" }),
      el("span", { class: "note", text: "24 h quantile cones recomputed at every 12 h anchor. Past anchors show the cone against what actually happened — right and wrong alike." }),
    ]));

    var controls = el("div", { class: "fc-controls" });
    ["conductivity", "ph", "inhibitor"].forEach(function (key) {
      controls.appendChild(el("button", {
        class: "chip-btn", text: d.params[key].label,
        "aria-pressed": String(self._param === key),
        onclick: function () { self._param = key; renderView(); },
      }));
    });
    var a = anchors[this._anchorIdx];
    controls.appendChild(el("span", { class: "label", text: "Forecast anchor", style: "margin-left:.8rem" }));
    var slider = el("input", { type: "range", min: 0, max: anchors.length - 1, step: 1,
      value: this._anchorIdx, "aria-label": "Forecast anchor", style: "width:180px;accent-color:var(--accent)" });
    slider.addEventListener("input", function (e) { self._anchorIdx = Number(e.target.value); renderView(); });
    controls.appendChild(slider);
    controls.appendChild(el("span", { class: "simclock", text: tShort(a.t) }));
    root.appendChild(controls);

    var card = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: d.params[this._param].label + " — history, cone, and what actually happened" }),
        el("div", { class: "right" }, [
          el("button", { class: "chip-btn", text: "How was this computed?", onclick: function (e) {
            showPop("<div class='p-title'>Forecast provenance</div>" + escapeHtml(d.forecast.provenance) +
              "<br><br><div class='p-title'>Anchors</div>24 h forecasts recomputed at every 12 h anchor from the data available up to that hour only.", e.clientX, e.clientY);
            e.stopPropagation();
          } }),
          provChip("forecast (offline)", d.forecast.provenance),
        ]),
      ]),
      el("div", { class: "chart chart-lg", id: "fcChart" }),
    ]);
    root.appendChild(card);

    this._renderChart(d);

    /* anomaly strip */
    var strip = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Anomaly episodes" }),
        el("div", { class: "right" }, [provChip("anomaly (offline)", d.anomalies_provenance)]),
      ]),
    ]);
    if (!(d.anomalies || []).length) {
      strip.appendChild(el("div", { class: "empty-note", text: "No anomaly episodes in this scenario's window." }));
    } else {
      d.anomalies.forEach(function (an) {
        var card2 = el("div", { class: "alert-card sev-" + (an.severity === "critical" ? "critical" : "warning") });
        card2.appendChild(el("div", { class: "a-row" }, [
          el("span", { class: "a-time", text: tShort(an.t) }),
          el("span", { class: "prov", text: an.param }),
          el("span", { class: "prov", text: "score " + fmt(an.score, 2), title: "detector system score" }),
          el("span", { class: "a-state " + (an.severity === "critical" ? "active" : ""), text: an.severity }),
        ]));
        card2.appendChild(el("div", { class: "a-text", text: an.note }));
        strip.appendChild(card2);
      });
    }
    root.appendChild(strip);

    /* model cards — the honesty story, first class */
    var mc = el("div", { class: "model-cards" });
    mc.appendChild(el("div", { class: "card model-card" }, [
      el("div", { class: "card-head" }, [el("h3", { text: "Chronos-T5 (small)" }),
        el("span", { class: "m-status ran", style: "margin-left:auto", text: "RAN" })]),
      el("p", { text: "Amazon's pretrained time-series foundation model, run offline on CPU, zero-shot, to produce the q10/q50/q90 cones you see here. No fine-tuning on this data." }),
      el("div", null, [provChip("forecast (offline)", d.forecast.provenance)]),
    ]));
    mc.appendChild(el("div", { class: "card model-card" }, [
      el("div", { class: "card-head" }, [el("h3", { text: "Statistical anomaly detector" }),
        el("span", { class: "m-status ran", style: "margin-left:auto", text: "RAN" })]),
      el("p", { text: "TGF's built-in detector: z-scores, rate-of-change and persistence checks per sensor, aggregated to a system score. It produced every anomaly flag on this page." }),
      el("div", null, [provChip("anomaly (offline)", d.anomalies_provenance)]),
    ]));
    mc.appendChild(el("div", { class: "card model-card" }, [
      el("div", { class: "card-head" }, [el("h3", { text: "MOMENT" }),
        el("span", { class: "m-status not-ran", style: "margin-left:auto", text: "NOT USED" })]),
      el("p", { text: "TGF supports MOMENT-based anomaly detection, but the repository ships no trained checkpoint — running it with random weights would produce meaningless scores, so this page does not claim it. The statistical detector ran instead." }),
    ]));
    root.appendChild(mc);
  },

  _renderChart: function (d) {
    var key = this._param;
    var p = d.params[key];
    var a = d.forecast.anchors[this._anchorIdx];
    var entry = a.params[key];
    var dom = $("#fcChart");
    if (entry.suppressed) {
      dom.outerHTML = "<div class='suppressed-note' style='height:300px'>" +
        escapeHtml(entry.note) + " — TGF refuses to forecast on data it cannot trust.</div>";
      return;
    }
    var histN = a.h;                       // history points up to the anchor
    var hist = d.series[key].slice(0, histN);
    var histT = d.series.t.slice(0, histN);
    var futT = entry.t;
    var allT = histT.concat(futT);
    var pad = function (arr, n) { return new Array(n).fill(null).concat(arr); };
    /* actual continuation (if the window extends past the anchor) */
    var actual = d.series[key].slice(histN, histN + entry.t.length);
    var series = [
      { name: "history", type: "line", data: hist.concat(new Array(futT.length).fill(null)),
        symbol: "none", lineStyle: { width: 1.7, color: cssv("--accent") },
        markArea: bandMarkArea(p.band),
        markLine: { silent: true, symbol: "none",
          label: { formatter: "anchor", color: cssv("--muted"), fontSize: 9 },
          lineStyle: { color: cssv("--faint"), type: "dashed" },
          data: [{ xAxis: tShort(histT[histT.length - 1]) }] } },
      { name: "q10–q90", type: "line", data: pad(entry.q10, histN), symbol: "none",
        lineStyle: { width: 0 }, stack: "cone", silent: true },
      { name: "q10–q90 band", type: "line", symbol: "none",
        data: pad(entry.q90.map(function (v, k) { return v - entry.q10[k]; }), histN),
        lineStyle: { width: 0 }, stack: "cone",
        areaStyle: { color: cssv("--accent"), opacity: 0.16 }, silent: true },
      { name: "forecast median", type: "line", data: pad(entry.q50, histN),
        symbol: "none", lineStyle: { width: 1.8, type: "dashed", color: cssv("--warn") } },
    ];
    if (actual.length) {
      series.push({ name: "what actually happened", type: "line", data: pad(actual, histN),
        symbol: "none", lineStyle: { width: 1.6, type: "dotted", color: cssv("--ok") } });
    }
    mkChart(dom, {
      animation: false,
      legend: { bottom: 0, textStyle: { color: cssv("--muted"), fontSize: 10 },
        data: ["history", "forecast median", "what actually happened"] },
      grid: { left: 52, right: 16, top: 18, bottom: 46 },
      tooltip: tooltipStyle(),
      xAxis: timeAxis(allT),
      yAxis: Object.assign({ type: "value", scale: true }, axisStyle()),
      series: series,
    });
  },
};

/* ============================================================== ADVISOR */
Views.advisor = {
  render: function (root, d, i) {
    var self = this;
    var rec = d.recommendation;
    var decided = S.decisions[S.scenario] || null;
    root.appendChild(el("div", { class: "view-head" }, [
      el("h2", { text: "Dosing advisor" }),
      el("span", { class: "note", text: "TGF recommends; a human decides. The decision below is the controller's latest, from the offline run." }),
    ]));

    var grid = el("div", { class: "adv-grid" });
    var left = el("div", null);

    /* recommendation card */
    var recCard = el("div", { class: "card" });
    recCard.appendChild(el("div", { class: "card-head" }, [
      el("h3", { text: "Recommendation — end of window" }),
      el("div", { class: "right" }, [
        el("span", { class: "rec-state " + rec.state, text: rec.state === "blocked" ? "BLOCKED" : "PROPOSED" }),
        provChip("controller (offline)", rec.provenance),
      ]),
    ]));
    if (rec.state === "blocked") {
      recCard.appendChild(el("div", { class: "held-note", text: "Dosing held by safety layer" }));
      recCard.appendChild(el("p", { style: "color:var(--muted);font-size:.84rem;margin-top:.3rem",
        text: rec.projected }));
    } else {
      recCard.appendChild(el("div", { class: "rec-dose", html: fmt(rec.dose_ml_min, 1) +
        "<span class='u'> mL/min · " + escapeHtml(rec.chemical) + " · " + rec.window_h + " h window</span>" }));
      recCard.appendChild(el("p", { style: "color:var(--muted);font-size:.8rem;margin:.3rem 0 0",
        text: "Projected: " + rec.projected }));
    }
    var ul = el("ul", { class: "rec-rationale" });
    rec.rationale.forEach(function (r) { ul.appendChild(el("li", { text: r })); });
    recCard.appendChild(ul);

    var btns = el("div", { class: "decision-btns" });
    var canDecide = rec.state !== "blocked";
    btns.appendChild(el("button", { class: "btn primary", text: "Approve dose",
      disabled: canDecide ? undefined : "true",
      onclick: function () { S.decisions[S.scenario] = "approved"; logLocalAction("Operator approved the recommended dose"); renderView(); } }));
    btns.appendChild(el("button", { class: "btn danger-outline", text: "Decline",
      disabled: canDecide ? undefined : "true",
      onclick: function () { S.decisions[S.scenario] = "declined"; logLocalAction("Operator declined the recommended dose"); renderView(); } }));
    if (decided) {
      btns.appendChild(el("span", { class: "prov warn", text: decided + " · local demo state",
        title: "Your choice only changes what this page shows — nothing is controlled from here." }));
      btns.appendChild(el("button", { class: "chip-btn", text: "reset",
        onclick: function () { delete S.decisions[S.scenario]; renderView(); } }));
    }
    recCard.appendChild(btns);
    recCard.appendChild(el("p", { style: "font-size:.74rem;color:var(--muted);margin-top:.7rem",
      text: "Both outcome trajectories were computed offline by TGF's simulator and controller. The closed-loop controller is validated in backtest only — in production TGF is advisory, and a human authorizes every dose." }));
    left.appendChild(recCard);

    /* projected trajectories */
    var projCard = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Projected inhibitor residual — next 24 h" }),
        el("div", { class: "right" }, [provChip("controller (offline)", rec.provenance), provChip("backtest-only",
          "The closed-loop controller is validated in backtest only; these trajectories are simulator output, not plant behavior.")]),
      ]),
      el("div", { class: "chart chart-md", id: "advProj" }),
      el("p", { style: "font-size:.72rem;color:var(--muted)", id: "advProjNote",
        text: decided ? (decided === "approved" ? "Showing the approved-dose path against the no-dose path." : "Showing the declined (no-dose) path against the recommended path.")
                      : "Approve or decline to highlight an outcome. Both paths are pre-computed simulation output." }),
    ]);
    left.appendChild(projCard);

    /* dosing timeline */
    var tlCard = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Dosing over the window — controller output per hour" }),
        el("div", { class: "right" }, [provChip("controller (offline)", d.timeline.provenance)]),
      ]),
      el("div", { class: "chart chart-md", id: "advTimeline" }),
    ]);
    left.appendChild(tlCard);
    grid.appendChild(left);

    /* right column: safety checks, inventory, operator log */
    var right = el("div", null);
    var scCard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Safety checks on this recommendation" }),
        el("div", { class: "right" }, [provChip("controller (offline)", "Evaluated by the TGF safety layer in the offline run.")]),
      ]),
      el("div", { class: "checks" }, rec.safety_checks.map(function (c) {
        return el("div", { class: "check " + (c.pass ? "pass" : "fail") }, [
          el("span", { class: "c-ic", text: c.pass ? "✓" : "✕" }),
          el("div", null, [
            el("div", { class: "c-name", text: c.name }),
            el("div", { class: "c-detail", text: c.detail }),
          ]),
        ]);
      })),
    ]);
    right.appendChild(scCard);

    var inv = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Chemical inventory" }),
        el("div", { class: "right" }, [provChip("simulated", d.ops.provenance)]),
      ]),
      el("div", { class: "inv-cards" }, d.ops.inventory.map(function (it) {
        return el("div", { class: "inv" }, [
          el("div", { class: "i-chem", text: it.chemical }),
          el("div", { class: "i-days" + (it.days_left < 7 ? " low" : ""), text: fmt(it.days_left, 0) + " days left" }),
          el("div", { class: "i-meta", text: fmt(it.used_kg, 0) + " kg used of " + fmt(it.stock_start_kg, 0) + " kg · " + it.basis }),
        ]);
      })),
      el("div", { class: "chart chart-sm", id: "advStock", style: "margin-top:.6rem" }),
    ]);
    right.appendChild(inv);

    var logCard = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Operator log" }),
        el("div", { class: "right" }, [provChip("simulated", "Seeded log entries from the simulated run; your own actions are appended locally and never leave this browser.")]),
      ]),
      el("ul", { class: "oplog", id: "advLog" }),
    ]);
    right.appendChild(logCard);
    grid.appendChild(right);
    root.appendChild(grid);

    /* charts */
    var wd = rec.with_dose, wo = rec.without_dose;
    var band = d.params.inhibitor.band;
    var emph = decided === "approved" ? "with" : decided === "declined" ? "without" : null;
    mkChart($("#advProj"), {
      animation: true, animationDuration: 300,
      legend: { bottom: 0, textStyle: { color: cssv("--muted"), fontSize: 10 } },
      grid: { left: 46, right: 14, top: 14, bottom: 44 }, tooltip: tooltipStyle(),
      xAxis: timeAxis(wd.t),
      yAxis: Object.assign({ type: "value", scale: true, name: "ppm",
        nameTextStyle: { color: cssv("--muted"), fontSize: 9 } }, axisStyle()),
      series: [
        { name: "with recommended dose", type: "line", data: wd.inhibitor, symbol: "none",
          lineStyle: { width: emph === "with" ? 2.6 : 1.6, color: cssv("--ok"),
            opacity: emph === "without" ? 0.35 : 1 },
          areaStyle: emph === "with" ? { opacity: 0.1, color: cssv("--ok") } : undefined,
          markArea: bandMarkArea(band) },
        { name: "without dose", type: "line", data: wo.inhibitor, symbol: "none",
          lineStyle: { width: emph === "without" ? 2.6 : 1.6, color: cssv("--bad"),
            opacity: emph === "with" ? 0.35 : 1, type: "dashed" },
          areaStyle: emph === "without" ? { opacity: 0.08, color: cssv("--bad") } : undefined },
      ],
    });

    var blockedAreas = [];
    var tl = d.timeline; var startIx = null;
    tl.blocked.forEach(function (b, k) {
      if (b && startIx === null) startIx = k;
      if ((!b || k === tl.blocked.length - 1) && startIx !== null) {
        blockedAreas.push([{ xAxis: tShort(tl.t[startIx]) }, { xAxis: tShort(tl.t[b ? k : k - 1]) }]);
        startIx = null;
      }
    });
    mkChart($("#advTimeline"), {
      animation: false,
      grid: { left: 46, right: 14, top: 16, bottom: 26 }, tooltip: tooltipStyle(),
      xAxis: timeAxis(tl.t),
      yAxis: Object.assign({ type: "value", name: "mL/min",
        nameTextStyle: { color: cssv("--muted"), fontSize: 9 } }, axisStyle()),
      series: [{
        name: "recommended dose", type: "line", data: tl.dose_ml_min, symbol: "none", step: "end",
        lineStyle: { width: 1.6, color: cssv("--accent") },
        areaStyle: { opacity: 0.12, color: cssv("--accent") },
        markArea: blockedAreas.length ? { silent: true,
          itemStyle: { color: cssv("--bad-soft") },
          label: { show: true, formatter: "dosing held", position: "insideTop", color: cssv("--bad"), fontSize: 9 },
          data: blockedAreas } : undefined,
        markLine: { silent: true, symbol: "none",
          label: { formatter: "sim clock", color: cssv("--muted"), fontSize: 9 },
          lineStyle: { color: cssv("--faint"), type: "dashed" },
          data: [{ xAxis: tShort(d.series.t[i]) }] },
      }],
    });

    var ss = d.ops.inventory_stock_series;
    mkChart($("#advStock"), {
      animation: false,
      legend: { bottom: 0, textStyle: { color: cssv("--muted"), fontSize: 9.5 } },
      grid: { left: 40, right: 10, top: 8, bottom: 40 }, tooltip: tooltipStyle(),
      xAxis: timeAxis(ss.t),
      yAxis: Object.assign({ type: "value", name: "kg",
        nameTextStyle: { color: cssv("--muted"), fontSize: 9 } }, axisStyle()),
      series: [
        { name: "inhibitor stock", type: "line", data: ss.inhibitor, symbol: "none",
          lineStyle: { width: 1.6, color: cssv("--accent") } },
        { name: "biocide stock", type: "line", data: ss.biocide, symbol: "none",
          lineStyle: { width: 1.6, color: cssv("--warn") } },
      ],
    });

    this._renderLog(d);
  },

  _renderLog: function (d) {
    var log = $("#advLog"); if (!log) return;
    log.innerHTML = "";
    var entries = (d.ops.operator_log || []).concat(d._localLog || []);
    entries.sort(function (a, b) { return a.t < b.t ? -1 : 1; });
    entries.slice(-14).reverse().forEach(function (e) {
      var li = el("li", null, [
        el("span", { class: "lt", text: tShort(e.t) }),
        el("span", { text: e.text }),
      ]);
      if (e.source === "local") li.appendChild(el("span", { class: "prov warn", text: "local demo state",
        title: "This entry records your click in this browser session only." }));
      log.appendChild(li);
    });
  },

  update: function (root, d, i) {
    /* only the timeline sim-clock line depends on the clock; re-render is cheap enough on scrub end */
  },
};

/* =============================================================== SAFETY */
Views.safety = {
  render: function (root, d, i) {
    root.appendChild(el("div", { class: "view-head" }, [
      el("h2", { text: "Safety & interlocks" }),
      el("span", { class: "note", text: "Every recommendation passes a layered defense before it is even shown to an operator. Lamps show the state at the sim clock." }),
    ]));

    var pipeCard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Layered defense" }),
        el("div", { class: "right" }, [provChip("controller (offline)",
          "Check states come from the TGF safety layer, evaluated hourly in the offline run.")]),
      ]),
      el("div", { class: "saf-pipe", id: "safPipe" }),
      el("p", { style: "font-size:.76rem;color:var(--muted);margin:.6rem 0 0",
        text: "If any layer fails, the dose is clamped or held entirely — TGF does not dose on data it cannot trust. In production TGF is advisory, and a human authorizes every dose." }),
    ]);
    root.appendChild(pipeCard);

    var grid = el("div", { class: "grid", style: "grid-template-columns:1fr 1fr;margin-top:.9rem" });
    var meters = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Dose limits at the sim clock" }),
        el("div", { class: "right" }, [provChip("controller (offline)", d.timeline.provenance)]),
      ]),
      el("div", { id: "safMeters" }),
    ]);
    grid.appendChild(meters);

    var log = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "Interlock events" }),
        el("div", { class: "right" }, [provChip("simulated", "Derived from the offline run's hourly safety trace.")]),
      ]),
      el("div", { id: "safEvents" }),
    ]);
    grid.appendChild(log);
    root.appendChild(grid);

    this.update(root, d, i);
  },

  update: function (root, d, i) {
    var pipe = $("#safPipe");
    if (pipe) {
      pipe.innerHTML = "";
      var blocked = blockedAt(i);
      d.recommendation.safety_checks.forEach(function (c, ix) {
        var isSensor = c.name.toLowerCase().indexOf("sensor") >= 0;
        var pass = isSensor ? !blocked : c.pass;
        pipe.appendChild(el("div", { class: "saf-node " + (pass ? "pass" : "fail") }, [
          el("span", { class: "lamp" }),
          el("div", { class: "s-name", text: c.name }),
          el("div", { class: "s-detail", text: c.detail }),
        ]));
        if (ix < d.recommendation.safety_checks.length - 1) {
          pipe.appendChild(el("span", { class: "saf-arrow", html: "&#8594;" }));
        }
      });
    }

    var m = $("#safMeters");
    if (m) {
      m.innerHTML = "";
      var tl = d.timeline;
      var dose = tl.dose_ml_min[i], pmax = tl.pump_max_ml_min;
      var cum = tl.cum24_kg[i], cap = tl.daily_cap_kg;
      var mk = function (label, val, max, unit, dec) {
        var frac = clamp(val / max, 0, 1);
        var cls = frac > 0.9 ? "bad" : frac > 0.7 ? "warn" : "";
        var box = el("div", { class: "meter" });
        box.appendChild(el("div", { class: "m-track" }, [
          el("div", { class: "m-fill " + cls, style: "width:" + (frac * 100) + "%" }),
        ]));
        box.appendChild(el("div", { class: "m-lbl" }, [
          el("span", { text: label }),
          el("span", { class: "mono", text: fmt(val, dec) + " / " + fmt(max, dec) + " " + unit }),
        ]));
        return box;
      };
      m.appendChild(mk("Dose rate vs pump ceiling", dose, pmax, "mL/min", 1));
      m.appendChild(mk("Inhibitor dosed in prior 24 h vs daily cap", cum, cap, "kg", 1));
      if (blockedAt(i)) {
        m.appendChild(el("div", { class: "held-note", style: "margin-top:.5rem", text: "Dosing held by safety layer" }));
      }
    }

    var ev = $("#safEvents");
    if (ev) {
      ev.innerHTML = "";
      var tl = d.timeline; var events = [];
      for (var k = 1; k <= i && k < tl.blocked.length; k++) {
        if (tl.blocked[k] && !tl.blocked[k - 1]) events.push({ t: tl.t[k], text: "Sensor-sanity interlock tripped — dosing held", cls: "sev-critical" });
        if (!tl.blocked[k] && tl.blocked[k - 1]) events.push({ t: tl.t[k], text: "Interlock cleared — dosing available again", cls: "sev-warning" });
      }
      (d.alerts || []).forEach(function (a) {
        if (a.source === "safety" && a.t <= d.series.t[i]) events.push({ t: a.t, text: a.text, cls: "sev-critical" });
      });
      if (!events.length) {
        ev.appendChild(el("div", { class: "empty-note", text: "No interlock trips up to the sim clock. The layers are checked every cycle regardless." }));
      } else {
        events.sort(function (a, b) { return a.t < b.t ? 1 : -1; });
        events.forEach(function (e) {
          var card = el("div", { class: "alert-card " + e.cls });
          card.appendChild(el("div", { class: "a-row" }, [el("span", { class: "a-time", text: tShort(e.t) })]));
          card.appendChild(el("div", { class: "a-text", text: e.text }));
          ev.appendChild(card);
        });
      }
    }
  },
};

/* ============================================================== METHODS */
Views.methods = {
  render: function (root, d, i) {
    root.appendChild(el("div", { class: "view-head" }, [
      el("h2", { text: "Data & methods" }),
      el("span", { class: "note", text: "What ran, what didn't, the formulas, and how to rebuild every number on this page." }),
    ]));

    /* what is / is not */
    var isnot = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [el("h3", { text: "What this demo is / is not" })]),
      el("div", { class: "isnot" }, [
        el("div", null, [
          el("div", { class: "label", text: "What it is" }),
          el("ul", null, [
            el("li", { text: "Real TGF code, run offline on synthetic scenarios;" }),
            el("li", { text: "real water-chemistry formulas computed in your browser;" }),
            el("li", { text: "the actual advisory workflow TGF uses." }),
          ]),
        ]),
        el("div", null, [
          el("div", { class: "label", text: "What it is not" }),
          el("ul", null, [
            el("li", { text: "A live plant connection;" }),
            el("li", { text: "autonomous dosing (closed-loop is backtest-only);" }),
            el("li", { text: "real plant data;" }),
            el("li", { text: "a performance claim." }),
          ]),
        ]),
      ]),
    ]);
    root.appendChild(isnot);

    var grid = el("div", { class: "methods-grid", style: "margin-top:.9rem" });

    /* provenance */
    var prov = S.provenance;
    var provCard = el("div", { class: "card" }, [
      el("div", { class: "card-head" }, [
        el("h3", { text: "What ran (provenance)" }),
        el("div", { class: "right" }, [el("a", { class: "chip-btn", href: "data/provenance.json", text: "raw JSON" })]),
      ]),
    ]);
    var ran = el("ul", { class: "ran-list" });
    var whatRan = prov ? prov.what_ran : {
      forecast: d.forecast.provenance, anomalies: d.anomalies_provenance,
      indices: d.indices.provenance, controller: d.recommendation.provenance,
    };
    Object.keys(whatRan).forEach(function (k) {
      ran.appendChild(el("li", null, [
        el("span", { class: "r-what", text: k }),
        el("span", { text: whatRan[k] }),
      ]));
    });
    provCard.appendChild(ran);
    if (prov) {
      provCard.appendChild(el("p", { style: "font-size:.72rem;color:var(--muted);margin-top:.6rem",
        html: "Generated " + escapeHtml(prov.generated) + " · repo commit <code>" +
          escapeHtml((prov.repo_commit || "").slice(0, 10)) + "</code> · Python " + escapeHtml(prov.python || "") }));
      provCard.appendChild(el("p", { style: "font-size:.72rem;color:var(--muted)",
        text: "Forecast anchors: " + (prov.forecast_anchors || "") }));
    }
    provCard.appendChild(el("p", { style: "font-size:.74rem;margin-top:.6rem", html:
      "Regenerate everything: <code>python docs/demo/generate_data.py</code> — deterministic, seeded, no external data." }));
    grid.appendChild(provCard);

    /* formulas + what-if */
    var wi = el("div", { class: "card whatif" });
    wi.appendChild(el("div", { class: "card-head" }, [
      el("h3", { text: "The indices, computed here" }),
      el("div", { class: "right" }, [provChip("computed in-browser",
        "chem.js recomputes LSI/RSI/PSI live from these inputs; the page cross-checks the result against the Python engine's value at load (they agree within 0.01).")]),
    ]));
    wi.appendChild(el("p", { style: "font-size:.78rem;color:var(--muted)",
      text: "These three indices are computed live in your browser from the simulated water chemistry — the same formulas TGF's physics engine uses. Drag the sliders to explore." }));
    var inp = Object.assign({}, d.indices_inputs_now);
    var out = el("div", { class: "gauges3", id: "wiGauges" });
    wi.appendChild(out);
    var mkSlider = function (label, key, min, max, step, unit) {
      var lab = el("label", { text: label + " — ", for: "wi-" + key });
      var valSpan = el("span", { class: "mono", text: fmt(inp[key], step < 1 ? 1 : 0) + (unit || "") });
      lab.appendChild(valSpan);
      var sl = el("input", { type: "range", id: "wi-" + key, min: min, max: max, step: step, value: inp[key] });
      sl.addEventListener("input", function (e) {
        inp[key] = Number(e.target.value);
        valSpan.textContent = fmt(inp[key], step < 1 ? 1 : 0) + (unit || "");
        drawWi();
      });
      wi.appendChild(lab); wi.appendChild(sl);
    };
    function drawWi() {
      var got = Chem.all(inp);
      disposeChartsIn(out);
      out.innerHTML = "";
      [["lsi", "LSI", -2, 2], ["rsi", "RSI", 4, 10], ["psi", "PSI", 4, 10]].forEach(function (def) {
        var cell = el("div", { class: "gauge-cell" });
        var ch = el("div", { class: "g-chart" });
        cell.appendChild(ch);
        cell.appendChild(el("div", { class: "g-verdict", text: idxVerdict(def[0], got[def[0]]) }));
        out.appendChild(cell);
        mkChart(ch, gaugeOpt(clamp(got[def[0]], def[2], def[3]), def[2], def[3], idxZones(def[0]), def[1]));
      });
      var f = $("#wiFormula");
      if (f) {
        var phs = Chem.saturationPH(inp.temp_c, inp.tds, inp.calcium, inp.alkalinity);
        f.innerHTML =
          "pHs = (9.3 + A + B) − (C + D) = <b>" + fmt(phs, 2) + "</b> &nbsp; <span style='color:var(--muted)'>(A: TDS " + fmt(inp.tds, 0) +
          ", B: T " + fmt(inp.temp_c, 1) + " °C, C: Ca " + fmt(inp.calcium, 0) + ", D: alk " + fmt(inp.alkalinity, 0) + ")</span><br>" +
          "LSI = pH − pHs = " + fmt(inp.ph, 2) + " − " + fmt(phs, 2) + " = <b>" + fmt(got.lsi, 2) + "</b> · " + Chem.citations.lsi + "<br>" +
          "RSI = 2·pHs − pH = <b>" + fmt(got.rsi, 2) + "</b> · " + Chem.citations.rsi + "<br>" +
          "PSI = 2·pHs − pHeq(alk) = <b>" + fmt(got.psi, 2) + "</b> · " + Chem.citations.psi;
      }
    }
    mkSlider("Temperature", "temp_c", 15, 45, 0.5, " °C");
    mkSlider("pH", "ph", 6.5, 9.0, 0.05, "");
    wi.appendChild(el("div", { class: "formula", id: "wiFormula" }));
    wi.appendChild(el("p", { style: "font-size:.7rem;color:var(--muted)", html:
      "Formulas per Langelier (1936), Ryznar (1944), Puckorius &amp; Brooke (1991) — the maintainer's " +
      "<a href='https://github.com/Madhvansh/cooling-tower-chem'>cooling-tower-chem</a> library carries the same code the physics engine vendors." }));
    grid.appendChild(wi);
    root.appendChild(grid);
    drawWi();   // draw after the card is attached so the gauges measure a real width

    /* pipeline + stack */
    var pipe = el("div", { class: "card", style: "margin-top:.9rem" }, [
      el("div", { class: "card-head" }, [el("h3", { text: "How this page was made" })]),
      el("div", { class: "pipe-steps" }, [
        el("div", { class: "pipe-step" }, [el("div", { class: "n", text: "1" }),
          el("div", { class: "t", text: "Seeded scenario generator" }),
          el("div", { class: "d", text: "synthetic driving series, plausible dynamics" })]),
        el("div", { class: "pipe-step" }, [el("div", { class: "n", text: "2" }),
          el("div", { class: "t", text: "TGF pipeline" }),
          el("div", { class: "d", text: "indices · Chronos forecasts · anomaly detection · MPC + safety layer" })]),
        el("div", { class: "pipe-step" }, [el("div", { class: "n", text: "3" }),
          el("div", { class: "t", text: "JSON" }),
          el("div", { class: "d", text: "one file per scenario + run provenance" })]),
        el("div", { class: "pipe-step" }, [el("div", { class: "n", text: "4" }),
          el("div", { class: "t", text: "This console" }),
          el("div", { class: "d", text: "static page; chemistry recomputed live in the browser" })]),
      ]),
      el("p", { style: "font-size:.74rem;color:var(--muted);margin-top:.7rem", html:
        "Static page — no backend, no login, no external requests. Charts by Apache ECharts 5.6.0 (Apache-2.0), vendored at <code>vendor/echarts.min.js</code>. " +
        "View source: <a href='https://github.com/Madhvansh/TGF/tree/main/docs/demo'>docs/demo on GitHub</a> · " +
        "<a href='https://github.com/Madhvansh/TGF/blob/main/docs/demo/generate_data.py'>generate_data.py</a>." }),
    ]);
    root.appendChild(pipe);
  },
};

/* ------------------------------------------------------------------ boot */
if (window.__appReady) { bootScenario(); }
