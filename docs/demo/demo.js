/*
 * demo.js — TGF Live Bench app logic.
 *
 * Loads a scenario, paints the tower/KPIs, the live chemistry gauges, the
 * offline forecast + anomaly charts, the advisory dosing card with its safety
 * checklist, and the event feed. Chemistry indices are recomputed in the
 * browser from each scenario's inputs (chem.js); everything else is read from
 * the scenario JSON exactly as the offline pipeline emitted it.
 *
 * Data source: the baseline scenario is inlined in the page for instant paint;
 * the others are fetched on demand. To point the page at freshly generated
 * files, change DATA_BASE below — nothing else moves.
 */
(function () {
  "use strict";

  // --- integration seam: swap fixtures/ for data/ and this is the only edit ---
  var DATA_BASE = "./data/";

  var $ = function (id) { return document.getElementById(id); };
  var params = {}, current = null, currentId = "baseline";
  var scenarioCache = {};
  var inFlight = {};
  var openKpi = null;

  var KPI_KEYS = ["ph", "conductivity", "orp", "inhibitor", "coc", "makeup_hardness"];
  var FORECAST_KEYS = ["conductivity", "ph", "inhibitor"];

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  function boot() {
    initTheme();
    syncBannerHeight();
    window.addEventListener("resize", syncBannerHeight);
    var baseline = readInline("scenario-baseline");
    if (baseline) { scenarioCache.baseline = baseline; }
    wireChips();
    wireTheme();
    wireMath();
    wireWhatIf();
    var provA = $("provLink");
    if (provA) provA.setAttribute("href", DATA_BASE + "provenance.json");

    var initial = scenarioFromHash() || "baseline";
    selectScenario(initial, true);
    window.addEventListener("hashchange", function () {
      var id = scenarioFromHash();
      if (id && id !== currentId) selectScenario(id, false);
    });
  }

  function syncBannerHeight() {
    var b = document.querySelector(".banner");
    if (b) document.documentElement.style.setProperty("--banner-h", b.offsetHeight + "px");
  }

  function readInline(id) {
    var node = document.getElementById(id);
    if (!node) return null;
    try { return JSON.parse(node.textContent); } catch (e) { return null; }
  }

  function scenarioFromHash() {
    var m = /scenario=([a-z]+)/.exec(location.hash || "");
    var id = m && m[1];
    return (id && ["baseline", "scaling", "makeup", "fault"].indexOf(id) >= 0) ? id : null;
  }

  // ---------------------------------------------------------------------------
  // Scenario loading + switching
  // ---------------------------------------------------------------------------
  function selectScenario(id, initial) {
    setChecked(id);
    clearNotice();
    if (scenarioCache[id]) { applyScenario(scenarioCache[id], id, !initial); return; }
    if (inFlight[id]) return;
    inFlight[id] = true;
    fetch(DATA_BASE + "scenario-" + id + ".json", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("http " + r.status); return r.json(); })
      .then(function (data) { scenarioCache[id] = data; inFlight[id] = false; applyScenario(data, id, !initial); })
      .catch(function () {
        inFlight[id] = false;
        showNotice("open via the hosted page to load other scenarios");
        // keep whatever is already shown; revert the radio to the loaded scenario
        setChecked(currentId);
      });
  }

  function setChecked(id) {
    var chips = document.querySelectorAll(".scenario-chip");
    Array.prototype.forEach.call(chips, function (c) {
      c.setAttribute("aria-checked", c.getAttribute("data-scenario") === id ? "true" : "false");
    });
  }

  function applyScenario(data, id, announce) {
    current = data; currentId = id; params = data.params || {};
    var sub = $("scenarioSub");
    if (sub) sub.textContent = data.meta.title + " — " + data.meta.subtitle;
    if (announce && history.replaceState) {
      history.replaceState(null, "", "#scenario=" + id);
    } else if (!location.hash) {
      // leave clean URL on first paint
    }
    var main = $("main");
    if (main) { main.classList.remove("fade-swap"); void main.offsetWidth; main.classList.add("fade-swap"); }

    renderTower();
    renderKpis();
    renderGauges();
    renderMath();
    renderForecasts();
    renderAdvisory();
    renderFeed();
    crossCheck();
  }

  function showNotice(msg) {
    var n = $("fetchNotice");
    if (n) n.innerHTML = '<p class="notice" role="status">' + esc(msg) + "</p>";
  }
  function clearNotice() { var n = $("fetchNotice"); if (n) n.innerHTML = ""; }

  // ---------------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------------
  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("tgf-theme"); } catch (e) {}
    if (saved === "dark" || saved === "light") {
      document.documentElement.setAttribute("data-theme", saved);
    }
    reflectThemeButton();
  }
  function currentTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
  }
  function wireTheme() {
    var btn = $("themeToggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("tgf-theme", next); } catch (e) {}
      reflectThemeButton();
      // gauges/charts pick up CSS vars automatically; re-render for stroke colors baked as attrs
      if (current) { renderGauges(); renderForecasts(); renderAdvisory(); renderKpis(); if (openKpi) reopenKpi(); }
    });
  }
  function reflectThemeButton() {
    var btn = $("themeToggle"), ico = $("themeIcon");
    if (!btn) return;
    var dark = currentTheme() === "dark";
    btn.setAttribute("aria-pressed", dark ? "true" : "false");
    if (ico) ico.textContent = dark ? "☾" : "☀";
  }

  // ---------------------------------------------------------------------------
  // 01 · Tower schematic
  // ---------------------------------------------------------------------------
  function renderTower() {
    var host = $("towerSvg");
    if (!host) return;
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var flow = reduce ? "" : ' class="flow"';
    host.innerHTML =
      '<svg viewBox="0 0 360 240" width="100%" role="img" aria-label="Cooling-tower loop schematic" preserveAspectRatio="xMidYMid meet">' +
      // tower body
      '<rect x="46" y="30" width="120" height="86" rx="6" fill="var(--surface)" stroke="var(--hairline)"/>' +
      '<path d="M60 44 H152 M60 58 H152 M60 72 H152 M60 86 H152 M60 100 H152" stroke="var(--hairline)" stroke-width="1"/>' +
      '<text x="106" y="24" text-anchor="middle" font-size="10" fill="var(--muted)">fill / fans</text>' +
      // fans
      '<circle cx="78" cy="30" r="9" fill="none" stroke="var(--muted)" stroke-width="1.5"/>' +
      '<circle cx="134" cy="30" r="9" fill="none" stroke="var(--muted)" stroke-width="1.5"/>' +
      '<path d="M78 30 L84 26 M78 30 L74 24 M78 30 L72 33" stroke="var(--muted)" stroke-width="1.2"/>' +
      '<path d="M134 30 L140 26 M134 30 L130 24 M134 30 L128 33" stroke="var(--muted)" stroke-width="1.2"/>' +
      // basin
      '<rect x="46" y="120" width="120" height="26" rx="4" fill="var(--band-fill)" stroke="var(--hairline)"/>' +
      '<text x="106" y="137" text-anchor="middle" font-size="10" fill="var(--muted)">basin</text>' +
      // recirc loop to heat load
      '<path d="M166 133 H250 V70 H300"' + flow + ' fill="none" stroke="var(--accent)" stroke-width="2.5"/>' +
      '<rect x="298" y="52" width="46" height="36" rx="4" fill="var(--surface)" stroke="var(--hairline)"/>' +
      '<text x="321" y="74" text-anchor="middle" font-size="9" fill="var(--muted)">heat load</text>' +
      '<path d="M300 88 V150 H166"' + flow + ' fill="none" stroke="var(--accent)" stroke-width="2.5"/>' +
      // makeup in
      '<path d="M8 133 H46"' + flow + ' fill="none" stroke="var(--in-band)" stroke-width="2.5"/>' +
      '<text x="10" y="126" font-size="10" fill="var(--muted)">makeup in</text>' +
      // blowdown out
      '<path d="M106 146 V182 H150"' + flow + ' fill="none" stroke="var(--watch)" stroke-width="2.5"/>' +
      '<text x="120" y="196" font-size="10" fill="var(--muted)">blowdown out</text>' +
      // dosing point marker
      '<circle cx="232" cy="133" r="6" fill="var(--accent)"/>' +
      '<circle cx="232" cy="133" r="10" fill="none" stroke="var(--accent)" stroke-width="1" opacity="0.5"/>' +
      '<text x="232" y="118" text-anchor="middle" font-size="9.5" fill="var(--accent)">dosing point</text>' +
      // recirc pump
      '<circle cx="200" cy="150" r="8" fill="var(--surface)" stroke="var(--muted)"/>' +
      '<text x="200" y="172" text-anchor="middle" font-size="9" fill="var(--muted)">recirc pump</text>' +
      '</svg>';
  }

  // ---------------------------------------------------------------------------
  // 01 · KPI tiles
  // ---------------------------------------------------------------------------
  function bandState(key, v) {
    var b = params[key] && params[key].band;
    if (!b) return "in";
    var lo = b[0], hi = b[1], margin = (hi - lo) * 0.1;
    if (v < lo || v > hi) return "breach";
    if (v < lo + margin || v > hi - margin) return "watch";
    return "in";
  }

  function renderKpis() {
    var host = $("kpis");
    if (!host || !current) return;
    host.innerHTML = "";
    openKpi = null;
    KPI_KEYS.forEach(function (key) {
      var p = params[key]; if (!p) return;
      var arr = current.series[key]; if (!arr) return;
      var v = arr[arr.length - 1];
      var state = bandState(key, v);
      var last24 = arr.slice(-24);
      var prev = arr.length > 24 ? arr[arr.length - 25] : arr[0];
      var d = p.decimals;

      var btn = document.createElement("button");
      btn.className = "kpi";
      btn.type = "button";
      btn.setAttribute("aria-expanded", "false");
      btn.setAttribute("aria-label", p.label + " " + fmt(v, d) + " " + p.unit + ", state " + stateWord(state));

      var top = document.createElement("div"); top.className = "kpi-top";
      top.innerHTML = '<span class="kpi-label">' + esc(p.label) + '</span>' +
        '<span class="state-dot state-' + state + '" aria-hidden="true"></span>';
      btn.appendChild(top);

      var val = document.createElement("div"); val.className = "kpi-val";
      val.innerHTML = '<span class="tnum">' + fmt(v, d) + '</span>' + (p.unit ? '<span class="unit">' + esc(p.unit) + '</span>' : '');
      btn.appendChild(val);

      var band = document.createElement("div"); band.className = "kpi-band";
      band.textContent = "band " + fmt(p.band[0], d) + "–" + fmt(p.band[1], d);
      btn.appendChild(band);

      var mid = document.createElement("div"); mid.className = "kpi-mid";
      var spark = Charts.sparkline(last24, { width: 108, height: 26, band: p.band, color: stateColor(state) });
      mid.appendChild(spark);
      mid.appendChild(deltaChip(v, prev, p));
      btn.appendChild(mid);

      var foot = document.createElement("div"); foot.className = "kpi-foot";
      foot.innerHTML = provChip("simulated", "series") + '<span class="kpi-band">click to expand</span>';
      btn.appendChild(foot);

      btn.addEventListener("click", function () { toggleKpi(key, btn); });
      host.appendChild(btn);
    });
  }

  function deltaChip(cur, prev, p) {
    var span = document.createElement("span");
    var diff = cur - prev;
    var center = (p.band[0] + p.band[1]) / 2;
    var away = Math.abs(cur - center) - Math.abs(prev - center);
    var thresh = (p.band[1] - p.band[0]) * 0.02;
    var cls = "flat", title = "little change vs prior 24 h";
    if (away > thresh) { cls = "up"; title = "drifting toward a guideline limit"; }
    else if (away < -thresh) { cls = "down"; title = "settling toward mid-band"; }
    var arrow = diff > 0.0001 ? "▲" : (diff < -0.0001 ? "▼" : "▬");
    span.className = "delta " + cls;
    span.title = title;
    span.textContent = arrow + " " + (diff >= 0 ? "+" : "") + fmt(diff, p.decimals) + " /24h";
    return span;
  }

  function toggleKpi(key, btn) {
    if (openKpi && openKpi.key === key) { closeKpi(); return; }
    closeKpi();
    var p = params[key];
    var panel = document.createElement("div");
    panel.className = "kpi-expand";
    panel.innerHTML = '<h3>' + esc(p.label) + ' · 7-day trend <span class="prov" data-kind="simulated">simulated</span></h3>';
    var chartHost = document.createElement("div");
    panel.appendChild(chartHost);
    btn.parentNode.insertBefore(panel, btn.nextSibling);
    btn.setAttribute("aria-expanded", "true");
    Charts.lineChart(chartHost, {
      t: current.series.t,
      series: [{ values: current.series[key], color: "var(--accent)", label: p.label }],
      band: p.band, yDecimals: p.decimals, height: 210,
      ariaLabel: p.label + " 7-day trend with guideline band",
      summary: p.label + " over 7 simulated days; guideline band " + fmt(p.band[0], p.decimals) + " to " + fmt(p.band[1], p.decimals) + " " + p.unit + "."
    });
    openKpi = { key: key, btn: btn, panel: panel };
    wireProv(panel);
    try { panel.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "nearest" }); } catch (e) {}
  }
  function reopenKpi() { if (openKpi) { var k = openKpi.key, b = openKpi.btn; closeKpi(); var btns = document.querySelectorAll(".kpi"); /* re-find */ Array.prototype.forEach.call(btns, function (x) { if (x.getAttribute("aria-label").indexOf(params[k].label) === 0) toggleKpi(k, x); }); } }
  function closeKpi() {
    if (!openKpi) return;
    if (openKpi.panel && openKpi.panel.parentNode) openKpi.panel.parentNode.removeChild(openKpi.panel);
    if (openKpi.btn) openKpi.btn.setAttribute("aria-expanded", "false");
    openKpi = null;
  }

  // ---------------------------------------------------------------------------
  // 02 · Gauges + math + what-if
  // ---------------------------------------------------------------------------
  function gaugeInputs() {
    var inp = current.indices_inputs_now;
    return {
      ph: parseFloat($("wiPh").value || inp.ph),
      temp_c: parseFloat($("wiTemp").value || inp.temp_c),
      tds: inp.tds, calcium: inp.calcium, alkalinity: inp.alkalinity
    };
  }

  function renderGauges() {
    var host = $("gauges");
    if (!host || !current) return;
    var inp = current.indices_inputs_now;
    // sync sliders to scenario defaults
    if ($("wiPh")) { $("wiPh").value = inp.ph; $("wiPhVal").textContent = fmt(inp.ph, 2); }
    if ($("wiTemp")) { $("wiTemp").value = inp.temp_c; $("wiTempVal").textContent = fmt(inp.temp_c, 1) + " °C"; }
    host.innerHTML =
      gaugeCard("lsi", "LSI") + gaugeCard("rsi", "RSI") + gaugeCard("psi", "PSI");
    drawGauges(gaugeInputs());
    wireProv(host);
  }
  function gaugeCard(key, title) {
    return '<div class="gauge-card"><h3>' + title + '</h3>' +
      '<div id="gauge-' + key + '"></div>' +
      '<div style="margin-top:.4rem"><span class="prov" data-kind="browser" data-prov="browser">computed in-browser</span></div></div>';
  }
  function drawGauges(inp) {
    var vals = Chem.all(inp);
    gaugeOne("lsi", "LSI", vals.lsi, -2, 2, 2,
      [{ to: -0.5, color: "var(--breach)" }, { to: 0.5, color: "var(--in-band)" }, { to: 2, color: "var(--watch)" }],
      Chem.lsiVerdict(vals.lsi));
    gaugeOne("rsi", "RSI", vals.rsi, 4, 10, 2,
      [{ to: 6, color: "var(--watch)" }, { to: 7, color: "var(--in-band)" }, { to: 10, color: "var(--breach)" }],
      Chem.rsiVerdict(vals.rsi));
    gaugeOne("psi", "PSI", vals.psi, 4, 10, 2,
      [{ to: 6, color: "var(--watch)" }, { to: 7, color: "var(--in-band)" }, { to: 10, color: "var(--breach)" }],
      Chem.psiVerdict(vals.psi));
    return vals;
  }
  function gaugeOne(key, label, value, min, max, decimals, zones, verdict) {
    var host = $("gauge-" + key);
    if (host) Charts.gauge(host, { value: value, min: min, max: max, decimals: decimals, zones: zones, label: label, verdict: verdict });
  }

  function wireMath() {
    var btn = $("mathToggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var panel = $("mathPanel");
      var open = !panel.hidden;
      panel.hidden = open;
      btn.setAttribute("aria-expanded", open ? "false" : "true");
      btn.textContent = open ? "Show the math" : "Hide the math";
      if (!open) renderMath();
    });
  }
  function renderMath() {
    var panel = $("mathPanel");
    if (!panel || panel.hidden || !current) return;
    var inp = gaugeInputs();
    var A = (Math.log10(inp.tds) - 1) / 10;
    var B = -13.12 * Math.log10(inp.temp_c + 273.15) + 34.55;
    var C = Math.log10(inp.calcium) - 0.4;
    var D = Math.log10(inp.alkalinity);
    var pHs = (9.3 + A + B) - (C + D);
    var pHeq = 1.465 * Math.log10(inp.alkalinity) + 4.54;
    var v = Chem.all(inp);
    panel.innerHTML =
      mathRow("Saturation pH (shared term)",
        "pHs = (9.3 + A + B) − (C + D)  where  A=" + f(A, 3) + ", B=" + f(B, 3) + ", C=" + f(C, 3) + ", D=" + f(D, 3),
        "= (9.3 + " + f(A, 3) + " + " + f(B, 3) + ") − (" + f(C, 3) + " + " + f(D, 3) + ") = " + f(pHs, 3),
        "A=(log₁₀TDS−1)/10, B=−13.12·log₁₀(T+273.15)+34.55, C=log₁₀(Ca)−0.4, D=log₁₀(Alk); TDS=" + f(inp.tds, 0) + ", T=" + f(inp.temp_c, 1) + "°C, Ca=" + f(inp.calcium, 0) + ", Alk=" + f(inp.alkalinity, 0) + " ppm CaCO₃") +
      mathRow("LSI — Langelier Saturation Index",
        "LSI = pH − pHs = " + f(inp.ph, 2) + " − " + f(pHs, 3),
        "= " + f(v.lsi, 2) + "  (" + Chem.lsiVerdict(v.lsi) + ")",
        "Langelier (1936)") +
      mathRow("RSI — Ryznar Stability Index",
        "RSI = 2·pHs − pH = 2·" + f(pHs, 3) + " − " + f(inp.ph, 2),
        "= " + f(v.rsi, 2) + "  (" + Chem.rsiVerdict(v.rsi) + ")",
        "Ryznar (1944)") +
      mathRow("PSI — Puckorius Scaling Index",
        "PSI = 2·pHs − pHeq,  pHeq = 1.465·log₁₀(Alk) + 4.54 = " + f(pHeq, 3),
        "= 2·" + f(pHs, 3) + " − " + f(pHeq, 3) + " = " + f(v.psi, 2) + "  (" + Chem.psiVerdict(v.psi) + ")",
        "Puckorius &amp; Brooke (1991)") +
      '<p class="cite" style="margin-top:.6rem">Same closed-form indices as the vendored ' +
      '<a href="https://github.com/Madhvansh/TGF" rel="noopener">cooling-tower-chem</a> library the TGF physics engine imports.</p>';
  }
  function mathRow(title, formula, result, cite) {
    return '<div class="math-row"><div style="font-weight:600;font-size:.86rem;margin-bottom:.3rem">' + title + '</div>' +
      '<div class="formula">' + formula + '<br>' + result + '</div>' +
      '<div class="cite">' + cite + '</div></div>';
  }

  function wireWhatIf() {
    ["wiPh", "wiTemp"].forEach(function (id) {
      var el = $(id);
      if (!el) return;
      el.addEventListener("input", function () {
        if ($("wiPhVal")) $("wiPhVal").textContent = fmt(parseFloat($("wiPh").value), 2);
        if ($("wiTempVal")) $("wiTempVal").textContent = fmt(parseFloat($("wiTemp").value), 1) + " °C";
        if (current) { drawGauges(gaugeInputs()); renderMath(); }
      });
    });
    var reset = $("wiReset");
    if (reset) reset.addEventListener("click", function (e) {
      e.preventDefault();
      if (!current) return;
      var inp = current.indices_inputs_now;
      $("wiPh").value = inp.ph; $("wiTemp").value = inp.temp_c;
      $("wiPhVal").textContent = fmt(inp.ph, 2);
      $("wiTempVal").textContent = fmt(inp.temp_c, 1) + " °C";
      drawGauges(gaugeInputs()); renderMath();
    });
  }

  // ---------------------------------------------------------------------------
  // 03 · Forecast + anomalies
  // ---------------------------------------------------------------------------
  function renderForecasts() {
    var host = $("forecastCharts");
    if (!host || !current) return;
    host.innerHTML = "";
    var nowT = current.series.t[current.series.t.length - 1];
    FORECAST_KEYS.forEach(function (key) {
      var p = params[key]; if (!p) return;
      var fc = current.forecast && current.forecast.params[key];
      var card = document.createElement("div"); card.className = "fchart";
      var head = document.createElement("div"); head.className = "fchart-head";
      head.innerHTML = '<h3>' + esc(p.label) + (p.unit ? ' <span style="color:var(--muted);font-weight:400">(' + esc(p.unit) + ')</span>' : '') + '</h3>' +
        computedPopover(key);
      card.appendChild(head);

      var legend = document.createElement("div"); legend.className = "legend";
      legend.innerHTML =
        '<i><span class="sw hist"></span>history</i>' +
        '<i><span class="sw med"></span>forecast median</i>' +
        '<i><span class="sw cone"></span>q10–q90</i>' +
        '<i><span class="dia">◆</span> anomaly</i>' +
        '<span class="prov" data-kind="simulated" data-prov="series" style="margin-left:auto">simulated</span>';
      card.appendChild(legend);

      var chartHost = document.createElement("div");
      card.appendChild(chartHost);
      host.appendChild(card);

      var markers = anomalyMarkers(key);
      Charts.lineChart(chartHost, {
        t: current.series.t,
        series: [{ values: current.series[key], color: "var(--ink)", label: p.label }],
        band: p.band, yDecimals: p.decimals, height: 200,
        nowT: nowT,
        forecast: fc ? { t: fc.t, q50: fc.q50, q10: fc.q10, q90: fc.q90 } : null,
        markers: markers,
        ariaLabel: p.label + " history, guideline band, 24-hour forecast cone and anomaly flags",
        summary: forecastSummary(p, key, fc, markers)
      });
      wireProv(card);
      wirePopover(card);
    });
  }

  function anomalyMarkers(key) {
    if (!current.anomalies) return [];
    var tIndex = {}; current.series.t.forEach(function (t, i) { tIndex[t] = i; });
    return current.anomalies.filter(function (a) { return a.param === key; }).map(function (a) {
      var i = tIndex[a.t];
      var val = (i != null) ? current.series[key][i] : null;
      return { t: a.t, value: val, severity: a.severity, score: a.score, note: a.note, param: a.param };
    }).filter(function (m) { return m.value != null; });
  }

  function forecastSummary(p, key, fc, markers) {
    var arr = current.series[key];
    var s = p.label + ": last value " + fmt(arr[arr.length - 1], p.decimals) + " " + p.unit +
      ", guideline band " + fmt(p.band[0], p.decimals) + " to " + fmt(p.band[1], p.decimals) + ".";
    if (fc) s += " 24-hour forecast median ends near " + fmt(fc.q50[fc.q50.length - 1], p.decimals) + ".";
    if (markers.length) s += " " + markers.length + " anomaly flag" + (markers.length > 1 ? "s" : "") + " on this series.";
    return s;
  }

  function computedPopover(key) {
    var pid = "pop-" + key;
    return '<span class="popover"><button class="btn" type="button" aria-expanded="false" aria-controls="' + pid + '" data-pop="' + pid + '">How was this computed?</button>' +
      '<div class="popover-body" id="' + pid + '" hidden>' +
      '<b>Series:</b> ' + esc(current.series_provenance) + '<br><br>' +
      '<b>Forecast:</b> ' + esc(current.forecast ? current.forecast.provenance : "n/a") + '<br><br>' +
      '<b>Anomalies:</b> ' + esc(current.anomalies_provenance || "none flagged") + '</div></span>';
  }
  function wirePopover(scope) {
    var btns = scope.querySelectorAll("[data-pop]");
    Array.prototype.forEach.call(btns, function (b) {
      b.addEventListener("click", function () {
        var body = document.getElementById(b.getAttribute("data-pop"));
        var open = !body.hidden;
        body.hidden = open;
        b.setAttribute("aria-expanded", open ? "false" : "true");
      });
    });
  }

  // ---------------------------------------------------------------------------
  // 04 · Advisory dosing decision
  // ---------------------------------------------------------------------------
  var decideState = "none";
  function renderAdvisory() {
    var host = $("recCard");
    if (!host || !current) return;
    var r = current.recommendation;
    var blocked = r.state === "blocked";
    host.className = "rec-card" + (blocked ? " blocked" : "");

    var head;
    if (blocked) {
      head = '<div class="rec-headline"><span class="held-badge">Dosing held by safety layer</span></div>' +
        '<div class="rec-meta">' + esc(r.chemical) + ' · recommendation withheld</div>';
    } else {
      head = '<div class="rec-headline"><span class="rec-dose"><span class="tnum">' + fmt(r.dose_ml_min, 1) +
        '</span><span class="unit"> mL/min</span></span></div>' +
        '<div class="rec-meta">' + esc(r.chemical) + ' · dosing window ' + r.window_h + ' h</div>';
    }

    var rationale = '<ul class="rationale">' + r.rationale.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join("") + '</ul>';
    var projected = '<div class="projected">' + esc(r.projected) + '</div>';

    var checks = '<ul class="checklist" aria-label="Safety checks">' + r.safety_checks.map(function (c) {
      return '<li><span class="check-ico ' + (c.pass ? "ok" : "fail") + '" aria-hidden="true">' + (c.pass ? "✓" : "✕") + '</span>' +
        '<span><span class="check-name">' + esc(c.name) + '</span> — <span class="check-detail">' + esc(c.detail) + '</span>' +
        '<span class="visually-hidden">' + (c.pass ? "passed" : "failed") + '</span></span></li>';
    }).join("") + '</ul>';

    var buttons = '<div class="decide">' +
      '<button class="btn primary" id="approveBtn" type="button"' + (blocked ? " disabled" : "") + '>Approve</button>' +
      '<button class="btn" id="declineBtn" type="button"' + (blocked ? " disabled" : "") + '>Decline</button>' +
      '<a href="#" id="resetDecide" class="overlay-note">reset</a>' +
      '<span class="prov" data-kind="controller" data-prov="rec">controller (offline)</span>' +
      '<span class="prov" data-kind="backtest" data-prov="rec">backtest-only</span></div>';

    host.innerHTML = head + rationale + projected + checks + buttons;
    wireProv(host);

    decideState = "none";
    drawMini();
    wireDecide(blocked);
  }

  function wireDecide(blocked) {
    var a = $("approveBtn"), d = $("declineBtn"), reset = $("resetDecide");
    if (a && !blocked) a.addEventListener("click", function () { decideState = "approve"; drawMini(); updateOverlayNote(); });
    if (d && !blocked) d.addEventListener("click", function () { decideState = "decline"; drawMini(); updateOverlayNote(); });
    if (reset) reset.addEventListener("click", function (e) { e.preventDefault(); decideState = "none"; drawMini(); updateOverlayNote(); });
    updateOverlayNote(blocked);
  }
  function updateOverlayNote(blocked) {
    var n = $("overlayNote"); if (!n || !current) return;
    var r = current.recommendation;
    if (r.state === "blocked") { n.textContent = "Dosing held — no trajectory to authorize."; return; }
    var w = r.with_dose, wo = r.without_dose;
    if (decideState === "approve") n.textContent = "Approved (advisory): with-dose inhibitor ends ≈ " + fmt(last(w.inhibitor), 1) + " ppm, LSI ≈ " + fmt(last(w.lsi), 2) + ".";
    else if (decideState === "decline") n.textContent = "Declined: without-dose inhibitor ends ≈ " + fmt(last(wo.inhibitor), 1) + " ppm, LSI ≈ " + fmt(last(wo.lsi), 2) + ".";
    else n.textContent = "Approve or decline to overlay the projected trajectory. A human authorizes every dose.";
  }

  function drawMini() {
    var host = $("miniChart"); if (!host || !current) return;
    var r = current.recommendation;
    var w = r.with_dose, wo = r.without_dose;
    var p = params.inhibitor;
    var series = [
      { values: wo.inhibitor, color: "var(--muted)", label: "without dose", width: 1.4, dash: "4 3" },
      { values: w.inhibitor, color: "var(--accent)", label: "with dose", width: 1.4, dash: "4 3" }
    ];
    var overlays = [];
    if (decideState === "approve") overlays = [{ values: w.inhibitor, color: "var(--accent)", label: "with dose (approved)", dash: null }];
    else if (decideState === "decline") overlays = [{ values: wo.inhibitor, color: "var(--breach)", label: "without dose (declined)", dash: null }];
    Charts.lineChart(host, {
      t: w.t, series: series, overlays: overlays, overlayT: w.t,
      band: p.band, yDecimals: 1, height: 200, nowT: w.t[0],
      ariaLabel: "Projected inhibitor residual over the 24-hour decision horizon, with and without the recommended dose",
      summary: "Inhibitor residual over 24 h. Without dose ends near " + fmt(last(wo.inhibitor), 1) +
        " ppm; with dose ends near " + fmt(last(w.inhibitor), 1) + " ppm. Guideline band " + fmt(p.band[0], 1) + " to " + fmt(p.band[1], 1) + " ppm. Backtest trajectories."
    });
  }

  // ---------------------------------------------------------------------------
  // 05 · Event feed
  // ---------------------------------------------------------------------------
  var feedExpanded = false;
  function renderFeed() {
    var host = $("feed"); if (!host || !current) return;
    var items = [];
    (current.alerts || []).forEach(function (a) {
      items.push({ t: a.t, severity: a.severity, text: a.text, source: a.source });
    });
    (current.anomalies || []).forEach(function (a) {
      items.push({ t: a.t, severity: a.severity === "low" ? "info" : "warning",
        text: "Anomaly · " + a.param + " — " + a.note + " (score " + a.score + ")", source: "anomaly" });
    });
    items.sort(function (x, y) { return Date.parse(y.t) - Date.parse(x.t); });

    var LIMIT = 8;
    var shown = feedExpanded ? items : items.slice(0, LIMIT);
    host.innerHTML = shown.length ? shown.map(feedRow).join("") :
      '<li><span class="sev info"></span><span class="txt">No events in this scenario window.</span></li>';

    var more = $("feedMore");
    if (more) {
      if (items.length > LIMIT) {
        more.hidden = false;
        more.textContent = feedExpanded ? "Show fewer" : "Show all (" + items.length + ")";
        more.onclick = function () { feedExpanded = !feedExpanded; renderFeed(); };
      } else { more.hidden = true; }
    }
  }
  function feedRow(e) {
    return '<li><span class="sev ' + e.severity + '" aria-hidden="true"></span>' +
      '<time>' + shortTime(e.t) + '</time>' +
      '<span class="txt">' + esc(e.text) + '</span>' +
      '<span class="src">' + esc(e.source) + '</span></li>';
  }

  // ---------------------------------------------------------------------------
  // Provenance chip popovers (shared explanations from JSON)
  // ---------------------------------------------------------------------------
  function provChip(kind, ref) {
    var label = { simulated: "simulated", browser: "computed in-browser", forecast: "forecast (offline)",
      anomaly: "anomaly (offline)", controller: "controller (offline)", backtest: "backtest-only" }[kind] || kind;
    return '<span class="prov" data-kind="' + kind + '" data-prov="' + (ref || kind) + '">' + label + '</span>';
  }
  function provText(ref) {
    if (!current) return "";
    switch (ref) {
      case "series": return current.series_provenance;
      case "browser": return "LSI / RSI / PSI recomputed in your browser from the simulated inputs (chem.js).";
      case "forecast": return current.forecast ? current.forecast.provenance : "";
      case "anomaly": return current.anomalies_provenance || "";
      case "rec": return current.recommendation ? current.recommendation.provenance : "";
      default: return "";
    }
  }
  function wireProv(scope) {
    var chips = (scope || document).querySelectorAll(".prov[data-prov]");
    Array.prototype.forEach.call(chips, function (c) {
      if (c._wired) return; c._wired = true;
      var txt = provText(c.getAttribute("data-prov"));
      if (txt) {
        c.setAttribute("tabindex", "0");
        c.setAttribute("title", txt);
        c.setAttribute("aria-label", c.textContent + ": " + txt);
        var show = function (ev) { var r = c.getBoundingClientRect(); Charts._showTip(esc(txt), r.left + r.width / 2, r.bottom + 8); };
        c.addEventListener("mouseenter", show);
        c.addEventListener("focus", show);
        c.addEventListener("mouseleave", Charts._hideTip);
        c.addEventListener("blur", Charts._hideTip);
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Cross-check chem.js vs the Python engine values baked into the JSON
  // ---------------------------------------------------------------------------
  function crossCheck() {
    if (!current || !current.indices_now || !window.console || !console.assert) return;
    var v = Chem.all(current.indices_inputs_now), n = current.indices_now;
    console.assert(Math.abs(v.lsi - n.lsi) < 0.01, "LSI mismatch", v.lsi, n.lsi);
    console.assert(Math.abs(v.rsi - n.rsi) < 0.01, "RSI mismatch", v.rsi, n.rsi);
    console.assert(Math.abs(v.psi - n.psi) < 0.01, "PSI mismatch", v.psi, n.psi);
  }

  // ---------------------------------------------------------------------------
  // Chips wiring + keyboard
  // ---------------------------------------------------------------------------
  function wireChips() {
    var chips = document.querySelectorAll(".scenario-chip");
    Array.prototype.forEach.call(chips, function (c, i) {
      c.addEventListener("click", function () { selectScenario(c.getAttribute("data-scenario"), false); });
      c.addEventListener("keydown", function (e) {
        var arr = Array.prototype.slice.call(chips);
        var idx = arr.indexOf(c), next = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") next = arr[(idx + 1) % arr.length];
        else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = arr[(idx - 1 + arr.length) % arr.length];
        if (next) { e.preventDefault(); next.focus(); selectScenario(next.getAttribute("data-scenario"), false); }
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------
  function fmt(v, d) { return (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d == null ? 1 : d); }
  function f(v, d) { return fmt(v, d); }
  function last(a) { return a[a.length - 1]; }
  function stateWord(s) { return s === "in" ? "in band" : s === "watch" ? "watch" : "out of band"; }
  function stateColor(s) { return s === "in" ? "var(--in-band)" : s === "watch" ? "var(--watch)" : "var(--breach)"; }
  function reducedMotion() { return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches; }
  function shortTime(t) {
    var d = new Date(t);
    return (d.getUTCMonth() + 1) + "/" + d.getUTCDate() + " " + String(d.getUTCHours()).padStart(2, "0") + ":00";
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (m) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m];
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
