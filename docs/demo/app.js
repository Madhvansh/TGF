/* TGF Console — shell, state, playback, alerts drawer.
   Static page; all data is pre-computed simulation output (see Data & Methods). */
"use strict";

/* ---------------------------------------------------------------- helpers */
var $ = function (s, r) { return (r || document).querySelector(s); };
var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

function el(tag, attrs, children) {
  var n = document.createElement(tag);
  if (attrs) Object.keys(attrs).forEach(function (k) {
    if (k === "class") n.className = attrs[k];
    else if (k === "html") n.innerHTML = attrs[k];
    else if (k === "text") n.textContent = attrs[k];
    else if (k.indexOf("on") === 0) n.addEventListener(k.slice(2), attrs[k]);
    else n.setAttribute(k, attrs[k]);
  });
  (children || []).forEach(function (c) { if (c) n.appendChild(c); });
  return n;
}
function fmt(v, dec) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return Number(v).toFixed(dec === undefined ? 1 : dec);
}
function cssv(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function tShort(isoStr) {
  var d = new Date(isoStr);
  var M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return M[d.getUTCMonth()] + " " + d.getUTCDate() + " · " +
         String(d.getUTCHours()).padStart(2, "0") + ":00";
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

/* ------------------------------------------------------------------ state */
var SCENARIO_IDS = ["baseline", "scaling", "makeup", "fault"];
var VIEW_IDS = ["overview", "twin", "analytics", "forecast", "advisor", "safety", "methods"];

var S = {
  scenario: "baseline",
  view: "overview",
  data: {},               // id -> scenario doc
  simNow: {},             // id -> hour index (0-based into series)
  playing: false,
  speed: 12,              // sim-hours per real second
  acks: {},               // id -> {alertKey: true}   (local demo state only)
  decisions: {},          // id -> "approved" | "declined" | null (local demo state)
  charts: [],             // live echarts instances for the current view
  raf: null, lastTick: 0,
  provenance: null,
};

function D() { return S.data[S.scenario]; }
function NOW() { return S.simNow[S.scenario] !== undefined ? S.simNow[S.scenario] : 167; }

/* -------------------------------------------------------------- charts reg */
function mkChart(dom, opt) {
  var c = echarts.init(dom, null, { renderer: "canvas" });
  c.setOption(opt);
  S.charts.push(c);
  return c;
}
function disposeCharts() {
  S.charts.forEach(function (c) { try { c.dispose(); } catch (e) {} });
  S.charts = [];
}
window.addEventListener("resize", debounce(function () {
  S.charts.forEach(function (c) { try { c.resize(); } catch (e) {} });
}, 150));
function debounce(fn, ms) {
  var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); };
}

/* Shared echarts fragments */
function baseGrid() {
  return { left: 52, right: 16, top: 24, bottom: 26, containLabel: false };
}
function axisStyle() {
  return {
    axisLine: { lineStyle: { color: cssv("--line") } },
    axisLabel: { color: cssv("--muted"), fontSize: 10.5 },
    splitLine: { lineStyle: { color: cssv("--line-soft") } },
    axisTick: { show: false },
  };
}
function timeAxis(tArr) {
  var a = axisStyle();
  return Object.assign({ type: "category", data: tArr.map(tShort), boundaryGap: false,
    axisLabel: Object.assign(a.axisLabel, { interval: Math.max(1, Math.floor(tArr.length / 6)) }) }, a);
}
function tooltipStyle() {
  return {
    trigger: "axis",
    backgroundColor: cssv("--card-hi"), borderColor: cssv("--line"),
    textStyle: { color: cssv("--ink"), fontSize: 11.5 },
    axisPointer: { type: "cross", label: { backgroundColor: cssv("--panel"), color: cssv("--ink") },
                   lineStyle: { color: cssv("--faint") }, crossStyle: { color: cssv("--faint") } },
  };
}
function bandMarkArea(band) {
  return { silent: true, itemStyle: { color: cssv("--accent-soft"), opacity: 0.45 },
           data: [[{ yAxis: band[0] }, { yAxis: band[1] }]] };
}

/* ------------------------------------------------------------- data logic */
function valueAt(key, i) {
  var d = D(); var arr = d.series[key];
  return arr ? arr[Math.min(i, arr.length - 1)] : null;
}
function bandState(v, band) {
  if (v === null || v === undefined || !isFinite(v)) return "bad";
  if (v < band[0] || v > band[1]) return "bad";
  var span = band[1] - band[0];
  if (v < band[0] + 0.07 * span || v > band[1] - 0.07 * span) return "warn";
  return "ok";
}
function faultActiveAt(i) {
  var d = D();
  if (!d.sensor_fault) return false;
  var idx = d.series.t.indexOf(d.sensor_fault.stuck_from);
  return idx >= 0 && i >= idx;
}
function dropoutActiveAt(i) {
  var d = D();
  if (!d.sensor_fault) return false;
  var idx = d.series.t.indexOf(d.sensor_fault.dropout_from);
  return idx >= 0 && i >= idx;
}
function kindOf(key, i) {
  var d = D();
  if (d.sensor_fault && d.sensor_fault.param === key && faultActiveAt(i)) return "fault";
  return (d.params[key] && d.params[key].kind) || "estimated";
}
function blockedAt(i) {
  var d = D(); return !!(d.timeline && d.timeline.blocked[Math.min(i, d.timeline.blocked.length - 1)]);
}
function anchorFor(i) {
  var d = D(); var best = d.forecast.anchors[0];
  d.forecast.anchors.forEach(function (a) { if (a.h <= i + 1) best = a; });
  return best;
}
function verdictAt(i) {
  var d = D();
  if (blockedAt(i)) return { cls: "state-bad", text: "DOSING HELD — sensor fault",
    sub: "The pH probe failed sanity checks; the safety layer holds all dosing until it is verified." };
  var lsi = d.indices.lsi[i], rsi = d.indices.rsi[i];
  if (lsi > 1.0) return { cls: "state-warn", text: "Scaling risk building",
    sub: "LSI " + fmt(lsi, 2) + " — the loop is concentrating toward the scaling zone." };
  if (lsi < -0.3 && rsi > 7.5) return { cls: "state-warn", text: "Corrosion watch",
    sub: "LSI " + fmt(lsi, 2) + ", RSI " + fmt(rsi, 1) + " — the water has a corrosive tendency." };
  return { cls: "state-ok", text: "Stable operation",
    sub: "Chemistry indices sit mid-band; the advisory controller recommends only maintenance dosing." };
}
function alertKey(a) { return a.t + "|" + a.text; }
function visibleAlerts(i) {
  var d = D(); var tNow = d.series.t[i];
  var list = (d.alerts || []).filter(function (a) { return a.t <= tNow; });
  return list.map(function (a) {
    var resolved = a.resolved_t && a.resolved_t <= tNow;
    return Object.assign({}, a, { state: resolved ? "resolved" : "active", key: alertKey(a) });
  }).reverse();
}

/* -------------------------------------------------------------- popovers */
var popNode = null;
function showPop(html, x, y) {
  hidePop();
  popNode = el("div", { class: "pop", role: "tooltip", html: html });
  document.body.appendChild(popNode);
  var r = popNode.getBoundingClientRect();
  popNode.style.left = clamp(x - r.width / 2, 8, window.innerWidth - r.width - 8) + "px";
  popNode.style.top = (y + 14 + r.height > window.innerHeight ? y - r.height - 10 : y + 14) + "px";
}
function hidePop() { if (popNode) { popNode.remove(); popNode = null; } }
document.addEventListener("click", function (e) {
  var t = e.target.closest("[data-pop]");
  if (t) {
    e.preventDefault();
    showPop(t.getAttribute("data-pop"), e.clientX, e.clientY);
    e.stopPropagation();
  } else hidePop();
});
document.addEventListener("keydown", function (e) { if (e.key === "Escape") { hidePop(); closeDrawer(); closeSlideover(); } });
window.addEventListener("scroll", hidePop, true);

/* ---------------------------------------------------------------- loading */
function parseInline() {
  var node = $("#scenario-baseline");
  try { return JSON.parse(node.textContent); } catch (e) { return null; }
}
function loadScenario(id) {
  if (S.data[id]) return Promise.resolve(S.data[id]);
  if (id === "baseline") {
    var d = parseInline();
    if (d && d.meta) { S.data.baseline = d; return Promise.resolve(d); }
  }
  return fetch("./data/scenario-" + id + ".json").then(function (r) {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  }).then(function (d) { S.data[id] = d; return d; });
}
function loadProvenance() {
  return fetch("./data/provenance.json").then(function (r) { return r.ok ? r.json() : null; })
    .then(function (p) { S.provenance = p; return p; }).catch(function () { return null; });
}
function crossCheck(d) {
  if (!window.Chem || !d.indices_inputs_now || !d.indices_now) return;
  var inp = d.indices_inputs_now;
  var got = Chem.all(inp);
  ["lsi", "rsi", "psi"].forEach(function (k) {
    console.assert(Math.abs(got[k] - d.indices_now[k]) < 0.01,
      "index cross-check drifted for " + k, got[k], d.indices_now[k]);
  });
}

/* ---------------------------------------------------------------- shell UI */
function renderScenarioChips() {
  var box = $("#scChips"); box.innerHTML = "";
  var titles = { baseline: "Baseline", scaling: "Scaling excursion", makeup: "Corrosive makeup", fault: "Sensor-fault drill" };
  SCENARIO_IDS.forEach(function (id) {
    box.appendChild(el("button", {
      class: "sc-chip", role: "radio", "aria-checked": String(S.scenario === id),
      text: titles[id],
      onclick: function () { switchScenario(id); },
    }));
  });
}
function switchScenario(id) {
  if (S.scenario === id) return;
  stopPlayback();
  S.scenario = id;
  setHash();
  renderScenarioChips();
  bootScenario();
}
function switchView(v) {
  if (VIEW_IDS.indexOf(v) < 0) v = "overview";
  S.view = v;
  setHash();
  $$("#rail a").forEach(function (a) {
    if (a.getAttribute("data-view") === v) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  renderView();
}
function setHash() {
  var h = "#view=" + S.view + "&scenario=" + S.scenario;
  if (location.hash !== h) { S._ignoreHash = true; location.hash = h; }
}
function parseHash() {
  var m = {}; location.hash.replace(/^#/, "").split("&").forEach(function (kv) {
    var p = kv.split("="); if (p[0]) m[p[0]] = decodeURIComponent(p[1] || "");
  });
  return m;
}
window.addEventListener("hashchange", function () {
  if (S._ignoreHash) { S._ignoreHash = false; return; }
  var h = parseHash();
  var changedScenario = h.scenario && SCENARIO_IDS.indexOf(h.scenario) >= 0 && h.scenario !== S.scenario;
  if (h.view && VIEW_IDS.indexOf(h.view) >= 0) S.view = h.view;
  if (changedScenario) { S.scenario = h.scenario; renderScenarioChips(); bootScenario(); }
  else switchView(S.view);
});

/* ------------------------------------------------------------ sim playback */
function simBounds() { var d = D(); return { min: 24, max: d.series.t.length - 1 }; }
function setSimNow(i, fromScrub) {
  var b = simBounds();
  i = clamp(Math.round(i), b.min, b.max);
  S.simNow[S.scenario] = i;
  var d = D();
  $("#simClock").innerHTML = "<span class='d'>" + tShort(d.series.t[i]) + "</span><span class='hcount'> · h" + (i + 1) + "/" + (b.max + 1) + "</span>";
  if (!fromScrub) $("#scrubber").value = i;
  updateBell();
  var view = window.Views && Views[S.view];
  if (view && view.update) view.update($("#view-" + S.view), d, i);
  if ($("#drawer").classList.contains("open")) renderDrawer();
}
function startPlayback() {
  var b = simBounds();
  if (NOW() >= b.max) setSimNow(b.min);
  S.playing = true;
  $("#playIcon").hidden = true; $("#pauseIcon").hidden = false;
  $("#playBtn").setAttribute("aria-label", "Pause simulation");
  S.lastTick = performance.now();
  S._acc = NOW();
  S.raf = setInterval(function () {
    if (!S.playing) return;
    var ts = performance.now();
    var dt = (ts - S.lastTick) / 1000; S.lastTick = ts;
    S._acc += dt * S.speed;
    var b2 = simBounds();
    if (S._acc >= b2.max) { S._acc = b2.max; setSimNow(b2.max); stopPlayback(); return; }
    if (Math.round(S._acc) !== NOW()) setSimNow(S._acc);
  }, 110);
}
function stopPlayback() {
  S.playing = false;
  if (S.raf) clearInterval(S.raf);
  $("#playIcon").hidden = false; $("#pauseIcon").hidden = true;
  $("#playBtn").setAttribute("aria-label", "Play simulation");
}

/* ------------------------------------------------------------ alerts drawer */
function updateBell() {
  var i = NOW();
  var acks = S.acks[S.scenario] || {};
  var n = visibleAlerts(i).filter(function (a) { return a.state === "active" && !acks[a.key]; }).length;
  var cnt = $("#bellCnt");
  cnt.hidden = n === 0; cnt.textContent = n;
}
function renderDrawer() {
  var body = $("#drawerBody"); body.innerHTML = "";
  var i = NOW();
  var d = D();
  var acks = S.acks[S.scenario] || (S.acks[S.scenario] = {});
  var list = visibleAlerts(i);
  var anoms = (d.anomalies || []).filter(function (a) { return a.t <= d.series.t[i]; });

  if (!list.length && !anoms.length) {
    body.appendChild(el("div", { class: "empty-note", text: "No alerts at this point of the simulated window." }));
    return;
  }
  list.forEach(function (a) {
    var card = el("div", { class: "alert-card sev-" + a.severity });
    var row = el("div", { class: "a-row" }, [
      el("span", { class: "a-time", text: tShort(a.t) }),
      el("span", { class: "a-state " + a.state, text: a.state }),
      el("span", { class: "prov", text: a.source, title: "alert source: " + a.source + " (simulated run)" }),
    ]);
    if (a.state === "active") {
      if (acks[a.key]) {
        row.appendChild(el("span", { class: "prov warn a-ack", text: "acknowledged · local demo state" }));
      } else {
        row.appendChild(el("button", {
          class: "chip-btn a-ack", text: "Acknowledge",
          onclick: function () { acks[a.key] = true; logLocalAction("Alert acknowledged — " + a.text); renderDrawer(); updateBell(); },
        }));
      }
    }
    card.appendChild(row);
    card.appendChild(el("div", { class: "a-text", text: a.text }));
    body.appendChild(card);
  });
  if (anoms.length) {
    body.appendChild(el("div", { class: "label", text: "Anomaly episodes", style: "margin:.7rem 0 .4rem" }));
    anoms.forEach(function (a) {
      var card = el("div", { class: "alert-card sev-" + (a.severity === "critical" ? "critical" : "warning") });
      card.appendChild(el("div", { class: "a-row" }, [
        el("span", { class: "a-time", text: tShort(a.t) }),
        el("span", { class: "prov", text: "score " + fmt(a.score, 2), title: D().anomalies_provenance }),
      ]));
      card.appendChild(el("div", { class: "a-text", text: a.param + " — " + a.note }));
      body.appendChild(card);
    });
  }
}
function openDrawer() {
  renderDrawer();
  $("#drawer").classList.add("open"); $("#drawer").setAttribute("aria-hidden", "false");
  $("#scrim").classList.add("on");
}
function closeDrawer() {
  $("#drawer").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true");
  if (!$("#slideover").classList.contains("open")) $("#scrim").classList.remove("on");
}

/* local demo actions are appended to the Advisor operator log (browser-only) */
function logLocalAction(text) {
  var d = D();
  if (!d._localLog) d._localLog = [];
  d._localLog.push({ t: d.series.t[NOW()], text: text, source: "local" });
  if (S.view === "advisor") renderView();
}

/* ---------------------------------------------------------------- slideover */
function openSlideover(title, buildFn) {
  var so = $("#slideover");
  $("#soTitle").textContent = title;
  var body = $("#soBody"); body.innerHTML = "";
  buildFn(body);
  so.classList.add("open"); so.setAttribute("aria-hidden", "false");
  $("#scrim").classList.add("on");
}
function closeSlideover() {
  var so = $("#slideover");
  so.classList.remove("open"); so.setAttribute("aria-hidden", "true");
  if (!$("#drawer").classList.contains("open")) $("#scrim").classList.remove("on");
}

/* ------------------------------------------------------------------- theme */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("tgf-console-theme", t); } catch (e) {}
  var meta = $('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", t === "dark" ? "#0A0F16" : "#F2F5F9");
  renderView();
}

/* ------------------------------------------------------------------ render */
function renderView() {
  hidePop(); closeSlideover();
  disposeCharts();
  var root = $("#viewRoot");
  root.innerHTML = "";
  var d = D();
  if (!d) return;
  var section = el("section", { class: "view active", id: "view-" + S.view, "aria-label": S.view });
  root.appendChild(section);
  if (window.Views && Views[S.view]) Views[S.view].render(section, d, NOW());
  var live = $("#liveRegion");
  if (live) live.textContent = "Showing " + S.view + " — " + d.meta.title;
}

function bootScenario() {
  var id = S.scenario;
  var root = $("#viewRoot");
  root.innerHTML = '<div class="skel" style="height:220px;margin-bottom:1rem"></div><div class="skel" style="height:340px"></div>';
  loadScenario(id).then(function (d) {
    crossCheck(d);
    if (S.simNow[id] === undefined) S.simNow[id] = d.series.t.length - 1;
    var b = simBounds();
    var sc = $("#scrubber");
    sc.min = b.min; sc.max = b.max; sc.value = S.simNow[id];
    setSimNow(S.simNow[id]);
    switchView(S.view);
  }).catch(function () {
    root.innerHTML = "";
    root.appendChild(el("div", { class: "card", html:
      "<h3>Couldn't load this scenario</h3><p style='color:var(--muted)'>" +
      "Scenario files are fetched from <code>./data/</code> — open via the hosted page to load other scenarios.</p>" }));
  });
}

/* -------------------------------------------------------------------- boot */
(function boot() {
  var saved = null;
  try { saved = localStorage.getItem("tgf-console-theme"); } catch (e) {}
  if (saved === "light" || (saved === null && window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches)) {
    document.documentElement.setAttribute("data-theme", saved === "light" ? "light" : "dark");
  }
  document.body.appendChild(el("div", { id: "liveRegion", class: "visually-hidden", "aria-live": "polite" }));

  var h = parseHash();
  if (h.scenario && SCENARIO_IDS.indexOf(h.scenario) >= 0) S.scenario = h.scenario;
  if (h.view && VIEW_IDS.indexOf(h.view) >= 0) S.view = h.view;

  renderScenarioChips();
  loadProvenance();

  $("#playBtn").addEventListener("click", function () { S.playing ? stopPlayback() : startPlayback(); });
  $("#speedSel").addEventListener("change", function (e) { S.speed = Number(e.target.value); });
  $("#scrubber").addEventListener("input", function (e) { stopPlayback(); setSimNow(Number(e.target.value), true); });
  $("#bellBtn").addEventListener("click", function () {
    $("#drawer").classList.contains("open") ? closeDrawer() : openDrawer();
  });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#soClose").addEventListener("click", closeSlideover);
  $("#scrim").addEventListener("click", function () { closeDrawer(); closeSlideover(); });
  $("#themeBtn").addEventListener("click", function () {
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
  $("#simPill").addEventListener("click", function () { switchView("methods"); });
  $$("#rail a").forEach(function (a) {
    a.addEventListener("click", function (e) { e.preventDefault(); switchView(a.getAttribute("data-view")); });
  });

  // Views are registered by views.js (loaded after this file); boot happens there.
  window.__appReady = true;
})();
