/*
 * chem.js — cooling-tower water-chemistry indices, computed in the browser.
 *
 * Pure functions, no dependencies. These are the same closed-form saturation
 * indices the TGF physics engine uses; the page recomputes them live so the
 * "Show the math" panel and the what-if sliders stay honest — nothing is
 * pre-baked.
 *
 * Saturation pH (pHs) after Langelier, W.F. (1936), "The analytical control of
 *   anti-corrosion water treatment", J. AWWA 28(10):1500-1521:
 *     pHs = (9.3 + A + B) - (C + D)
 *       A = (log10(TDS) - 1) / 10
 *       B = -13.12 * log10(Tk) + 34.55,  Tk = degC + 273.15
 *       C = log10(Ca hardness as CaCO3) - 0.4
 *       D = log10(total alkalinity as CaCO3)
 *
 * LSI (Langelier Saturation Index, 1936):  LSI = pH - pHs
 *   > 0 scaling tendency, ~0 balanced, < 0 corrosive tendency.
 *
 * RSI (Ryznar Stability Index) after Ryznar, J.W. (1944), "A new index for
 *   determining the amount of calcium carbonate scale formed by water",
 *   J. AWWA 36:472-486:  RSI = 2 * pHs - pH
 *   < 6 scaling, 6-7 balanced, > 7 corrosive (rises with corrosivity).
 *
 * PSI (Puckorius Scaling Index) after Puckorius, P.R. & Brooke, J.M. (1991),
 *   "A water treatment index worth watching", Materials Performance 30(4):
 *   uses an equilibrium pH from alkalinity instead of measured pH:
 *     pHeq = 1.465 * log10(total alkalinity) + 4.54
 *     PSI  = 2 * pHs - pHeq
 *   < 6 scaling, ~6 balanced, > 7 corrosive.
 *
 * Companion library (maintainer's cooling-tower-chem) carries the same formulas
 * and the vendored copy the TGF physics engine imports.
 *
 * Units: temp_c in °C; tds, calcium, alkalinity in ppm as CaCO3.
 */
(function (global) {
  "use strict";

  var log10 = Math.log10 || function (x) { return Math.log(x) / Math.LN10; };

  // Guard: indices are only defined for positive concentrations / TDS.
  function safe(x) { return x > 0 ? x : 1e-6; }

  function saturationPH(temp_c, tds, calcium, alkalinity) {
    var A = (log10(safe(tds)) - 1.0) / 10.0;
    var B = -13.12 * log10(temp_c + 273.15) + 34.55;
    var C = log10(safe(calcium)) - 0.4;
    var D = log10(safe(alkalinity));
    return (9.3 + A + B) - (C + D);
  }

  function equilibriumPH(alkalinity) {
    return 1.465 * log10(safe(alkalinity)) + 4.54;
  }

  function lsi(ph, temp_c, tds, calcium, alkalinity) {
    return ph - saturationPH(temp_c, tds, calcium, alkalinity);
  }

  function rsi(ph, temp_c, tds, calcium, alkalinity) {
    return 2.0 * saturationPH(temp_c, tds, calcium, alkalinity) - ph;
  }

  function psi(ph, temp_c, tds, calcium, alkalinity) {
    return 2.0 * saturationPH(temp_c, tds, calcium, alkalinity) - equilibriumPH(alkalinity);
  }

  // Verdicts drive the gauge zones and the textual read-out.
  function lsiVerdict(v) {
    if (v <= -0.5) return "corrosive tendency";
    if (v >= 0.5) return "scaling risk";
    return "balanced";
  }
  function rsiVerdict(v) {
    if (v < 6.0) return "scaling risk";
    if (v <= 7.0) return "balanced";
    return "corrosive tendency";
  }
  function psiVerdict(v) {
    if (v < 6.0) return "scaling risk";
    if (v <= 7.0) return "balanced";
    return "corrosive tendency";
  }

  // Compute all three from a single inputs object {ph, temp_c, tds, calcium, alkalinity}.
  function all(inp) {
    return {
      lsi: lsi(inp.ph, inp.temp_c, inp.tds, inp.calcium, inp.alkalinity),
      rsi: rsi(inp.ph, inp.temp_c, inp.tds, inp.calcium, inp.alkalinity),
      psi: psi(inp.ph, inp.temp_c, inp.tds, inp.calcium, inp.alkalinity)
    };
  }

  var Chem = {
    saturationPH: saturationPH,
    equilibriumPH: equilibriumPH,
    lsi: lsi,
    rsi: rsi,
    psi: psi,
    all: all,
    lsiVerdict: lsiVerdict,
    rsiVerdict: rsiVerdict,
    psiVerdict: psiVerdict,
    // Citations kept as data for the "Show the math" panel.
    citations: {
      lsi: "Langelier (1936)",
      rsi: "Ryznar (1944)",
      psi: "Puckorius & Brooke (1991)"
    }
  };

  if (typeof module !== "undefined" && module.exports) module.exports = Chem;
  global.Chem = Chem;
})(typeof window !== "undefined" ? window : this);
