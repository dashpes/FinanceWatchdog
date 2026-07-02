/* Chart room: candlestick per symbol with trade markers. */

(function () {
  "use strict";
  const A = window.Archie;

  function load() {
    const symbol = A.el("symbol-pick").value;
    const days = A.el("days-pick").value;
    if (!symbol) return;
    A.getJSON("/api/charts/price/" + encodeURIComponent(symbol) + "?days=" + days).then(function (data) {
      if (!data.candles.length) {
        A.fill("chart-note", "No price history stored for " + A.esc(symbol) +
          " — the collector fills this in as it runs.");
        return;
      }
      A.fill("chart-note", data.candles.length + " sessions · " +
        (data.trades.length ? data.trades.length + " of Archie's fills marked" : "no fills in this window"));
      ArchieCharts.candles(A.el("candle-chart"), data);
    }).catch(function (e) { A.fill("chart-note", A.esc(e.message)); });
  }

  A.getJSON("/api/charts/symbols").then(function (data) {
    const pick = A.el("symbol-pick");
    pick.innerHTML = (data.symbols || []).map(function (s) {
      return "<option value='" + A.esc(s) + "'>" + A.esc(s) + "</option>";
    }).join("");
    const wanted = new URLSearchParams(location.search).get("symbol");
    if (wanted && data.symbols.indexOf(wanted.toUpperCase()) >= 0) pick.value = wanted.toUpperCase();
    load();
  });

  A.el("symbol-pick").addEventListener("change", load);
  A.el("days-pick").addEventListener("change", load);
})();
