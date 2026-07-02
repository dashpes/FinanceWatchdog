/* The Study: calibration aggregates, Brier trend, weight adaptations. */

(function () {
  "use strict";
  const A = window.Archie;

  A.getJSON("/api/learning/summary").then(function (data) {
    const t = data.totals || {};
    const n = t.n_outcomes || 0;
    A.fill("letter", n
      ? "<p class='salutation'>The record holds <b>" + n + (n === 1 ? " outcome" : " outcomes") + "</b>. " +
        "Archie called the direction right <b>" + A.pct(t.win_rate, 0) + "</b> of the time, with a mean " +
        "<abbr title='Squared error of conviction as a forecast; 0 is perfect, 0.25 is guessing'>Brier</abbr> of <b>" +
        (t.mean_brier != null ? t.mean_brier.toFixed(3) : "—") + "</b>.</p>" +
        "<p>These records feed back into sizing: names he keeps getting right earn a larger tilt.</p>"
      : "<p class='salutation'>The record is empty — outcomes appear here as theses are re-evaluated over time.</p>");

    A.fill("cards", [
      { k: "Outcomes recorded", v: String(n) },
      { k: "Directional win rate", v: A.pct(t.win_rate, 0) },
      { k: "Mean Brier", v: t.mean_brier != null ? t.mean_brier.toFixed(3) : "—", sub: "0 perfect · 0.25 coin toss" },
    ].map(function (x) {
      return "<div class='card'><div class='k'>" + x.k + "</div><div class='v'>" + x.v + "</div>" +
        (x.sub ? "<div class='sub'>" + x.sub + "</div>" : "") + "</div>";
    }).join(""));

    A.rows("per-symbol", data.per_symbol, function (s) {
      return "<tr><td class='sym'>" + A.esc(s.symbol) + "</td>" +
        "<td class='num'>" + s.n + "</td>" +
        "<td class='num'>" + A.pct(s.hit_rate, 0) + "</td>" +
        "<td class='num'>" + A.pct(s.ewma_hit_rate, 0) + "</td>" +
        "<td class='num'>" + (s.brier != null ? s.brier.toFixed(3) : "—") + "</td></tr>";
    }, "No outcomes per name yet.", 5);

    const series = (data.outcome_series || []).filter(function (e) { return e.brier != null; });
    if (series.length > 1) {
      ArchieCharts.line(
        A.el("brier-chart"),
        series.map(function (e) { return e.as_of_date; }),
        series.map(function (e) { return e.brier; }),
        "Brier"
      );
    }

    A.rows("adaptations", data.adaptations, function (e) {
      return "<tr><td>" + A.day(e.as_of_date) + "</td><td class='sym'>" + A.esc(e.symbol || "") + "</td>" +
        "<td class='num'>" + (e.before_value != null ? e.before_value.toFixed(3) : "—") + "</td>" +
        "<td class='num'>" + (e.after_value != null ? e.after_value.toFixed(3) : "—") + "</td>" +
        "<td>" + (e.applied ? "<span class='chip ok'>applied</span>" : "<span class='chip'>recorded</span>") + "</td>" +
        "<td class='why'>" + A.esc(e.note || "") + "</td></tr>";
    }, "No adaptations recorded yet.", 6);
  }).catch(function (e) {
    A.fill("letter", "<p class='loss'>" + A.esc(e.message) + "</p>");
  });
})();
