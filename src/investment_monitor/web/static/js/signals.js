/* Signals: confluence findings + insider drill-down. */

(function () {
  "use strict";
  const A = window.Archie;

  function loadFindings() {
    const params = new URLSearchParams();
    const kind = A.el("kind-filter").value;
    if (kind) params.set("kind", kind);
    params.set("days", A.el("days-filter").value);
    A.getJSON("/api/signals/confluence?" + params).then(function (data) {
      const list = data.findings || [];
      A.fill("letter",
        "<p class='salutation'>The confluence engine reports <b>" + list.length +
        (list.length === 1 ? " finding" : " findings") + "</b> in the window — places where " +
        "insiders, volume, and other sources agree.</p>");
      A.rows("findings", list, function (f) {
        return "<tr><td class='sym'><a href='/charts?symbol=" + encodeURIComponent(f.ticker) + "'>" + A.esc(f.ticker) + "</a></td>" +
          "<td>" + A.esc((f.kind || "").replace(/_/g, " ")) + "</td>" +
          "<td class='num'>" + (f.score != null ? f.score.toFixed(2) : "—") + "</td>" +
          "<td class='num advanced-only'>" + (f.n_actors || "—") + "</td>" +
          "<td class='num advanced-only'>" + (f.n_sources || "—") + "</td>" +
          "<td class='num'>" + (f.total_value != null ? A.money(f.total_value) : "—") + "</td>" +
          "<td class='num " + A.pnlClass(f.price_change_pct) + "'>" + A.signedPct(f.price_change_pct) + "</td>" +
          "<td class='why'>" + A.esc(f.narrative || "") + "</td></tr>";
      }, "No findings in this window.", 8);
    }).catch(function (e) { A.fill("letter", "<p class='loss'>" + A.esc(e.message) + "</p>"); });
  }

  function loadInsiders() {
    const t = A.el("insider-ticker").value.trim();
    const params = new URLSearchParams({ limit: "50" });
    if (t) params.set("ticker", t);
    A.getJSON("/api/signals/insiders?" + params).then(function (data) {
      A.rows("insiders", data.transactions, function (x) {
        return "<tr><td class='sym'>" + A.esc(x.ticker) + "</td>" +
          "<td class='small'>" + A.day(x.filing_date) + "</td>" +
          "<td>" + A.esc(x.owner_name || "") + "</td>" +
          "<td class='small muted'>" + A.esc(x.owner_title || "") + "</td>" +
          "<td class='num'>" + (x.shares != null ? Number(x.shares).toLocaleString() : "—") + "</td>" +
          "<td class='num'>" + (x.price_per_share != null ? A.money(x.price_per_share) : "—") + "</td>" +
          "<td class='num'>" + (x.total_value != null ? A.money(x.total_value) : "—") + "</td>" +
          "<td class='advanced-only small'>" + (x.sec_url ? "<a href='" + A.esc(x.sec_url) + "' target='_blank' rel='noopener'>filing</a>" : "") + "</td></tr>";
      }, t ? "No filings stored for " + A.esc(t.toUpperCase()) + "." : "Type a ticker to browse filings — 50 most recent shown otherwise.", 8);
    }).catch(function () { /* keep */ });
  }

  A.el("kind-filter").addEventListener("change", loadFindings);
  A.el("days-filter").addEventListener("change", loadFindings);
  A.el("insider-ticker").addEventListener("change", loadInsiders);
  loadFindings();
  loadInsiders();
})();
