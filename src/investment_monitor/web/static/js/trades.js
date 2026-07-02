/* Ledger: per-name realized P&L, filterable orders, run history. */

(function () {
  "use strict";
  const A = window.Archie;

  A.getJSON("/api/pnl").then(function (data) {
    const total = data.total_realized;
    A.fill("letter",
      "<p class='salutation'>The realised account stands at <b class='" + A.pnlClass(total) + "'>" +
      A.signedMoney(total) + "</b> across the robo's own closed trades.</p>" +
      "<p>Every order below carries the gate's verdict — including the ones it declined, and why.</p>");
    A.rows("pnl", data.per_symbol, function (s) {
      return "<tr><td class='sym'>" + A.esc(s.symbol) + "</td>" +
        "<td class='num " + A.pnlClass(s.realized) + "'>" + A.signedMoney(s.realized) + "</td>" +
        "<td class='num advanced-only'>" + (s.quantity || 0) + " sh @ " + A.money(s.avg_cost) + "</td></tr>";
    }, "Nothing realised yet.", 3);
  }).catch(function (e) { A.fill("letter", "<p class='loss'>" + A.esc(e.message) + "</p>"); });

  function loadOrders() {
    const params = new URLSearchParams();
    const sym = A.el("symbol-filter").value.trim();
    const side = A.el("side-filter").value;
    if (sym) params.set("symbol", sym);
    if (side) params.set("side", side);
    if (A.el("placed-only").checked) params.set("placed_only", "true");
    params.set("limit", "150");
    A.getJSON("/api/orders?" + params).then(function (data) {
      A.rows("orders", data.orders, function (o) {
        const size = o.notional != null ? A.money(o.notional) : (o.quantity != null ? o.quantity + " sh" : "—");
        const gate = o.gate_accepted
          ? (o.preflight_ok === false
            ? "<span class='chip warn' title='" + A.esc(o.preflight_reason || "") + "'>preflight</span>"
            : "<span class='chip ok'>accepted</span>")
          : "<span class='chip bad' title='" + A.esc(o.gate_reason || "") + "'>" + A.esc(o.gate_code || "rejected") + "</span>";
        return "<tr><td class='small'>" + A.when(o.created_at) + "</td>" +
          "<td class='" + (o.side === "buy" ? "gain" : "loss") + "'>" + A.esc(o.side) + "</td>" +
          "<td class='sym'>" + A.esc(o.symbol) + "</td>" +
          "<td class='num'>" + size + "</td>" +
          "<td class='num'>" + (o.fill_price != null ? A.money(o.fill_price) : "—") + "</td>" +
          "<td>" + gate + "</td>" +
          "<td class='why'>" + A.esc(o.rationale || o.reason || "") + "</td></tr>";
      }, "No orders match.", 7);
    }).catch(function () { /* table shows stale content */ });
  }

  ["symbol-filter", "side-filter", "placed-only"].forEach(function (id) {
    A.el(id).addEventListener("change", loadOrders);
  });
  loadOrders();

  A.getJSON("/api/runs?limit=60").then(function (data) {
    A.rows("runs", data.runs, function (r) {
      const cls = r.status === "completed" ? "ok" : (r.status === "paused" ? "warn" : "bad");
      return "<tr><td class='small'>" + A.when(r.started_at) + "</td>" +
        "<td><span class='chip " + cls + "' title='" + A.esc(r.notes || "") + "'>" + A.esc(r.status) + "</span></td>" +
        "<td>" + (r.dry_run ? "paper" : "live") + "</td>" +
        "<td>" + A.esc(r.source || "") + "</td>" +
        "<td class='num'>" + r.num_proposed + "</td><td class='num'>" + r.num_accepted + "</td>" +
        "<td class='num'>" + r.num_rejected + "</td><td class='num'>" + r.num_placed + "</td>" +
        "<td class='num'>" + A.money(r.total_value) + "</td></tr>";
    }, "No runs recorded yet.", 9);
  }).catch(function () { /* appendix degrades */ });
})();
