/* Thesis detail: full narrative, monitoring charts, conditions, orders, learning. */

(function () {
  "use strict";
  const A = window.Archie;
  const symbol = decodeURIComponent(location.pathname.split("/").pop());

  function conditionRows(obj) {
    const keys = Object.keys(obj || {});
    if (!keys.length) return "<dt>None recorded</dt><dd class='muted'><i>No standing instructions.</i></dd>";
    return keys.map(function (k) {
      const v = obj[k];
      return "<dt>" + A.esc(k.replace(/_/g, " ")) + "</dt><dd>" +
        A.esc(typeof v === "object" ? JSON.stringify(v) : String(v)) + "</dd>";
    }).join("");
  }

  A.getJSON("/api/theses/" + encodeURIComponent(symbol)).then(function (t) {
    document.getElementById("title").textContent = t.symbol + " — " + t.status;
    A.fill("narrative", A.esc(t.narrative || "No narrative recorded."));
    A.fill("conditions", conditionRows(t.invalidation_conditions));

    const c = [
      { k: "Conviction", v: t.conviction != null ? t.conviction.toFixed(2) : "—" },
      { k: "Target weight", v: A.pct(t.target_weight) },
      { k: "Opened", v: "<span style='font-size:15px'>" + A.day(t.created_at) + "</span>" },
      { k: "Last evaluated", v: "<span style='font-size:15px'>" + A.when(t.last_evaluated_at) + "</span>" },
    ];
    A.fill("cards", c.map(function (x) {
      return "<div class='card'><div class='k'>" + x.k + "</div><div class='v'>" + x.v + "</div></div>";
    }).join(""));

    A.rows("orders", t.orders, function (o) {
      const size = o.notional != null ? A.money(o.notional) : (o.quantity != null ? o.quantity + " sh" : "—");
      const gate = o.gate_accepted
        ? "<span class='chip ok'>accepted</span>"
        : "<span class='chip bad' title='" + A.esc(o.gate_reason || "") + "'>" + A.esc(o.gate_code || "rejected") + "</span>";
      return "<tr><td class='small'>" + A.when(o.created_at) + "</td>" +
        "<td class='" + (o.side === "buy" ? "gain" : "loss") + "'>" + A.esc(o.side) + "</td>" +
        "<td class='num'>" + size + "</td>" +
        "<td class='num'>" + (o.fill_price != null ? A.money(o.fill_price) : "—") + "</td>" +
        "<td>" + gate + "</td><td class='why'>" + A.esc(o.rationale || "") + "</td></tr>";
    }, "No orders yet under this idea.", 6);

    A.rows("learning", t.learning_events, function (e) {
      return "<tr><td>" + A.day(e.as_of_date) + "</td><td>" + A.esc(e.kind) + "</td>" +
        "<td class='num'>" + (e.conviction != null ? e.conviction.toFixed(2) : "—") + "</td>" +
        "<td class='num " + A.pnlClass(e.realized_return) + "'>" + A.signedPct(e.realized_return, 2) + "</td>" +
        "<td class='num'>" + (e.brier != null ? e.brier.toFixed(3) : "—") + "</td>" +
        "<td class='why'>" + A.esc(e.note || "") + "</td></tr>";
    }, "No learning events recorded yet.", 6);

    const hist = (t.conviction_history || []).map(function (h) {
      return { ts: (h.ts || "").slice(0, 10), conviction: h.conviction };
    }).filter(function (h) { return h.conviction != null; });
    if (hist.length) ArchieCharts.conviction(A.el("conviction-chart"), hist);
  }).catch(function (e) {
    A.fill("letter", "<p class='loss'>" + A.esc(e.message) + "</p>");
  });

  A.getJSON("/api/theses/" + encodeURIComponent(symbol) + "/monitor").then(function (m) {
    const parts = ["<p class='salutation'>On the matter of <b>" + A.esc(m.symbol) + "</b>.</p>"];
    if (m.entry && m.entry.fill_price != null) {
      let s = "I entered at <b>" + A.money(m.entry.fill_price) + "</b> on " + A.day(m.entry.filled_at);
      if (m.latest_close != null) s += "; the last close was <b>" + A.money(m.latest_close) + "</b>";
      if (m.return_since_entry != null) {
        s += ", a move of <b class='" + A.pnlClass(m.return_since_entry) + "'>" +
          A.signedPct(m.return_since_entry) + "</b> since entry";
      }
      parts.push("<p>" + s + ".</p>");
    } else {
      parts.push("<p>No entry order has filled under this idea" +
        (m.latest_close != null ? "; the last close was <b>" + A.money(m.latest_close) + "</b>" : "") + ".</p>");
    }
    A.fill("letter", parts.join(""));

    const prices = m.prices || [];
    if (prices.length) {
      ArchieCharts.line(
        A.el("price-chart"),
        prices.map(function (p) { return p.date; }),
        prices.map(function (p) { return p.close; }),
        m.symbol,
        function (v) { return "$" + v; }
      );
    }
  }).catch(function () { /* monitor section degrades quietly */ });
})();
