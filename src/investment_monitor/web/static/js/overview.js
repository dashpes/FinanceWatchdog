/* Overview page: the letter, summary cards, equity curve, dealings, holdings. */

(function () {
  "use strict";
  const A = window.Archie;

  function letter(data) {
    const acct = data.account;
    const bot = data.bot || {};
    if (!acct) {
      return "<p class='salutation'>Good day.</p><p>The books are empty — no run has been " +
        "recorded yet and the broker has not answered. Once the first run completes, " +
        "your ledger opens here.</p>";
    }
    const parts = [];
    parts.push("<p class='salutation'>Good day.</p>");
    let s = "Your portfolio stands at <b>" + A.money(acct.total_value) + "</b>";
    if (acct.unrealized_gain != null) {
      s += ", carrying <b class='" + A.pnlClass(acct.unrealized_gain) + "'>" +
        A.signedMoney(acct.unrealized_gain) + "</b> unrealised";
    }
    s += ".";
    parts.push("<p>" + s + "</p>");
    const n = (data.todays_orders || []).length;
    parts.push("<p>" + (n
      ? "I have dealt <b>" + n + (n === 1 ? " order" : " orders") + "</b> today; the particulars are entered below."
      : "I have not dealt today; the book rests as it was.") + "</p>");
    const paper = bot.env_force_dry_run || bot.control_force_dry_run;
    if (bot.trading_paused) {
      parts.push("<p>Trading is <b>paused</b> at your instruction.</p>");
    } else if (paper) {
      parts.push("<p>We remain in <b>paper</b> — no real money moves.</p>");
    } else {
      parts.push("<p>We are dealing <b>live</b>.</p>");
    }
    if (acct.stale) {
      parts.push("<p class='muted small'>The broker has not answered just now; figures are as of " +
        A.when(acct.as_of) + ".</p>");
    }
    return parts.join("");
  }

  function cards(data) {
    const acct = data.account || {};
    const last = (data.bot || {}).last_run;
    const c = [];
    c.push({ k: "Portfolio value", v: A.money(acct.total_value) });
    c.push({ k: "Cash available", v: A.money(acct.settled_cash) });
    c.push({
      k: "Unrealised P&L",
      v: "<span class='" + A.pnlClass(acct.unrealized_gain) + "'>" + A.signedMoney(acct.unrealized_gain) + "</span>",
    });
    c.push({
      k: "Realised P&L",
      v: "<span class='" + A.pnlClass(data.realized_total) + "'>" + A.signedMoney(data.realized_total) + "</span>",
      sub: "from the robo's own closed trades",
    });
    if (last) {
      c.push({
        k: "Last run", v: "<span style='font-size:15px'>" + A.esc(last.status) + "</span>",
        sub: A.when(last.started_at) + " · " + (last.dry_run ? "paper" : "live") + " · " + A.esc(last.source || ""),
      });
    }
    A.fill("cards", c.map(function (x) {
      return "<div class='card'><div class='k'>" + x.k + "</div><div class='v'>" + x.v + "</div>" +
        (x.sub ? "<div class='sub'>" + x.sub + "</div>" : "") + "</div>";
    }).join(""));
  }

  function dealings(list) {
    A.rows("dealings", list, function (o) {
      const size = o.notional != null ? A.money(o.notional) : (o.quantity != null ? o.quantity + " sh" : "—");
      return "<tr><td class='" + (o.side === "buy" ? "gain" : "loss") + "'>" + A.esc(o.side) + "</td>" +
        "<td class='sym'>" + A.esc(o.symbol) + "</td>" +
        "<td class='num'>" + size + "</td>" +
        "<td class='num'>" + (o.fill_price != null ? A.money(o.fill_price) : "<span class='muted'>pending</span>") + "</td>" +
        "<td class='why'>" + A.esc(o.rationale || "") + "</td></tr>";
    }, "No dealings today.", 5);
  }

  function holdings(acct) {
    const list = (acct && acct.positions) ? acct.positions.slice().sort(function (a, b) {
      return (b.market_value || 0) - (a.market_value || 0);
    }) : null;
    A.rows("holdings", list, function (p) {
      return "<tr><td class='sym'>" + A.esc(p.symbol) + "</td>" +
        "<td class='num'>" + (p.quantity != null ? (+p.quantity).toFixed(4).replace(/\.?0+$/, "") : "—") + "</td>" +
        "<td class='num'>" + A.money(p.price) + "</td>" +
        "<td class='num'>" + A.money(p.market_value) + "</td>" +
        "<td class='num'>" + A.pct(p.weight) + "</td>" +
        "<td class='num " + A.pnlClass(p.unrealized_gain) + "'>" + A.signedMoney(p.unrealized_gain) + "</td>" +
        "<td class='num advanced-only'>" + A.signedPct(p.unrealized_return) + "</td></tr>";
    }, acct && acct.source === "last_run"
      ? "The broker is unreachable — holdings will appear when it answers."
      : "No holdings yet.", 7);
  }

  function exposure(acct) {
    if (!acct || !acct.positions || !acct.positions.length || !acct.total_value) return;
    const ws = acct.positions.map(function (p) { return p.weight || 0; }).sort(function (a, b) { return b - a; });
    const invested = ws.reduce(function (s, w) { return s + w; }, 0);
    const hhi = ws.reduce(function (s, w) { return s + w * w; }, 0);
    const cards = [
      { k: "Invested / cash", v: A.pct(invested) + " / " + A.pct(1 - invested) },
      { k: "Largest position", v: A.pct(ws[0]) },
      { k: "Top-3 concentration", v: A.pct(ws.slice(0, 3).reduce(function (s, w) { return s + w; }, 0)) },
      { k: "HHI", v: hhi.toFixed(3), sub: "sum of squared weights — higher is more concentrated" },
    ];
    A.fill("exposure-cards", cards.map(function (x) {
      return "<div class='card'><div class='k'>" + x.k + "</div><div class='v'>" + x.v + "</div>" +
        (x.sub ? "<div class='sub'>" + x.sub + "</div>" : "") + "</div>";
    }).join(""));
  }

  A.getJSON("/api/overview").then(function (data) {
    A.fill("letter", letter(data));
    cards(data);
    dealings(data.todays_orders);
    holdings(data.account);
    exposure(data.account);
  }).catch(function (e) {
    A.fill("letter", "<p class='loss'>The ledger could not be fetched: " + A.esc(e.message) + "</p>");
  });

  A.getJSON("/api/overview/equity").then(function (data) {
    const pts = data.points || [];
    if (!pts.length) return;
    ArchieCharts.line(
      A.el("equity-chart"),
      pts.map(function (p) { return p.date; }),
      pts.map(function (p) { return p.total_value; }),
      "Portfolio value",
      function (v) { return "$" + Number(v).toLocaleString(); }
    );
  }).catch(function () { /* chartless is fine */ });
})();
