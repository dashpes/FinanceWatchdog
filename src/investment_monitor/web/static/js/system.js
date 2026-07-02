/* The engine room: services, timers, run health, storage. */

(function () {
  "use strict";
  const A = window.Archie;

  function gb(bytes) {
    if (bytes == null) return "—";
    if (bytes > 1e9) return (bytes / 1e9).toFixed(1) + " GB";
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + " MB";
    return (bytes / 1e3).toFixed(0) + " KB";
  }

  A.getJSON("/api/system").then(function (data) {
    const problems = (data.problem_runs || []).length;
    const ctl = data.control || {};
    const parts = ["<p class='salutation'>A look below decks.</p>"];
    parts.push("<p>" + (problems
      ? "<b class='loss'>" + problems + (problems === 1 ? " recent run" : " recent runs") + " ended badly</b> — particulars in the table."
      : "The recent runs all ended cleanly.") + "</p>");
    if (ctl.trading_paused) {
      parts.push("<p>Trading is <b>paused</b>" + (ctl.reason ? " — “" + A.esc(ctl.reason) + "”" : "") +
        " (set by " + A.esc(ctl.updated_by || "operator") + ").</p>");
    }
    A.fill("letter", parts.join(""));

    const blocked = Object.keys(data.blocklist_learned || {}).length;
    A.fill("cards", [
      { k: "Ledger size", v: gb(data.db_size_bytes), sub: "data/portfolio.db" },
      { k: "Disk free", v: gb(data.disk_free_bytes) },
      { k: "Paper forced by .env", v: data.env_force_dry_run ? "yes" : "no", sub: "ROBO_FORCE_DRY_RUN" },
      { k: "Learned blocklist", v: String(blocked), sub: "names the broker refused to buy" },
    ].map(function (x) {
      return "<div class='card'><div class='k'>" + x.k + "</div><div class='v' style='font-size:18px'>" + x.v + "</div>" +
        (x.sub ? "<div class='sub'>" + x.sub + "</div>" : "") + "</div>";
    }).join(""));

    if (data.services) {
      A.el("services-section").style.display = "";
      A.rows("services", data.services, function (s) {
        const cls = s.active === "active" ? "ok" : s.active === "inactive" ? "warn" : "bad";
        return "<tr><td>" + A.esc(s.unit) + "</td>" +
          "<td><span class='chip " + cls + "'>" + A.esc(s.active) + "</span></td>" +
          "<td class='small'>" + A.esc(s.next_elapse || "—") + "</td></tr>";
      }, "No units found.", 3);
    }

    A.rows("runs", data.recent_runs, function (r) {
      const cls = r.status === "completed" ? "ok" : (r.status === "paused" ? "warn" : "bad");
      return "<tr><td class='small'>" + A.when(r.started_at) + "</td>" +
        "<td><span class='chip " + cls + "'>" + A.esc(r.status) + "</span></td>" +
        "<td>" + (r.dry_run ? "paper" : "live") + "</td>" +
        "<td>" + A.esc(r.source || "") + "</td>" +
        "<td class='num'>" + r.num_placed + "</td>" +
        "<td class='why small'>" + A.esc(r.notes || "") + "</td></tr>";
    }, "No runs recorded yet.", 6);
  }).catch(function (e) {
    A.fill("letter", "<p class='loss'>" + A.esc(e.message) + "</p>");
  });
})();
