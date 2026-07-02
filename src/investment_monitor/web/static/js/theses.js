/* Theses list: the book of ideas. */

(function () {
  "use strict";
  const A = window.Archie;

  function chip(status) {
    const cls = status === "active" ? "ok" : (status === "invalidated" || status === "exited") ? "bad" : "warn";
    return "<span class='chip " + cls + "'>" + A.esc(status) + "</span>";
  }

  function load(status) {
    A.getJSON("/api/theses" + (status ? "?status=" + encodeURIComponent(status) : "")).then(function (data) {
      const list = data.theses || [];
      const active = list.filter(function (t) { return t.status === "active"; }).length;
      A.fill("letter",
        "<p class='salutation'>The book holds <b>" + list.length + (list.length === 1 ? " idea" : " ideas") + "</b>" +
        (status ? " with status “" + A.esc(status) + "”" : ", of which <b>" + active + "</b> are active") +
        ".</p><p>Open any idea to read the full argument, its conviction history, and every order it produced.</p>");
      A.rows("theses", list, function (t) {
        return "<tr><td class='sym'><a href='/theses/" + encodeURIComponent(t.symbol) + "'>" + A.esc(t.symbol) + "</a></td>" +
          "<td>" + chip(t.status) + "</td>" +
          "<td class='num'>" + (t.conviction != null ? t.conviction.toFixed(2) : "—") + "</td>" +
          "<td class='num'>" + A.pct(t.target_weight) + "</td>" +
          "<td class='why'>" + A.esc(t.excerpt || "") + "</td>" +
          "<td class='num advanced-only small'>" + A.when(t.last_evaluated_at) + "</td></tr>";
      }, "The book is empty — the research loop has not promoted an idea yet.", 6);
    }).catch(function (e) {
      A.fill("letter", "<p class='loss'>" + A.esc(e.message) + "</p>");
    });
  }

  document.getElementById("status-filter").addEventListener("change", function (e) {
    load(e.target.value);
  });
  load("");
})();
