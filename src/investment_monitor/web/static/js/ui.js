/* Shared dashboard helpers: fetch, formatting, theme, advanced mode, PIN. */

window.Archie = (function () {
  "use strict";

  // ---- fetch ----------------------------------------------------------------

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    return res.json();
  }

  function token() { return sessionStorage.getItem("archie-token") || ""; }

  async function send(url, method, body) {
    let tok = token();
    if (!tok) {
      tok = prompt("Dashboard PIN (DASHBOARD_TOKEN from .env):") || "";
      if (!tok) throw new Error("cancelled");
      sessionStorage.setItem("archie-token", tok);
    }
    const res = await fetch(url, {
      method: method,
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + tok,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 401) {
      sessionStorage.removeItem("archie-token");
      throw new Error("wrong PIN — try again");
    }
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* keep */ }
      throw new Error(detail);
    }
    return res.json();
  }

  // ---- formatting ------------------------------------------------------------

  const moneyFmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

  function money(v) { return v === null || v === undefined ? "—" : moneyFmt.format(v); }

  function signedMoney(v) {
    if (v === null || v === undefined) return "—";
    const s = moneyFmt.format(Math.abs(v));
    return (v < -0.004 ? "−" : "+") + s;
  }

  function pnlClass(v) { return v === null || v === undefined ? "muted" : v < 0 ? "loss" : "gain"; }

  function pct(v, digits) {
    if (v === null || v === undefined) return "—";
    return (v * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function signedPct(v, digits) {
    if (v === null || v === undefined) return "—";
    const s = (Math.abs(v) * 100).toFixed(digits === undefined ? 1 : digits) + "%";
    return (v < 0 ? "−" : "+") + s;
  }

  function when(iso) {
    if (!iso) return "—";
    const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
    return d.toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  }

  function day(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // ---- DOM -------------------------------------------------------------------

  function el(id) { return document.getElementById(id); }

  function fill(id, html) { const n = el(id); if (n) n.innerHTML = html; }

  function rows(tbodyId, list, renderRow, emptyMsg, cols) {
    const body = el(tbodyId);
    if (!body) return;
    if (!list || !list.length) {
      body.innerHTML = '<tr><td colspan="' + (cols || 9) + '" class="muted"><i>' +
        esc(emptyMsg || "Nothing to report.") + "</i></td></tr>";
      return;
    }
    body.innerHTML = list.map(renderRow).join("");
  }

  function flash(id, msg, ok) {
    const n = el(id);
    if (!n) return;
    n.textContent = msg;
    n.className = "flash " + (ok ? "ok" : "err");
    setTimeout(function () { n.textContent = ""; }, 6000);
  }

  // ---- chrome: theme, advanced, mode chips -------------------------------------

  function initChrome() {
    const themeBtn = el("theme-toggle");
    if (themeBtn) {
      const sync = function () {
        const cur = document.documentElement.dataset.theme;
        themeBtn.textContent = cur === "study" ? "Paper" : "The Study";
      };
      sync();
      themeBtn.addEventListener("click", function () {
        const next = document.documentElement.dataset.theme === "study" ? "paper" : "study";
        document.documentElement.dataset.theme = next;
        localStorage.setItem("archie-theme", next);
        sync();
        document.dispatchEvent(new CustomEvent("archie:theme", { detail: next }));
      });
    }

    const advBtn = el("advanced-toggle");
    if (advBtn) {
      const apply = function (on) {
        document.body.dataset.advanced = on ? "on" : "off";
        advBtn.setAttribute("aria-pressed", on ? "true" : "false");
        document.querySelectorAll("details.appendix").forEach(function (d) { d.open = on; });
      };
      apply(localStorage.getItem("archie-advanced") === "on");
      advBtn.addEventListener("click", function () {
        const on = document.body.dataset.advanced !== "on";
        localStorage.setItem("archie-advanced", on ? "on" : "off");
        apply(on);
      });
    }

    // Trading-state chips in the nav, on every page.
    getJSON("/api/control").then(function (c) {
      const chips = [];
      if (c.trading_paused) chips.push('<span class="chip bad" title="' + esc(c.reason) + '">Paused</span>');
      const paper = c.env_force_dry_run || c.force_dry_run;
      chips.push(paper
        ? '<span class="chip warn" title="No real money moves.">Paper</span>'
        : '<span class="chip ok" title="Live trading enabled.">Live</span>');
      fill("mode-chips", chips.join(" "));
    }).catch(function () { /* chip-less is fine */ });
  }

  document.addEventListener("DOMContentLoaded", initChrome);

  return {
    getJSON: getJSON, send: send,
    money: money, signedMoney: signedMoney, pnlClass: pnlClass,
    pct: pct, signedPct: signedPct, when: when, day: day, esc: esc,
    el: el, fill: fill, rows: rows, flash: flash,
  };
})();
