/* Settings: controls (pause/kill), blocklist, and the tunables rulebook. */

(function () {
  "use strict";
  const A = window.Archie;

  // ---- controls ---------------------------------------------------------------

  function renderControl(c) {
    const paused = c.trading_paused;
    A.fill("pause-state", paused ? "<span class='loss'>Paused</span>" : "<span class='gain'>Trading</span>");
    A.fill("pause-detail", paused && c.reason ? "“" + A.esc(c.reason) + "”" : "");
    const pauseBtn = A.el("pause-btn");
    pauseBtn.textContent = paused ? "Resume trading" : "Pause trading";
    pauseBtn.disabled = false;
    pauseBtn.onclick = function () {
      const action = paused ? "resume" : "pause";
      const body = paused ? undefined : { reason: prompt("A note for the record (optional):") || "" };
      A.send("/api/control/" + action, "POST", body || {}).then(function (nc) {
        renderControl(nc);
        A.flash("control-flash", "Done — takes effect on the next run.", true);
      }).catch(function (e) { A.flash("control-flash", e.message, false); });
    };

    const killed = c.force_dry_run;
    A.fill("kill-state", killed ? "<span class='loss'>Engaged</span>" : "Off");
    const killBtn = A.el("kill-btn");
    killBtn.textContent = killed ? "Release kill switch" : "Force paper mode";
    killBtn.disabled = false;
    killBtn.onclick = function () {
      A.send("/api/control/" + (killed ? "unkill" : "kill"), "POST", {}).then(function (nc) {
        renderControl(nc);
        A.flash("control-flash", killed
          ? "Released. Live trading still requires .env and robo.yaml to agree."
          : "Paper mode forced — no real money can move.", true);
      }).catch(function (e) { A.flash("control-flash", e.message, false); });
    };

    const paper = c.env_force_dry_run || c.force_dry_run;
    A.fill("mode-state", paper ? "Paper" : "Config decides");
    A.fill("mode-detail", c.env_force_dry_run
      ? "ROBO_FORCE_DRY_RUN is set in .env — the hard floor"
      : (c.force_dry_run ? "forced by this dashboard" : "robo.yaml dry_run applies"));
  }

  A.getJSON("/api/control").then(renderControl).catch(function (e) {
    A.flash("control-flash", e.message, false);
  });

  // ---- blocklist ---------------------------------------------------------------

  function renderBlocklist(data) {
    const rows = [];
    Object.keys(data.learned || {}).forEach(function (sym) {
      rows.push({ sym: sym, why: data.learned[sym], origin: "learned" });
    });
    (data.static || []).forEach(function (sym) {
      rows.push({ sym: sym, why: "", origin: "robo.yaml" });
    });
    A.rows("blocklist", rows, function (r) {
      const act = r.origin === "learned"
        ? "<button class='plain' data-sym='" + A.esc(r.sym) + "'>remove</button>"
        : "<span class='muted small'>edit robo.yaml</span>";
      return "<tr><td class='sym'>" + A.esc(r.sym) + "</td><td class='why'>" + A.esc(r.why || "") + "</td>" +
        "<td class='small muted'>" + r.origin + "</td><td>" + act + "</td></tr>";
    }, "Nothing is blocklisted.", 4);
    document.querySelectorAll("#blocklist button[data-sym]").forEach(function (btn) {
      btn.onclick = function () {
        A.send("/api/blocklist/" + encodeURIComponent(btn.dataset.sym), "DELETE").then(function (d) {
          renderBlocklist(d);
          A.flash("block-flash", btn.dataset.sym + " removed.", true);
        }).catch(function (e) { A.flash("block-flash", e.message, false); });
      };
    });
  }

  A.getJSON("/api/blocklist").then(renderBlocklist).catch(function () { /* section degrades */ });

  A.el("block-add").onclick = function () {
    const symbol = A.el("block-symbol").value.trim();
    if (!symbol) return A.flash("block-flash", "A symbol is required.", false);
    A.send("/api/blocklist", "POST", { symbol: symbol, reason: A.el("block-reason").value.trim() })
      .then(function (d) {
        renderBlocklist(d);
        A.el("block-symbol").value = ""; A.el("block-reason").value = "";
        A.flash("block-flash", symbol.toUpperCase() + " will not be bought.", true);
      }).catch(function (e) { A.flash("block-flash", e.message, false); });
  };

  // ---- tunables rulebook ------------------------------------------------------------

  function control(t) {
    const id = "t-" + t.key.replace(/\./g, "-");
    if (t.type === "boolean") {
      return "<select id='" + id + "'><option value='true'" + (t.current === true ? " selected" : "") + ">true</option>" +
        "<option value='false'" + (t.current === false ? " selected" : "") + ">false</option></select>";
    }
    if (t.type === "enum") {
      return "<select id='" + id + "'>" + (t.choices || []).map(function (c) {
        return "<option value='" + A.esc(c) + "'" + (t.current === c ? " selected" : "") + ">" + A.esc(c) + "</option>";
      }).join("") + "</select>";
    }
    if (t.type === "integer" || t.type === "number") {
      const step = t.type === "integer" ? "1" : "any";
      return "<input type='number' id='" + id + "' value='" + A.esc(t.current) + "' step='" + step + "'" +
        (t.minimum != null ? " min='" + t.minimum + "'" : "") +
        (t.maximum != null ? " max='" + t.maximum + "'" : "") + ">";
    }
    return "<input type='text' id='" + id + "' value='" + A.esc(t.current == null ? "" : t.current) + "'>";
  }

  function range(t) {
    if (t.choices && t.choices.length) return "one of: " + t.choices.join(" · ");
    if (t.minimum != null || t.maximum != null) {
      return "range " + (t.minimum != null ? t.minimum : "…") + " – " + (t.maximum != null ? t.maximum : "…") +
        (t.unit ? " " + t.unit : "");
    }
    return t.unit || "";
  }

  A.getJSON("/api/settings").then(function (data) {
    const groups = {};
    (data.settings || []).forEach(function (t) {
      (groups[t.group || "general"] = groups[t.group || "general"] || []).push(t);
    });
    const html = Object.keys(groups).sort().map(function (g) {
      return "<h3 style='font-family:var(--serif);font-style:italic;font-weight:normal;margin:22px 0 4px'>" + A.esc(g) + "</h3>" +
        groups[g].map(function (t) {
          const id = "t-" + t.key.replace(/\./g, "-");
          return "<div class='field'>" +
            "<div><div class='name'>" + A.esc(t.title || t.key) +
            (t.safety ? " <span class='chip warn' title='Affects real-money behaviour'>guarded</span>" : "") +
            "</div><div class='desc'>" + A.esc(t.description || "") + "</div></div>" +
            "<div>" + control(t) + "<div class='range'>" + A.esc(range(t)) + "</div></div>" +
            "<div><button class='act' data-key='" + A.esc(t.key) + "' data-id='" + id + "'" +
            (t.safety ? " data-safety='1'" : "") + ">Set</button></div>" +
            "</div>";
        }).join("");
    }).join("");
    A.fill("tunables", html);

    document.querySelectorAll("#tunables button[data-key]").forEach(function (btn) {
      btn.onclick = function () {
        const key = btn.dataset.key;
        const value = A.el(btn.dataset.id).value;
        const body = { value: value };
        if (btn.dataset.safety) {
          if (!confirm("“" + key + "” affects real-money behaviour. Apply " + key + " = " + value + "?")) return;
          body.confirm = true;
        }
        A.send("/api/settings/" + encodeURIComponent(key), "PUT", body).then(function (r) {
          A.flash("settings-flash", key + " = " + r.value + " — takes effect on the next run.", true);
        }).catch(function (e) { A.flash("settings-flash", key + ": " + e.message, false); });
      };
    });
  }).catch(function (e) {
    A.fill("tunables", "<p class='loss'>" + A.esc(e.message) + "</p>");
  });
})();
