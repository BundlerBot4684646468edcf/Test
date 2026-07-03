"""Admin page for one salon: manage employees, services (durations, buffers,
prices) and the qualification matrix (who does what, with their own
duration). All changes go through the /salons/{slug}/... admin endpoints,
which keep Cal.com in sync."""

from . import models


def render_admin_page(salon: models.Salon) -> str:
    return ADMIN_HTML.replace("__SALON_NAME__", salon.name).replace("__SALON_SLUG__", salon.slug)


ADMIN_HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SALON_NAME__ — Verwaltung</title>
<style>
  :root {
    --line: #e4e2dd; --bg: #faf9f7; --ink: #2b2a27; --muted: #8a867e;
    --accent: #4a6b5d; --accent-soft: #e7efeb; --danger: #b3552f;
  }
  * { box-sizing: border-box; margin: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--ink); }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 16px 24px; border-bottom: 1px solid var(--line); background: #fff;
  }
  header h1 { font-size: 18px; font-weight: 600; margin-right: auto; }
  header h1 small { color: var(--muted); font-weight: 400; margin-left: 8px; }
  header a { color: var(--accent); font-size: 14px; text-decoration: none; }
  main { max-width: 1080px; margin: 0 auto; padding: 24px; display: grid; gap: 24px; }
  section { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 20px; }
  section h2 { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
  section p.hint { font-size: 13px; color: var(--muted); margin-bottom: 14px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 600; }
  td.num, th.num { text-align: right; }
  input[type=text], input[type=email], input[type=number] {
    border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px;
    font-size: 14px; font-family: inherit; background: #fff; color: var(--ink);
  }
  input[type=number] { width: 72px; text-align: right; }
  button {
    border: 1px solid var(--line); background: #fff; color: var(--ink);
    border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.primary:hover { opacity: 0.9; color: #fff; }
  button.danger { color: var(--danger); }
  button.danger:hover { border-color: var(--danger); }
  .addrow { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
  .matrix-cell { white-space: nowrap; }
  .matrix-cell input[type=number] { width: 60px; margin-left: 6px; }
  .badge { font-size: 11px; color: var(--danger); border: 1px solid currentColor; border-radius: 999px; padding: 1px 8px; margin-left: 6px; }
  #status { position: sticky; bottom: 0; padding: 10px 24px; font-size: 13px; color: var(--muted); background: var(--bg); }
  #status.error { color: var(--danger); }
  #status.ok { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>__SALON_NAME__ <small>Verwaltung</small></h1>
  <a href="/salons/__SALON_SLUG__/calendar">Kalender ansehen &rarr;</a>
</header>
<main>
  <section>
    <h2>Mitarbeiter</h2>
    <p class="hint">Neue Mitarbeiter werden automatisch ins Cal.com-Team eingeladen. Beim Entfernen werden ihre Buchungslinks entfernt — bestehende zukünftige Termine blockieren das Entfernen.</p>
    <table id="empTable"><thead><tr><th>Name</th><th>E-Mail</th><th></th></tr></thead><tbody></tbody></table>
    <div class="addrow">
      <input type="text" id="empName" placeholder="Name">
      <input type="email" id="empEmail" placeholder="E-Mail">
      <button class="primary" id="empAdd">+ Mitarbeiter</button>
    </div>
  </section>

  <section>
    <h2>Services</h2>
    <p class="hint">Standard-Dauer gilt für "egal wer"-Buchungen und für alle Mitarbeiter ohne eigene Dauer. Puffer = Aufräumzeit nach dem Termin, fließt in die Konfliktprüfung ein.</p>
    <table id="svcTable"><thead><tr>
      <th>Service</th><th class="num">Dauer (Min)</th><th class="num">Puffer (Min)</th><th class="num">Preis (€)</th><th></th><th></th>
    </tr></thead><tbody></tbody></table>
    <div class="addrow">
      <input type="text" id="svcName" placeholder="Service-Name">
      <input type="number" id="svcDur" placeholder="Min" value="30" min="5">
      <button class="primary" id="svcAdd">+ Service</button>
    </div>
  </section>

  <section>
    <h2>Wer macht was — und wie lange</h2>
    <p class="hint">Haken = Mitarbeiter bietet den Service an. Eigene Dauer nur eintragen, wenn sie vom Standard abweicht (leer = Standard-Dauer des Service). Erst "Zuordnung speichern" schreibt alles nach Cal.com.</p>
    <div style="overflow-x:auto"><table id="matrix"><thead></thead><tbody></tbody></table></div>
    <div class="addrow"><button class="primary" id="matrixSave">Zuordnung speichern</button></div>
  </section>
</main>
<div id="status"></div>
<script>
(function () {
  const SLUG = "__SALON_SLUG__";
  const statusEl = document.getElementById("status");
  let config = null;

  function setStatus(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = cls || "";
  }

  async function api(method, path, body) {
    const resp = await fetch("/salons/" + SLUG + path, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const err = new Error(data.detail || resp.statusText);
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  // 409 = upcoming bookings exist; let the admin decide whether to force
  async function apiWithForceRetry(method, path, body) {
    try {
      return await api(method, path, body);
    } catch (e) {
      if (e.status === 409 && confirm(e.message + "\\n\\nTrotzdem fortfahren?")) {
        if (body) return api(method, path, Object.assign({}, body, { force: true }));
        return api(method, path + (path.includes("?") ? "&" : "?") + "force=true");
      }
      throw e;
    }
  }

  function render() {
    const emp = document.querySelector("#empTable tbody");
    emp.innerHTML = "";
    for (const e of config.employees) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td></td><td></td><td></td>";
      tr.children[0].textContent = e.name;
      tr.children[1].textContent = e.email;
      const del = document.createElement("button");
      del.className = "danger";
      del.textContent = "Entfernen";
      del.onclick = () => run(apiWithForceRetry("DELETE", "/employees/" + e.id), e.name + " entfernt");
      tr.children[2].appendChild(del);
      emp.appendChild(tr);
    }

    const svc = document.querySelector("#svcTable tbody");
    svc.innerHTML = "";
    for (const s of config.services) {
      const tr = document.createElement("tr");
      const name = document.createElement("input"); name.type = "text"; name.value = s.name;
      const dur = document.createElement("input"); dur.type = "number"; dur.value = s.duration_min; dur.min = 5;
      const buf = document.createElement("input"); buf.type = "number"; buf.value = s.buffer_min; buf.min = 0;
      const price = document.createElement("input"); price.type = "number"; price.value = (s.price_cents / 100).toFixed(0); price.min = 0;
      const save = document.createElement("button"); save.textContent = "Speichern";
      save.onclick = () => run(api("PATCH", "/services/" + s.id, {
        name: name.value, duration_min: +dur.value, buffer_min: +buf.value, price_cents: Math.round(+price.value * 100),
      }), "'" + name.value + "' gespeichert (Cal.com aktualisiert)");
      const del = document.createElement("button"); del.className = "danger"; del.textContent = "Löschen";
      del.onclick = () => run(apiWithForceRetry("DELETE", "/services/" + s.id), "'" + s.name + "' gelöscht");
      for (const el of [name, dur, buf, price, save, del]) {
        const td = document.createElement("td");
        if (el === dur || el === buf || el === price) td.className = "num";
        td.appendChild(el);
        tr.appendChild(td);
      }
      if (!s.bookable_any) {
        const b = document.createElement("span");
        b.className = "badge";
        b.textContent = "niemand zugeordnet";
        tr.children[0].appendChild(b);
      }
      svc.appendChild(tr);
    }

    const head = document.querySelector("#matrix thead");
    const body = document.querySelector("#matrix tbody");
    head.innerHTML = ""; body.innerHTML = "";
    const hr = document.createElement("tr");
    hr.innerHTML = "<th>Mitarbeiter</th>";
    for (const s of config.services) {
      const th = document.createElement("th");
      th.textContent = s.name + " (" + s.duration_min + " Min)";
      hr.appendChild(th);
    }
    head.appendChild(hr);
    const qual = {};
    for (const q of config.qualifications) qual[q.employee_id + ":" + q.service_id] = q;
    for (const e of config.employees) {
      const tr = document.createElement("tr");
      const td0 = document.createElement("td"); td0.textContent = e.name; tr.appendChild(td0);
      for (const s of config.services) {
        const q = qual[e.id + ":" + s.id];
        const td = document.createElement("td"); td.className = "matrix-cell";
        const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !!q;
        const dur = document.createElement("input"); dur.type = "number"; dur.min = 5;
        dur.placeholder = s.duration_min;
        if (q && q.duration_min) dur.value = q.duration_min;
        dur.title = "Eigene Dauer in Minuten (leer = Standard)";
        cb.dataset.emp = e.name; cb.dataset.svc = s.name;
        td.appendChild(cb); td.appendChild(dur);
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
  }

  async function run(promise, okMsg) {
    setStatus("Speichere…");
    try {
      config = await promise;
      render();
      setStatus(okMsg, "ok");
    } catch (e) {
      setStatus("Fehler: " + e.message, "error");
    }
  }

  document.getElementById("empAdd").onclick = () => {
    const name = document.getElementById("empName").value.trim();
    const email = document.getElementById("empEmail").value.trim();
    if (!name || !email) return setStatus("Name und E-Mail angeben", "error");
    run(api("POST", "/employees", { name: name, email: email }), name + " eingeladen");
  };
  document.getElementById("svcAdd").onclick = () => {
    const name = document.getElementById("svcName").value.trim();
    const dur = +document.getElementById("svcDur").value;
    if (!name || !dur) return setStatus("Name und Dauer angeben", "error");
    run(api("POST", "/services", { name: name, duration_min: dur }),
      "'" + name + "' angelegt — jetzt unten Mitarbeiter zuordnen");
  };
  document.getElementById("matrixSave").onclick = () => {
    const quals = [];
    document.querySelectorAll("#matrix input[type=checkbox]").forEach((cb) => {
      if (!cb.checked) return;
      const dur = cb.parentElement.querySelector("input[type=number]").value;
      quals.push({
        employee_name: cb.dataset.emp,
        service_name: cb.dataset.svc,
        duration_min: dur ? +dur : null,
      });
    });
    run(apiWithForceRetry("PUT", "/qualifications", { qualifications: quals }),
      "Zuordnung gespeichert (Cal.com synchronisiert)");
  };

  api("GET", "/config").then((c) => { config = c; render(); setStatus("Geladen"); })
    .catch((e) => setStatus("Fehler: " + e.message, "error"));
})();
</script>
</body>
</html>
"""
