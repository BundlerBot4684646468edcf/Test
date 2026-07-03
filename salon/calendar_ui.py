"""Simple admin calendar for a salon: a week view of all Cal.com bookings
across the salon's event types. Read-only — booking/cancelling still goes
through Famulor or Cal.com itself."""

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from . import models
from .cal_client import CalComClient


def salon_event_type_ids(db: Session, salon: models.Salon) -> list[str]:
    """All Cal.com event types belonging to this salon: the Round-Robin
    'any employee' type per service plus every (employee, service) type."""
    ids = [s.cal_event_type_id for s in salon.services if s.cal_event_type_id]
    links = (
        db.query(models.EmployeeService)
        .join(models.Service)
        .filter(models.Service.salon_id == salon.id)
        .all()
    )
    ids += [l.cal_event_type_id for l in links if l.cal_event_type_id]
    return ids


def _to_local_iso(value: str | None, tz: ZoneInfo) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(tz).isoformat()


def normalize_bookings(raw_bookings: list[dict], timezone: str) -> list[dict]:
    """Reduce Cal.com booking objects to what the calendar needs, with
    start/end converted into the salon's timezone."""
    tz = ZoneInfo(timezone or "Europe/Berlin")
    out = []
    for b in raw_bookings:
        attendees = b.get("attendees") or []
        hosts = b.get("hosts") or []
        out.append({
            "id": b.get("id") or b.get("uid"),
            "title": b.get("title") or "",
            "start": _to_local_iso(b.get("start"), tz),
            "end": _to_local_iso(b.get("end"), tz),
            "status": b.get("status") or "",
            "customer": (attendees[0].get("name") if attendees else "") or "",
            "employee": (hosts[0].get("name") if hosts else "") or "",
        })
    return out


def list_salon_bookings(
    db: Session, cal: CalComClient, salon: models.Salon, date_from: str, date_to: str
) -> list[dict]:
    event_type_ids = salon_event_type_ids(db, salon)
    if not event_type_ids:
        return []
    raw = cal.get_bookings(event_type_ids, f"{date_from}T00:00:00Z", f"{date_to}T23:59:59Z")
    return normalize_bookings(raw, salon.timezone)


def render_calendar_page(salon: models.Salon) -> str:
    """Self-contained HTML week calendar; fetches bookings from our own
    /salons/{slug}/bookings endpoint client-side."""
    return CALENDAR_HTML.replace("__SALON_NAME__", salon.name).replace("__SALON_SLUG__", salon.slug)


CALENDAR_HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SALON_NAME__ — Kalender</title>
<style>
  :root {
    --line: #e4e2dd; --bg: #faf9f7; --ink: #2b2a27; --muted: #8a867e;
    --accent: #4a6b5d; --accent-soft: #e7efeb; --cancel: #b3552f;
  }
  * { box-sizing: border-box; margin: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--ink); }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 16px 24px; border-bottom: 1px solid var(--line); background: #fff;
    position: sticky; top: 0; z-index: 3;
  }
  header h1 { font-size: 18px; font-weight: 600; margin-right: auto; }
  header h1 small { color: var(--muted); font-weight: 400; margin-left: 8px; }
  .nav { display: flex; align-items: center; gap: 8px; }
  .nav button {
    border: 1px solid var(--line); background: #fff; color: var(--ink);
    border-radius: 8px; padding: 6px 12px; font-size: 14px; cursor: pointer;
  }
  .nav button:hover { border-color: var(--accent); color: var(--accent); }
  #range { font-size: 14px; color: var(--muted); min-width: 180px; text-align: center; }
  #status { font-size: 13px; color: var(--muted); padding: 6px 24px; }
  #status.error { color: var(--cancel); }
  .grid-wrap { overflow-x: auto; }
  .grid {
    display: grid; grid-template-columns: 56px repeat(7, minmax(120px, 1fr));
    min-width: 920px; background: #fff; border-top: 1px solid var(--line);
  }
  .day-head {
    padding: 8px 6px; text-align: center; font-size: 13px; border-left: 1px solid var(--line);
    border-bottom: 1px solid var(--line); position: sticky; top: 0; background: #fff;
  }
  .day-head .dow { color: var(--muted); }
  .day-head .dom { font-weight: 600; font-size: 15px; }
  .day-head.today .dom {
    background: var(--accent); color: #fff; border-radius: 999px;
    display: inline-block; min-width: 26px; padding: 2px 4px;
  }
  .time-col { border-bottom: 1px solid var(--line); }
  .hour-label {
    height: 48px; font-size: 11px; color: var(--muted);
    text-align: right; padding: 2px 6px 0 0;
  }
  .day-col { position: relative; border-left: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .hour-line { height: 48px; border-top: 1px solid var(--line); }
  .hour-line:first-child { border-top: none; }
  .booking {
    position: absolute; left: 3px; right: 3px; overflow: hidden;
    background: var(--accent-soft); border-left: 3px solid var(--accent);
    border-radius: 6px; padding: 3px 6px; font-size: 12px; line-height: 1.3;
    cursor: default;
  }
  .booking .who { font-weight: 600; }
  .booking .what, .booking .time { color: var(--muted); white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }
  .booking.cancelled { border-left-color: var(--cancel); background: #f6e9e3; text-decoration: line-through; }
  .empty-note { padding: 24px; color: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>__SALON_NAME__ <small>Terminkalender</small></h1>
  <div class="nav">
    <button id="prev" aria-label="Vorherige Woche">&#8592;</button>
    <button id="todayBtn">Heute</button>
    <button id="next" aria-label="Nächste Woche">&#8594;</button>
    <span id="range"></span>
  </div>
</header>
<div id="status"></div>
<div class="grid-wrap"><div class="grid" id="grid"></div></div>
<script>
(function () {
  const SLUG = "__SALON_SLUG__";
  const DAY_START = 8, DAY_END = 20, HOUR_PX = 48;
  const DOW = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"];
  const MONTHS = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"];

  function mondayOf(d) {
    const x = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    x.setDate(x.getDate() - ((x.getDay() + 6) % 7));
    return x;
  }
  function iso(d) {
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") +
      "-" + String(d.getDate()).padStart(2, "0");
  }

  const params = new URLSearchParams(location.search);
  let weekStart = mondayOf(params.get("week") ? new Date(params.get("week")) : new Date());

  const grid = document.getElementById("grid");
  const statusEl = document.getElementById("status");

  function buildGrid(days) {
    grid.innerHTML = "";
    const corner = document.createElement("div");
    corner.className = "day-head";
    grid.appendChild(corner);
    const todayIso = iso(new Date());
    for (const d of days) {
      const h = document.createElement("div");
      h.className = "day-head" + (iso(d) === todayIso ? " today" : "");
      h.innerHTML = '<div class="dow">' + DOW[(d.getDay() + 6) % 7] + '</div>' +
        '<div class="dom">' + d.getDate() + '</div>';
      grid.appendChild(h);
    }
    const timeCol = document.createElement("div");
    timeCol.className = "time-col";
    for (let h = DAY_START; h < DAY_END; h++) {
      const l = document.createElement("div");
      l.className = "hour-label";
      l.textContent = String(h).padStart(2, "0") + ":00";
      timeCol.appendChild(l);
    }
    grid.appendChild(timeCol);
    const cols = {};
    for (const d of days) {
      const col = document.createElement("div");
      col.className = "day-col";
      for (let h = DAY_START; h < DAY_END; h++) {
        col.appendChild(Object.assign(document.createElement("div"), { className: "hour-line" }));
      }
      grid.appendChild(col);
      cols[iso(d)] = col;
    }
    return cols;
  }

  function place(cols, b) {
    if (!b.start) return;
    const start = new Date(b.start);
    const end = b.end ? new Date(b.end) : new Date(start.getTime() + 30 * 60000);
    const col = cols[iso(start)];
    if (!col) return;
    const startH = start.getHours() + start.getMinutes() / 60;
    const endH = Math.max(startH + 0.4, end.getHours() + end.getMinutes() / 60);
    const el = document.createElement("div");
    el.className = "booking" + (String(b.status).toLowerCase().includes("cancel") ? " cancelled" : "");
    el.style.top = (startH - DAY_START) * HOUR_PX + "px";
    el.style.height = Math.max(20, (endH - startH) * HOUR_PX - 2) + "px";
    const hm = (d) => String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
    el.innerHTML =
      '<div class="who"></div><div class="what"></div><div class="time">' + hm(start) + "–" + hm(end) + "</div>";
    el.querySelector(".who").textContent = b.customer || "(ohne Name)";
    el.querySelector(".what").textContent = b.title + (b.employee ? " · " + b.employee : "");
    el.title = (b.customer || "") + " — " + b.title + (b.employee ? " bei " + b.employee : "");
    col.appendChild(el);
  }

  async function load() {
    const days = [];
    for (let i = 0; i < 7; i++) {
      const d = new Date(weekStart);
      d.setDate(d.getDate() + i);
      days.push(d);
    }
    const a = days[0], b = days[6];
    document.getElementById("range").textContent =
      a.getDate() + ". " + MONTHS[a.getMonth()] + " – " + b.getDate() + ". " + MONTHS[b.getMonth()] + " " + b.getFullYear();
    const cols = buildGrid(days);
    statusEl.textContent = "Lade Termine…";
    statusEl.classList.remove("error");
    try {
      const resp = await fetch("/salons/" + SLUG + "/bookings?date_from=" + iso(a) + "&date_to=" + iso(b));
      if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
      const bookings = await resp.json();
      bookings.forEach((bk) => place(cols, bk));
      statusEl.textContent = bookings.length
        ? bookings.length + " Termin(e) in dieser Woche"
        : "Keine Termine in dieser Woche";
    } catch (e) {
      statusEl.textContent = "Termine konnten nicht geladen werden: " + e.message;
      statusEl.classList.add("error");
    }
  }

  function shift(daysDelta) {
    weekStart.setDate(weekStart.getDate() + daysDelta);
    history.replaceState(null, "", "?week=" + iso(weekStart));
    load();
  }
  document.getElementById("prev").addEventListener("click", () => shift(-7));
  document.getElementById("next").addEventListener("click", () => shift(7));
  document.getElementById("todayBtn").addEventListener("click", () => {
    weekStart = mondayOf(new Date());
    history.replaceState(null, "", "?week=" + iso(weekStart));
    load();
  });
  load();
})();
</script>
</body>
</html>
"""
