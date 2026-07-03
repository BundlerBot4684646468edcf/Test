import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from . import models
from .cal_client import CalComClient

WEEKDAYS_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Function-calling schema for Famulor (voice/chat AI). Famulor invokes these
# as tools during a call/chat; the handlers below resolve salon/service names
# to Cal.com IDs and delegate to the Cal.com API.
FAMULOR_TOOLS = [
    {
        "name": "get_current_datetime",
        "description": (
            "Get the current date, time and weekday in the salon's timezone, "
            "plus the weekday of each of the next 7 days. ALWAYS call this "
            "before interpreting any relative date the customer says, such as "
            "'morgen', 'übermorgen', 'Samstag' or 'nächste Woche' — never "
            "guess what today's date or weekday is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "salon_slug": {"type": "string"},
            },
            "required": ["salon_slug"],
        },
    },
    {
        "name": "get_availability",
        "description": "Find available appointment slots for a salon service, optionally for a specific employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "salon_slug": {"type": "string"},
                "service_name": {"type": "string"},
                "employee_name": {"type": "string"},
                "date_from": {"type": "string", "description": "ISO date, e.g. 2026-07-01"},
                "date_to": {"type": "string", "description": "ISO date, e.g. 2026-07-07"},
            },
            "required": ["salon_slug", "service_name", "date_from", "date_to"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment for a customer with a salon.",
        "parameters": {
            "type": "object",
            "properties": {
                "salon_slug": {"type": "string"},
                "service_name": {"type": "string"},
                "employee_name": {"type": "string"},
                "start_at": {"type": "string", "description": "ISO datetime"},
                "customer_name": {"type": "string"},
                "customer_email": {"type": "string"},
                "customer_phone": {
                    "type": "string",
                    "description": (
                        "Customer's mobile number. Required for the SMS booking "
                        "confirmation and reminders — always ask the customer for it."
                    ),
                },
            },
            "required": ["salon_slug", "service_name", "start_at", "customer_name", "customer_email", "customer_phone"],
        },
    },
]


def _find_salon(db: Session, salon_slug: str) -> models.Salon:
    salon = db.query(models.Salon).filter_by(slug=salon_slug).first()
    if not salon:
        raise ValueError(f"Unknown salon: {salon_slug}")
    return salon


def _find_service(db: Session, salon: models.Salon, service_name: str) -> models.Service:
    service = (
        db.query(models.Service)
        .filter(models.Service.salon_id == salon.id, models.Service.name.ilike(service_name))
        .first()
    )
    if not service:
        raise ValueError(f"Unknown service '{service_name}' for salon '{salon.slug}'")
    return service


def _resolve_event_type(
    db: Session, service: models.Service, employee_name: str | None
) -> str:
    """Pick the Round-Robin "any employee" event type, or a specific
    employee's fixed-host event type (which carries that employee's own
    duration) when one is named."""
    if not employee_name:
        if not service.cal_event_type_id:
            raise ValueError(f"No employee currently offers '{service.name}'")
        return service.cal_event_type_id

    link = (
        db.query(models.EmployeeService)
        .join(models.Employee)
        .filter(
            models.EmployeeService.service_id == service.id,
            models.Employee.name.ilike(employee_name),
        )
        .first()
    )
    if not link:
        raise ValueError(f"{employee_name} bietet '{service.name}' nicht an")
    return link.cal_event_type_id


def _salon_now(salon: models.Salon) -> datetime:
    return datetime.now(ZoneInfo(salon.timezone or "Europe/Berlin"))


def _parse_iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD), got: {value!r}")


def _today_context(now: datetime) -> dict:
    today = now.date()
    return {
        "today": today.isoformat(),
        "today_weekday": WEEKDAYS_EN[today.weekday()],
        "today_weekday_de": WEEKDAYS_DE[today.weekday()],
    }


def normalize_phone(phone: str | None) -> str | None:
    """Best-effort E.164 normalization; Cal.com silently skips SMS for
    numbers it can't parse. Target market is Germany, so a bare leading 0
    (as customers dictate their number on the phone) is treated as +49."""
    if not phone:
        return None
    p = re.sub(r"[\s\-().\/]", "", phone)
    if p.startswith("00"):
        p = "+" + p[2:]
    elif p.startswith("0"):
        p = "+49" + p[1:]
    return p


def handle_get_current_datetime(db: Session, salon_slug: str) -> dict:
    salon = _find_salon(db, salon_slug)
    now = _salon_now(salon)
    today = now.date()
    return {
        **_today_context(now),
        "now": now.isoformat(timespec="minutes"),
        "time": now.strftime("%H:%M"),
        "timezone": salon.timezone,
        "next_days": [
            {
                "date": d.isoformat(),
                "weekday": WEEKDAYS_EN[d.weekday()],
                "weekday_de": WEEKDAYS_DE[d.weekday()],
            }
            for d in (today + timedelta(days=i) for i in range(1, 8))
        ],
    }


def handle_get_availability(
    db: Session,
    cal: CalComClient,
    salon_slug: str,
    service_name: str,
    date_from: str,
    date_to: str,
    employee_name: str | None = None,
) -> dict:
    salon = _find_salon(db, salon_slug)
    service = _find_service(db, salon, service_name)
    event_type_id = _resolve_event_type(db, service, employee_name)

    now = _salon_now(salon)
    today = now.date()
    from_d = _parse_iso_date(date_from, "date_from")
    to_d = _parse_iso_date(date_to, "date_to")
    if to_d < from_d:
        from_d, to_d = to_d, from_d
    if to_d < today:
        raise ValueError(
            f"Der angefragte Zeitraum {from_d.isoformat()} bis {to_d.isoformat()} liegt in der "
            f"Vergangenheit. Heute ist {WEEKDAYS_DE[today.weekday()]}, der {today.isoformat()}. "
            "Bitte das Datum neu bestimmen (get_current_datetime nutzen)."
        )
    # Voice bots routinely miscalculate relative dates; a range that merely
    # starts in the past is safe to clamp to today instead of failing the call.
    if from_d < today:
        from_d = today

    result = cal.get_slots(event_type_id, from_d.isoformat(), to_d.isoformat(), salon.timezone)
    result.update(_today_context(now))
    return result


def handle_book_appointment(
    db: Session,
    cal: CalComClient,
    salon_slug: str,
    service_name: str,
    start_at: str,
    customer_name: str,
    customer_email: str,
    employee_name: str | None = None,
    customer_phone: str | None = None,
) -> dict:
    salon = _find_salon(db, salon_slug)
    service = _find_service(db, salon, service_name)
    event_type_id = _resolve_event_type(db, service, employee_name)

    now = _salon_now(salon)
    try:
        start_dt = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"start_at must be an ISO datetime, got: {start_at!r}")
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=ZoneInfo(salon.timezone or "Europe/Berlin"))
    if start_dt < now:
        raise ValueError(
            f"Der Termin {start_at} liegt in der Vergangenheit. Heute ist "
            f"{WEEKDAYS_DE[now.date().weekday()]}, der {now.date().isoformat()}, "
            f"{now.strftime('%H:%M')} Uhr. Bitte das Datum neu bestimmen "
            "(get_current_datetime nutzen)."
        )

    return cal.create_booking(
        event_type_id,
        start_at,
        customer_name,
        customer_email,
        salon.timezone,
        attendee_phone=normalize_phone(customer_phone),
    )
