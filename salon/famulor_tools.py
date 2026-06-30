from sqlalchemy.orm import Session

from . import models
from .cal_client import CalComClient

# Function-calling schema for Famulor (voice/chat AI). Famulor invokes these
# as tools during a call/chat; the handlers below resolve salon/service names
# to Cal.com IDs and delegate to the Cal.com API.
FAMULOR_TOOLS = [
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
                "customer_phone": {"type": "string"},
            },
            "required": ["salon_slug", "service_name", "start_at", "customer_name", "customer_email"],
        },
    },
]


def _find_service(db: Session, salon_slug: str, service_name: str) -> models.Service:
    salon = db.query(models.Salon).filter_by(slug=salon_slug).first()
    if not salon:
        raise ValueError(f"Unknown salon: {salon_slug}")
    service = (
        db.query(models.Service)
        .filter(models.Service.salon_id == salon.id, models.Service.name.ilike(service_name))
        .first()
    )
    if not service:
        raise ValueError(f"Unknown service '{service_name}' for salon '{salon_slug}'")
    return service


def handle_get_availability(
    db: Session,
    cal: CalComClient,
    salon_slug: str,
    service_name: str,
    date_from: str,
    date_to: str,
    employee_name: str | None = None,
) -> dict:
    service = _find_service(db, salon_slug, service_name)
    return cal.get_slots(service.cal_event_type_id, date_from, date_to, service.salon.timezone)


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
    service = _find_service(db, salon_slug, service_name)
    return cal.create_booking(service.cal_event_type_id, start_at, customer_name, customer_email, service.salon.timezone)
