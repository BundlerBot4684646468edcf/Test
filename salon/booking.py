"""Own booking engine — replaces Cal.com.

Responsibilities:
- generate available slots from opening hours, per-employee durations,
  buffers and existing bookings
- create bookings double-booking-safe: the conflict check runs again inside
  the booking transaction, so a slot that was answered as free but got taken
  in the meantime is rejected (SlotUnavailableError -> HTTP 409)
- round-robin: with no employee preference, a slot is offered if ANY
  qualified employee is free; on booking the freest employee that day gets
  it (load balancing), so requests spread fairly across the team

All datetimes here are naive and mean the salon's local timezone.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from . import models

SLOT_STEP_MIN = 15
MAX_SLOTS = 40  # keep tool responses digestible for a voice bot


class SlotUnavailableError(Exception):
    """The requested time is outside opening hours or already taken."""


def salon_now(salon: models.Salon) -> datetime:
    """Current time as naive local datetime in the salon's timezone."""
    return datetime.now(ZoneInfo(salon.timezone or "Europe/Berlin")).replace(tzinfo=None)


def parse_local(start_at: str, salon: models.Salon) -> datetime:
    """Accept ISO datetimes with or without timezone; result is naive local."""
    try:
        dt = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"start_at must be an ISO datetime, got: {start_at!r}")
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo(salon.timezone or "Europe/Berlin")).replace(tzinfo=None)
    return dt


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def day_window(salon: models.Salon, day: date) -> tuple[datetime, datetime] | None:
    """Opening window for one day, or None when the salon is closed."""
    hours = next((h for h in salon.opening_hours if h.weekday == day.weekday()), None)
    if hours is None or hours.closed:
        return None
    return (
        datetime.combine(day, _parse_hhmm(hours.open_time)),
        datetime.combine(day, _parse_hhmm(hours.close_time)),
    )


def qualified_links(db: Session, service: models.Service) -> list[models.EmployeeService]:
    links = db.query(models.EmployeeService).filter_by(service_id=service.id).all()
    return [l for l in links if l.employee.active]


def _bookings_between(
    db: Session, employee_ids: list[int], window_start: datetime, window_end: datetime
) -> dict[int, list[models.Booking]]:
    """Confirmed bookings per employee that could conflict inside the window
    (generous margin for buffers of neighbouring bookings)."""
    margin = timedelta(hours=6)
    rows = (
        db.query(models.Booking)
        .filter(
            models.Booking.employee_id.in_(employee_ids),
            models.Booking.status == "confirmed",
            models.Booking.start < window_end + margin,
            models.Booking.end > window_start - margin,
        )
        .all()
    )
    by_employee: dict[int, list[models.Booking]] = {eid: [] for eid in employee_ids}
    for b in rows:
        by_employee[b.employee_id].append(b)
    return by_employee


def _is_free(
    existing: list[models.Booking], start: datetime, end: datetime, buffer_min: int
) -> bool:
    """No overlap with any existing booking, buffers included on both sides."""
    new_end_blocked = end + timedelta(minutes=buffer_min)
    for b in existing:
        existing_end_blocked = b.end + timedelta(minutes=b.buffer_min or 0)
        if b.start < new_end_blocked and start < existing_end_blocked:
            return False
    return True


def available_slots(
    db: Session,
    salon: models.Salon,
    service: models.Service,
    from_day: date,
    to_day: date,
    employee: models.Employee | None = None,
) -> dict:
    links = qualified_links(db, service)
    if employee is not None:
        links = [l for l in links if l.employee_id == employee.id]
        if not links:
            raise ValueError(f"{employee.name} bietet '{service.name}' nicht an")
    if not links:
        raise ValueError(f"No employee currently offers '{service.name}'")

    now = salon_now(salon)
    window_start = datetime.combine(from_day, time.min)
    window_end = datetime.combine(to_day, time.max)
    busy = _bookings_between(db, [l.employee_id for l in links], window_start, window_end)

    slots: list[str] = []
    truncated = False
    day = from_day
    while day <= to_day:
        window = day_window(salon, day)
        if window:
            open_dt, close_dt = window
            step = timedelta(minutes=SLOT_STEP_MIN)
            cursor = open_dt
            while cursor < close_dt:
                if cursor >= now:
                    for link in links:
                        duration = link.effective_duration_min
                        end = cursor + timedelta(minutes=duration)
                        if end <= close_dt and _is_free(
                            busy[link.employee_id], cursor, end, service.buffer_min or 0
                        ):
                            slots.append(cursor.isoformat(timespec="minutes"))
                            break  # one free employee is enough to offer the slot
                if len(slots) >= MAX_SLOTS:
                    truncated = True
                    break
                cursor += step
        if truncated:
            break
        day += timedelta(days=1)

    result = {
        "service": service.name,
        "employee": employee.name if employee else None,
        "duration_min": links[0].effective_duration_min if employee else service.duration_min,
        "slots": slots,
    }
    if truncated:
        result["note"] = "Weitere Slots vorhanden — Zeitraum eingrenzen für mehr."
    return result


def _pick_round_robin(
    links: list[models.EmployeeService],
    busy: dict[int, list[models.Booking]],
    start: datetime,
    buffer_min: int,
) -> models.EmployeeService | None:
    """Among employees free at the slot, pick the one with the fewest
    bookings that day (load balancing)."""
    free = []
    for link in links:
        end = start + timedelta(minutes=link.effective_duration_min)
        if _is_free(busy[link.employee_id], start, end, buffer_min):
            day_load = sum(1 for b in busy[link.employee_id] if b.start.date() == start.date())
            free.append((day_load, link.employee_id, link))
    if not free:
        return None
    free.sort(key=lambda item: (item[0], item[1]))
    return free[0][2]


def create_booking(
    db: Session,
    salon: models.Salon,
    service: models.Service,
    start: datetime,
    customer_name: str,
    customer_email: str,
    customer_phone: str | None = None,
    employee: models.Employee | None = None,
) -> models.Booking:
    links = qualified_links(db, service)
    if employee is not None:
        links = [l for l in links if l.employee_id == employee.id]
        if not links:
            raise ValueError(f"{employee.name} bietet '{service.name}' nicht an")
    if not links:
        raise ValueError(f"No employee currently offers '{service.name}'")

    window = day_window(salon, start.date())
    if window is None:
        raise SlotUnavailableError(
            f"Der Salon ist am {start.date().isoformat()} geschlossen."
        )
    open_dt, close_dt = window

    buffer_min = service.buffer_min or 0
    busy = _bookings_between(
        db, [l.employee_id for l in links],
        datetime.combine(start.date(), time.min), datetime.combine(start.date(), time.max),
    )

    # The conflict re-check and the INSERT share this transaction, so a slot
    # taken between availability answer and booking is rejected here.
    link = _pick_round_robin(links, busy, start, buffer_min)
    end_candidates = {
        l.employee_id: start + timedelta(minutes=l.effective_duration_min) for l in links
    }
    if link is not None and not (
        open_dt <= start and end_candidates[link.employee_id] <= close_dt
    ):
        raise SlotUnavailableError(
            f"Außerhalb der Öffnungszeiten ({open_dt.strftime('%H:%M')}–{close_dt.strftime('%H:%M')})."
        )
    if link is None:
        who = employee.name if employee else "kein Mitarbeiter"
        raise SlotUnavailableError(
            f"Der Termin um {start.strftime('%H:%M')} Uhr am {start.date().isoformat()} "
            f"ist inzwischen vergeben ({who} ist nicht mehr frei). Bitte anderen Slot wählen."
        )

    booking = models.Booking(
        salon_id=salon.id,
        employee_id=link.employee_id,
        service_id=service.id,
        customer_name=customer_name,
        customer_email=customer_email,
        customer_phone=customer_phone,
        start=start,
        end=start + timedelta(minutes=link.effective_duration_min),
        buffer_min=buffer_min,
        status="confirmed",
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    return booking


def cancel_booking_by_id(db: Session, salon: models.Salon, booking_id: int) -> models.Booking:
    booking = (
        db.query(models.Booking)
        .filter_by(id=booking_id, salon_id=salon.id)
        .first()
    )
    if not booking:
        raise ValueError(f"Unbekannte Buchung: {booking_id}")
    booking.status = "cancelled"
    db.commit()
    return booking


def cancel_booking(
    db: Session,
    salon: models.Salon,
    start: datetime,
    customer_phone: str | None = None,
    customer_name: str | None = None,
) -> models.Booking:
    """Cancel by appointment time plus phone or name — what a caller can
    realistically provide on the phone."""
    query = db.query(models.Booking).filter_by(
        salon_id=salon.id, start=start, status="confirmed"
    )
    candidates = query.all()
    if customer_phone:
        candidates = [b for b in candidates if b.customer_phone == customer_phone]
    elif customer_name:
        candidates = [b for b in candidates if b.customer_name.lower() == customer_name.lower()]
    if not candidates:
        raise ValueError(
            f"Kein passender Termin am {start.isoformat(timespec='minutes')} gefunden."
        )
    booking = candidates[0]
    booking.status = "cancelled"
    db.commit()
    return booking


def upcoming_bookings(
    db: Session,
    salon: models.Salon,
    employee_id: int | None = None,
    service_id: int | None = None,
) -> list[models.Booking]:
    query = db.query(models.Booking).filter(
        models.Booking.salon_id == salon.id,
        models.Booking.status == "confirmed",
        models.Booking.start >= salon_now(salon),
    )
    if employee_id is not None:
        query = query.filter(models.Booking.employee_id == employee_id)
    if service_id is not None:
        query = query.filter(models.Booking.service_id == service_id)
    return query.all()
