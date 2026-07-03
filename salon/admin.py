"""Admin operations: edit services, employees, qualifications and opening
hours after onboarding. Since the booking engine reads everything live from
the database, changes take effect immediately — the only thing to protect
is upcoming bookings, which deletions must not silently orphan
(UpcomingBookingsError, overridable with force: those bookings get
cancelled explicitly)."""

from sqlalchemy.orm import Session

from . import booking, models, schemas

WEEKDAYS_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


class UpcomingBookingsError(Exception):
    """Deletion would orphan upcoming bookings; retry with force to cancel them."""


def _salon_links(db: Session, salon: models.Salon) -> list[models.EmployeeService]:
    return (
        db.query(models.EmployeeService)
        .join(models.Service)
        .filter(models.Service.salon_id == salon.id)
        .all()
    )


def get_salon_config(db: Session, salon: models.Salon) -> dict:
    links = _salon_links(db, salon)
    qualified_service_ids = {l.service_id for l in links if l.employee.active}
    return {
        "name": salon.name,
        "slug": salon.slug,
        "timezone": salon.timezone,
        "employees": [
            {"id": e.id, "name": e.name, "email": e.email}
            for e in salon.employees
            if e.active
        ],
        "services": [
            {
                "id": s.id,
                "name": s.name,
                "duration_min": s.duration_min,
                "buffer_min": s.buffer_min,
                "price_cents": s.price_cents,
                "bookable_any": s.id in qualified_service_ids,
            }
            for s in salon.services
        ],
        "qualifications": [
            {
                "employee_id": l.employee_id,
                "service_id": l.service_id,
                "employee_name": l.employee.name,
                "service_name": l.service.name,
                "duration_min": l.duration_min,
                "effective_duration_min": l.effective_duration_min,
            }
            for l in links
            if l.employee.active
        ],
        "opening_hours": [
            {
                "weekday": h.weekday,
                "weekday_de": WEEKDAYS_DE[h.weekday],
                "open_time": h.open_time,
                "close_time": h.close_time,
                "closed": h.closed,
            }
            for h in sorted(salon.opening_hours, key=lambda h: h.weekday)
        ],
    }


def _guard_upcoming(bookings: list[models.Booking], force: bool, db: Session):
    """Block the change while upcoming bookings depend on it — unless forced,
    then cancel them explicitly so the calendar reflects reality."""
    if not bookings:
        return
    if not force:
        raise UpcomingBookingsError(
            f"Es hängen noch {len(bookings)} bevorstehende Termin(e) daran. "
            "Erst umbuchen/stornieren — oder mit force=true trotzdem fortfahren "
            "(die Termine werden dann storniert)."
        )
    for b in bookings:
        b.status = "cancelled"
    db.flush()


# ----------------- Services -----------------

def add_service(db: Session, salon: models.Salon, data: schemas.ServiceIn) -> dict:
    if any(s.name.lower() == data.name.lower() for s in salon.services):
        raise ValueError(f"Service '{data.name}' existiert bereits")
    db.add(models.Service(
        salon_id=salon.id,
        name=data.name,
        duration_min=data.duration_min,
        buffer_min=data.buffer_min,
        price_cents=data.price_cents,
    ))
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


def update_service(
    db: Session, salon: models.Salon, service_id: int, patch: schemas.ServicePatch
) -> dict:
    service = next((s for s in salon.services if s.id == service_id), None)
    if not service:
        raise ValueError(f"Unbekannter Service: {service_id}")
    if patch.name is not None:
        service.name = patch.name
    if patch.duration_min is not None:
        service.duration_min = patch.duration_min
    if patch.buffer_min is not None:
        service.buffer_min = patch.buffer_min
    if patch.price_cents is not None:
        service.price_cents = patch.price_cents
    # Existing bookings keep their booked end/buffer (snapshotted); only new
    # availability uses the changed values.
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


def delete_service(
    db: Session, salon: models.Salon, service_id: int, force: bool = False
) -> dict:
    service = next((s for s in salon.services if s.id == service_id), None)
    if not service:
        raise ValueError(f"Unbekannter Service: {service_id}")
    _guard_upcoming(booking.upcoming_bookings(db, salon, service_id=service.id), force, db)
    for link in db.query(models.EmployeeService).filter_by(service_id=service.id).all():
        db.delete(link)
    db.delete(service)
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


# ----------------- Employees -----------------

def add_employee(db: Session, salon: models.Salon, data: schemas.EmployeeIn) -> dict:
    if any(e.name.lower() == data.name.lower() and e.active for e in salon.employees):
        raise ValueError(f"Mitarbeiter '{data.name}' existiert bereits")
    db.add(models.Employee(salon_id=salon.id, name=data.name, email=data.email))
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


def remove_employee(
    db: Session, salon: models.Salon, employee_id: int, force: bool = False
) -> dict:
    employee = next((e for e in salon.employees if e.id == employee_id and e.active), None)
    if not employee:
        raise ValueError(f"Unbekannter Mitarbeiter: {employee_id}")
    _guard_upcoming(booking.upcoming_bookings(db, salon, employee_id=employee.id), force, db)
    for link in db.query(models.EmployeeService).filter_by(employee_id=employee.id).all():
        db.delete(link)
    # Soft-delete: keep the row so past bookings stay attributable.
    employee.active = False
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


# ----------------- Qualifications (who does what, and how long) -----------------

def set_qualifications(
    db: Session,
    salon: models.Salon,
    quals: list[schemas.EmployeeServiceIn],
    force: bool = False,
) -> dict:
    """Reconcile the full (employee, service) matrix in one call — the admin
    UI always submits the complete desired state."""
    employees = {e.name.lower(): e for e in salon.employees if e.active}
    services = {s.name.lower(): s for s in salon.services}

    desired: dict[tuple[int, int], int | None] = {}
    for q in quals:
        employee = employees.get(q.employee_name.lower())
        if not employee:
            raise ValueError(f"Unbekannter Mitarbeiter: {q.employee_name}")
        service = services.get(q.service_name.lower())
        if not service:
            raise ValueError(f"Unbekannter Service: {q.service_name}")
        desired[(employee.id, service.id)] = q.duration_min

    current = {
        (l.employee_id, l.service_id): l
        for l in _salon_links(db, salon)
        if l.employee.active
    }

    removed = [l for key, l in current.items() if key not in desired]
    at_risk = [
        b
        for l in removed
        for b in booking.upcoming_bookings(
            db, salon, employee_id=l.employee_id, service_id=l.service_id
        )
    ]
    _guard_upcoming(at_risk, force, db)

    for link in removed:
        db.delete(link)
    for (employee_id, service_id), duration_min in desired.items():
        if (employee_id, service_id) in current:
            current[(employee_id, service_id)].duration_min = duration_min
        else:
            db.add(models.EmployeeService(
                employee_id=employee_id, service_id=service_id, duration_min=duration_min
            ))
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


# ----------------- Opening hours -----------------

def set_opening_hours(
    db: Session, salon: models.Salon, hours: list[schemas.OpeningHoursIn]
) -> dict:
    by_weekday = {h.weekday: h for h in hours}
    for h in by_weekday.values():
        if not (0 <= h.weekday <= 6):
            raise ValueError(f"Ungültiger Wochentag: {h.weekday}")
        if not h.closed and h.open_time >= h.close_time:
            raise ValueError(
                f"{WEEKDAYS_DE[h.weekday]}: Öffnung ({h.open_time}) muss vor Schließung ({h.close_time}) liegen"
            )
    for row in salon.opening_hours:
        incoming = by_weekday.get(row.weekday)
        if incoming:
            row.open_time = incoming.open_time
            row.close_time = incoming.close_time
            row.closed = incoming.closed
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)
