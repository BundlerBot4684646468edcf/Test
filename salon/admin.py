"""Admin operations: edit services, employees and qualifications after
onboarding, keeping the Cal.com event types in sync so availability,
durations, buffers and double-booking protection stay correct.

Invariants maintained against Cal.com:
- every service with >=1 qualified employee has a Round-Robin team event
  type whose hosts are exactly the qualified, active employees
- every (employee, service) qualification has a fixed-host event type
  whose length is the employee override or the service default
- deleting a service/employee/qualification never silently orphans
  upcoming bookings (UpcomingBookingsError, overridable with force)
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from . import models, schemas
from .cal_client import CalComClient


class UpcomingBookingsError(Exception):
    """Deletion would orphan upcoming bookings; retry with force to override."""


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "-")


def _salon_links(db: Session, salon: models.Salon) -> list[models.EmployeeService]:
    return (
        db.query(models.EmployeeService)
        .join(models.Service)
        .filter(models.Service.salon_id == salon.id)
        .all()
    )


def get_salon_config(db: Session, salon: models.Salon) -> dict:
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
                # without a Round-Robin event type the service can't be booked "egal wer"
                "bookable_any": bool(s.cal_event_type_id),
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
            for l in _salon_links(db, salon)
        ],
    }


def _assert_no_upcoming_bookings(cal: CalComClient, event_type_ids: list[str], force: bool):
    ids = [i for i in event_type_ids if i]
    if force or not ids:
        return
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=365)
    bookings = cal.get_bookings(ids, now.isoformat(), horizon.isoformat())
    upcoming = [b for b in bookings if "cancel" not in str(b.get("status", "")).lower()]
    if upcoming:
        raise UpcomingBookingsError(
            f"Es hängen noch {len(upcoming)} bevorstehende Termin(e) daran. "
            "Erst umbuchen/stornieren — oder mit force=true trotzdem löschen."
        )


def _sync_round_robin(db: Session, cal: CalComClient, salon: models.Salon, service: models.Service):
    """Make the service's Round-Robin event type match its current set of
    qualified active employees (create/update/delete as needed)."""
    links = db.query(models.EmployeeService).filter_by(service_id=service.id).all()
    host_ids = [l.employee.cal_user_id for l in links if l.employee.active and l.employee.cal_user_id]

    if not host_ids:
        if service.cal_event_type_id:
            cal.delete_event_type(salon.cal_team_id, service.cal_event_type_id)
            service.cal_event_type_id = None
        return

    if service.cal_event_type_id:
        cal.update_event_type(
            salon.cal_team_id,
            service.cal_event_type_id,
            {"hosts": [{"userId": int(uid), "isFixed": False} for uid in host_ids]},
        )
    else:
        event_type = cal.create_event_type(
            salon.cal_team_id,
            service.name,
            _slugify(f"{service.name}-any"),
            service.duration_min,
            host_ids,
            scheduling_type="ROUND_ROBIN",
            buffer_min=service.buffer_min,
        )
        service.cal_event_type_id = str(event_type.get("id", ""))


# ----------------- Services -----------------

def add_service(db: Session, cal: CalComClient, salon: models.Salon, data: schemas.ServiceIn) -> dict:
    if any(s.name.lower() == data.name.lower() for s in salon.services):
        raise ValueError(f"Service '{data.name}' existiert bereits")
    service = models.Service(
        salon_id=salon.id,
        name=data.name,
        duration_min=data.duration_min,
        buffer_min=data.buffer_min,
        price_cents=data.price_cents,
    )
    db.add(service)
    db.commit()
    db.refresh(salon)
    # Event types are created once qualifications are assigned — until then
    # the service is not bookable, which get_salon_config surfaces.
    return get_salon_config(db, salon)


def update_service(
    db: Session, cal: CalComClient, salon: models.Salon, service_id: int, patch: schemas.ServicePatch
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
    db.flush()

    team_id = salon.cal_team_id
    if service.cal_event_type_id:
        payload = {"title": service.name, "lengthInMinutes": service.duration_min}
        if patch.buffer_min is not None:
            payload["afterEventBuffer"] = service.buffer_min
        cal.update_event_type(team_id, service.cal_event_type_id, payload)

    links = db.query(models.EmployeeService).filter_by(service_id=service.id).all()
    for link in links:
        if not link.cal_event_type_id:
            continue
        payload = {"title": f"{service.name} mit {link.employee.name}"}
        if link.duration_min is None:
            # inherits the service default, so the new default must reach Cal.com
            payload["lengthInMinutes"] = service.duration_min
        if patch.buffer_min is not None:
            payload["afterEventBuffer"] = service.buffer_min
        cal.update_event_type(team_id, link.cal_event_type_id, payload)

    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


def delete_service(
    db: Session, cal: CalComClient, salon: models.Salon, service_id: int, force: bool = False
) -> dict:
    service = next((s for s in salon.services if s.id == service_id), None)
    if not service:
        raise ValueError(f"Unbekannter Service: {service_id}")

    links = db.query(models.EmployeeService).filter_by(service_id=service.id).all()
    event_type_ids = [service.cal_event_type_id] + [l.cal_event_type_id for l in links]
    _assert_no_upcoming_bookings(cal, event_type_ids, force)

    for et_id in event_type_ids:
        if et_id:
            cal.delete_event_type(salon.cal_team_id, et_id)
    for link in links:
        db.delete(link)
    db.delete(service)
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


# ----------------- Employees -----------------

def add_employee(db: Session, cal: CalComClient, salon: models.Salon, data: schemas.EmployeeIn) -> dict:
    if any(e.name.lower() == data.name.lower() and e.active for e in salon.employees):
        raise ValueError(f"Mitarbeiter '{data.name}' existiert bereits")
    membership = cal.invite_team_member(salon.cal_team_id, data.email)
    employee = models.Employee(
        salon_id=salon.id,
        name=data.name,
        email=data.email,
        cal_user_id=str(membership.get("userId", "")),
    )
    db.add(employee)
    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


def remove_employee(
    db: Session, cal: CalComClient, salon: models.Salon, employee_id: int, force: bool = False
) -> dict:
    employee = next((e for e in salon.employees if e.id == employee_id and e.active), None)
    if not employee:
        raise ValueError(f"Unbekannter Mitarbeiter: {employee_id}")

    links = db.query(models.EmployeeService).filter_by(employee_id=employee.id).all()
    _assert_no_upcoming_bookings(cal, [l.cal_event_type_id for l in links], force)

    affected_service_ids = {l.service_id for l in links}
    for link in links:
        if link.cal_event_type_id:
            cal.delete_event_type(salon.cal_team_id, link.cal_event_type_id)
        db.delete(link)
    # Soft-delete: keep the row so past bookings stay attributable. The
    # Cal.com team membership is left in place (removing seats is a manual
    # follow-up in Cal.com, see docs).
    employee.active = False
    db.flush()

    for service in salon.services:
        if service.id in affected_service_ids:
            _sync_round_robin(db, cal, salon, service)

    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)


# ----------------- Qualifications (who does what, and how long) -----------------

def set_qualifications(
    db: Session,
    cal: CalComClient,
    salon: models.Salon,
    quals: list[schemas.EmployeeServiceIn],
    force: bool = False,
) -> dict:
    """Reconcile the full (employee, service) matrix in one call — the admin
    UI always submits the complete desired state."""
    employees = {e.name.lower(): e for e in salon.employees if e.active}
    services = {s.name.lower(): s for s in salon.services}

    desired: dict[tuple[int, int], tuple[models.Employee, models.Service, int | None]] = {}
    for q in quals:
        employee = employees.get(q.employee_name.lower())
        if not employee:
            raise ValueError(f"Unbekannter Mitarbeiter: {q.employee_name}")
        service = services.get(q.service_name.lower())
        if not service:
            raise ValueError(f"Unbekannter Service: {q.service_name}")
        desired[(employee.id, service.id)] = (employee, service, q.duration_min)

    current = {(l.employee_id, l.service_id): l for l in _salon_links(db, salon)}

    removed = [l for key, l in current.items() if key not in desired]
    _assert_no_upcoming_bookings(cal, [l.cal_event_type_id for l in removed], force)

    affected_service_ids: set[int] = set()
    for link in removed:
        if link.cal_event_type_id:
            cal.delete_event_type(salon.cal_team_id, link.cal_event_type_id)
        affected_service_ids.add(link.service_id)
        db.delete(link)

    for key, (employee, service, duration_min) in desired.items():
        if key in current:
            link = current[key]
            if link.duration_min != duration_min:
                link.duration_min = duration_min
                if link.cal_event_type_id:
                    cal.update_event_type(
                        salon.cal_team_id,
                        link.cal_event_type_id,
                        {"lengthInMinutes": duration_min or service.duration_min},
                    )
        else:
            event_type = cal.create_event_type(
                salon.cal_team_id,
                f"{service.name} mit {employee.name}",
                _slugify(f"{service.name}-{employee.name}"),
                duration_min or service.duration_min,
                [employee.cal_user_id],
                buffer_min=service.buffer_min,
            )
            db.add(
                models.EmployeeService(
                    employee_id=employee.id,
                    service_id=service.id,
                    duration_min=duration_min,
                    cal_event_type_id=str(event_type.get("id", "")),
                )
            )
            affected_service_ids.add(service.id)
    db.flush()

    for service in salon.services:
        if service.id in affected_service_ids:
            _sync_round_robin(db, cal, salon, service)

    db.commit()
    db.refresh(salon)
    return get_salon_config(db, salon)
