from collections import defaultdict

from sqlalchemy.orm import Session

from . import models, schemas

# Default weekly opening hours for new salons; editable in the admin UI.
DEFAULT_OPENING_HOURS = [
    # (weekday 0=Mo, open, close, closed)
    (0, "09:00", "18:00", False),
    (1, "09:00", "18:00", False),
    (2, "09:00", "18:00", False),
    (3, "09:00", "18:00", False),
    (4, "09:00", "18:00", False),
    (5, "09:00", "14:00", False),
    (6, "09:00", "18:00", True),
]


def onboard_salon(db: Session, data: schemas.SalonOnboardIn) -> models.Salon:
    """Create a new salon with employees, services, the who-does-what matrix
    (with per-employee durations) and default opening hours. Everything lives
    in our own database — availability and double-booking protection are
    handled by salon/booking.py."""
    salon = models.Salon(name=data.name, slug=data.slug, timezone=data.timezone)
    db.add(salon)
    db.flush()

    for weekday, open_time, close_time, closed in DEFAULT_OPENING_HOURS:
        db.add(models.OpeningHours(
            salon_id=salon.id, weekday=weekday,
            open_time=open_time, close_time=close_time, closed=closed,
        ))

    employees_by_name: dict[str, models.Employee] = {}
    for emp in data.employees:
        employee = models.Employee(salon_id=salon.id, name=emp.name, email=emp.email)
        db.add(employee)
        db.flush()
        employees_by_name[emp.name.lower()] = employee

    services_by_name: dict[str, models.Service] = {}
    for svc in data.services:
        service = models.Service(
            salon_id=salon.id,
            name=svc.name,
            duration_min=svc.duration_min,
            buffer_min=svc.buffer_min,
            price_cents=svc.price_cents,
        )
        db.add(service)
        db.flush()
        services_by_name[svc.name.lower()] = service

    qualifications = data.qualifications or [
        schemas.EmployeeServiceIn(employee_name=emp.name, service_name=svc.name)
        for emp in data.employees
        for svc in data.services
    ]

    quals_by_service: dict[int, list[schemas.EmployeeServiceIn]] = defaultdict(list)
    for q in qualifications:
        service = services_by_name[q.service_name.lower()]
        quals_by_service[service.id].append(q)

    for service in services_by_name.values():
        for q in quals_by_service.get(service.id, []):
            employee = employees_by_name[q.employee_name.lower()]
            db.add(
                models.EmployeeService(
                    employee_id=employee.id,
                    service_id=service.id,
                    duration_min=q.duration_min,
                )
            )

    db.commit()
    db.refresh(salon)
    return salon
