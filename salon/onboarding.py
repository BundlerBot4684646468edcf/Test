from collections import defaultdict

from sqlalchemy.orm import Session

from . import models, schemas
from .cal_client import CalComClient


def onboard_salon(db: Session, data: schemas.SalonOnboardIn, cal: CalComClient) -> models.Salon:
    """Provision a new salon in Cal.com.

    For each service this creates:
    - one Round-Robin team event type covering every qualified employee, used
      when the customer has no employee preference ("egal wer")
    - one individual fixed-host event type per (employee, service), used when
      Famulor/the customer names a specific employee. Its duration can
      override the service default, since the same service can take a
      different amount of time depending on who performs it.
    """
    team = cal.create_team(data.name)
    team_id = str(team["id"])

    salon = models.Salon(name=data.name, slug=data.slug, timezone=data.timezone, cal_team_id=team_id)
    db.add(salon)
    db.flush()

    employees_by_name: dict[str, models.Employee] = {}
    for emp in data.employees:
        membership = cal.invite_team_member(team_id, emp.email)
        cal_user_id = str(membership.get("userId", ""))
        employee = models.Employee(salon_id=salon.id, name=emp.name, email=emp.email, cal_user_id=cal_user_id)
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
        quals = quals_by_service.get(service.id, [])
        host_ids = [employees_by_name[q.employee_name.lower()].cal_user_id for q in quals]

        if host_ids:
            any_slug = f"{service.name}-any".lower().replace(" ", "-")
            any_event_type = cal.create_event_type(
                team_id, service.name, any_slug, service.duration_min, host_ids,
                scheduling_type="ROUND_ROBIN", buffer_min=service.buffer_min,
            )
            service.cal_event_type_id = str(any_event_type.get("id", ""))

        for q in quals:
            employee = employees_by_name[q.employee_name.lower()]
            duration = q.duration_min or service.duration_min
            slug = f"{service.name}-{employee.name}".lower().replace(" ", "-")
            event_type = cal.create_event_type(
                team_id, f"{service.name} mit {employee.name}", slug, duration,
                [employee.cal_user_id], buffer_min=service.buffer_min,
            )
            db.add(
                models.EmployeeService(
                    employee_id=employee.id,
                    service_id=service.id,
                    duration_min=q.duration_min,
                    cal_event_type_id=str(event_type.get("id", "")),
                )
            )

    db.commit()
    db.refresh(salon)
    return salon
