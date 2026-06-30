from sqlalchemy.orm import Session

from . import models, schemas
from .cal_client import CalComClient


def onboard_salon(db: Session, data: schemas.SalonOnboardIn, cal: CalComClient) -> models.Salon:
    """Provision a new salon: Cal.com team, hosts (employees), event types (services)."""
    team = cal.create_team(data.name)
    team_id = str(team["id"])

    salon = models.Salon(name=data.name, slug=data.slug, timezone=data.timezone, cal_team_id=team_id)
    db.add(salon)
    db.flush()

    host_ids = []
    for emp in data.employees:
        membership = cal.invite_team_member(team_id, emp.email)
        cal_user_id = str(membership.get("userId", ""))
        host_ids.append(cal_user_id)
        db.add(models.Employee(salon_id=salon.id, name=emp.name, email=emp.email, cal_user_id=cal_user_id))

    for svc in data.services:
        slug = svc.name.lower().replace(" ", "-")
        event_type = cal.create_event_type(team_id, svc.name, slug, svc.duration_min, host_ids)
        db.add(
            models.Service(
                salon_id=salon.id,
                name=svc.name,
                duration_min=svc.duration_min,
                buffer_min=svc.buffer_min,
                price_cents=svc.price_cents,
                cal_event_type_id=str(event_type.get("id", "")),
            )
        )

    db.commit()
    db.refresh(salon)
    return salon
