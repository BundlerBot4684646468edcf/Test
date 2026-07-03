from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from . import admin, models, schemas
from .admin_ui import render_admin_page
from .cal_client import CalComClient
from .calendar_ui import list_salon_bookings, render_calendar_page
from .config import CAL_API_KEY
from .db import Base, engine, get_db
from .famulor_tools import (
    FAMULOR_TOOLS,
    handle_book_appointment,
    handle_get_availability,
    handle_get_current_datetime,
)
from .onboarding import onboard_salon

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Salon Booking API")


def get_cal_client() -> CalComClient:
    return CalComClient()


def _cal_error(e: RuntimeError) -> HTTPException:
    return HTTPException(status_code=502, detail=str(e))


@app.get("/debug/cal-ping")
def cal_ping(cal: CalComClient = Depends(get_cal_client)):
    """Quick connectivity check — confirms API key works and Cal.com is reachable."""
    if not CAL_API_KEY:
        raise HTTPException(status_code=500, detail="CAL_API_KEY not set")
    try:
        result = cal._request("GET", "/me")
        return {"ok": True, "cal_user": result}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/salons")
def create_salon(
    payload: schemas.SalonOnboardIn,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    try:
        salon = onboard_salon(db, payload, cal)
    except RuntimeError as e:
        raise _cal_error(e)
    return {"id": salon.id, "slug": salon.slug, "cal_team_id": salon.cal_team_id}


@app.get("/famulor/tools")
def list_tools():
    return FAMULOR_TOOLS


@app.post("/famulor/tools/get_current_datetime")
def get_current_datetime(
    payload: schemas.CurrentDatetimeQuery,
    db: Session = Depends(get_db),
):
    try:
        return handle_get_current_datetime(db, payload.salon_slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/famulor/tools/get_availability")
def get_availability(
    payload: schemas.AvailabilityQuery,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    try:
        return handle_get_availability(db, cal, **payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise _cal_error(e)


@app.post("/famulor/tools/book_appointment")
def book_appointment(
    payload: schemas.BookingIn,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    try:
        return handle_book_appointment(db, cal, **payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise _cal_error(e)


def _find_salon_or_404(db: Session, salon_slug: str) -> models.Salon:
    salon = db.query(models.Salon).filter_by(slug=salon_slug).first()
    if not salon:
        raise HTTPException(status_code=404, detail=f"Unknown salon: {salon_slug}")
    return salon


@app.get("/salons/{salon_slug}/bookings")
def salon_bookings(
    salon_slug: str,
    date_from: str,
    date_to: str,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    try:
        return list_salon_bookings(db, cal, salon, date_from, date_to)
    except RuntimeError as e:
        raise _cal_error(e)


@app.get("/salons/{salon_slug}/calendar", response_class=HTMLResponse)
def salon_calendar(salon_slug: str, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return render_calendar_page(salon)


# ----------------- Admin: manage services/employees/qualifications -----------------

def _run_admin(fn, *args, **kwargs):
    """Map admin-layer errors onto HTTP: unknown things -> 404, deletions that
    would orphan upcoming bookings -> 409, Cal.com failures -> 502."""
    try:
        return fn(*args, **kwargs)
    except admin.UpcomingBookingsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise _cal_error(e)


@app.get("/salons/{salon_slug}/admin", response_class=HTMLResponse)
def salon_admin(salon_slug: str, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return render_admin_page(salon)


@app.get("/salons/{salon_slug}/config")
def salon_config(salon_slug: str, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return admin.get_salon_config(db, salon)


@app.post("/salons/{salon_slug}/services")
def add_service(
    salon_slug: str,
    payload: schemas.ServiceIn,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run_admin(admin.add_service, db, cal, salon, payload)


@app.patch("/salons/{salon_slug}/services/{service_id}")
def update_service(
    salon_slug: str,
    service_id: int,
    payload: schemas.ServicePatch,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run_admin(admin.update_service, db, cal, salon, service_id, payload)


@app.delete("/salons/{salon_slug}/services/{service_id}")
def delete_service(
    salon_slug: str,
    service_id: int,
    force: bool = False,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run_admin(admin.delete_service, db, cal, salon, service_id, force)


@app.post("/salons/{salon_slug}/employees")
def add_employee(
    salon_slug: str,
    payload: schemas.EmployeeIn,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run_admin(admin.add_employee, db, cal, salon, payload)


@app.delete("/salons/{salon_slug}/employees/{employee_id}")
def remove_employee(
    salon_slug: str,
    employee_id: int,
    force: bool = False,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run_admin(admin.remove_employee, db, cal, salon, employee_id, force)


@app.put("/salons/{salon_slug}/qualifications")
def set_qualifications(
    salon_slug: str,
    payload: schemas.QualificationsIn,
    db: Session = Depends(get_db),
    cal: CalComClient = Depends(get_cal_client),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run_admin(
        admin.set_qualifications, db, cal, salon, payload.qualifications, payload.force
    )
