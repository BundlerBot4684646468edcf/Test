from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from . import admin, models, schemas
from .admin_ui import render_admin_page
from .booking import SlotUnavailableError, cancel_booking_by_id
from .calendar_ui import list_salon_bookings, render_calendar_page
from .db import Base, engine, get_db
from .famulor_tools import (
    FAMULOR_TOOLS,
    handle_book_appointment,
    handle_cancel_appointment,
    handle_get_availability,
    handle_get_current_datetime,
)
from .onboarding import onboard_salon

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Salon Booking API")


def _find_salon_or_404(db: Session, salon_slug: str) -> models.Salon:
    salon = db.query(models.Salon).filter_by(slug=salon_slug).first()
    if not salon:
        raise HTTPException(status_code=404, detail=f"Unknown salon: {salon_slug}")
    return salon


def _run(fn, *args, **kwargs):
    """Map domain errors onto HTTP: unknown things -> 404, slots that are
    gone/blocked -> 409 (also deletions that would orphan bookings)."""
    try:
        return fn(*args, **kwargs)
    except (SlotUnavailableError, admin.UpcomingBookingsError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/salons")
def create_salon(payload: schemas.SalonOnboardIn, db: Session = Depends(get_db)):
    if db.query(models.Salon).filter_by(slug=payload.slug).first():
        raise HTTPException(status_code=409, detail=f"Salon '{payload.slug}' existiert bereits")
    salon = _run(onboard_salon, db, payload)
    return {"id": salon.id, "slug": salon.slug}


# ----------------- Famulor function-calling tools -----------------

@app.get("/famulor/tools")
def list_tools():
    return FAMULOR_TOOLS


@app.post("/famulor/tools/get_current_datetime")
def get_current_datetime(payload: schemas.CurrentDatetimeQuery, db: Session = Depends(get_db)):
    return _run(handle_get_current_datetime, db, payload.salon_slug)


@app.post("/famulor/tools/get_availability")
def get_availability(payload: schemas.AvailabilityQuery, db: Session = Depends(get_db)):
    return _run(handle_get_availability, db, **payload.model_dump())


@app.post("/famulor/tools/book_appointment")
def book_appointment(payload: schemas.BookingIn, db: Session = Depends(get_db)):
    return _run(handle_book_appointment, db, **payload.model_dump())


@app.post("/famulor/tools/cancel_appointment")
def cancel_appointment(payload: schemas.CancelIn, db: Session = Depends(get_db)):
    return _run(handle_cancel_appointment, db, **payload.model_dump())


# ----------------- Calendar -----------------

@app.get("/salons/{salon_slug}/bookings")
def salon_bookings(
    salon_slug: str, date_from: str, date_to: str, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return list_salon_bookings(db, salon, date_from, date_to)


@app.delete("/salons/{salon_slug}/bookings/{booking_id}")
def cancel_booking(salon_slug: str, booking_id: int, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    booking = _run(cancel_booking_by_id, db, salon, booking_id)
    return {"id": booking.id, "status": booking.status}


@app.get("/salons/{salon_slug}/calendar", response_class=HTMLResponse)
def salon_calendar(salon_slug: str, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return render_calendar_page(salon)


# ----------------- Admin: services/employees/qualifications/opening hours -----------------

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
    salon_slug: str, payload: schemas.ServiceIn, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.add_service, db, salon, payload)


@app.patch("/salons/{salon_slug}/services/{service_id}")
def update_service(
    salon_slug: str,
    service_id: int,
    payload: schemas.ServicePatch,
    db: Session = Depends(get_db),
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.update_service, db, salon, service_id, payload)


@app.delete("/salons/{salon_slug}/services/{service_id}")
def delete_service(
    salon_slug: str, service_id: int, force: bool = False, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.delete_service, db, salon, service_id, force)


@app.post("/salons/{salon_slug}/employees")
def add_employee(
    salon_slug: str, payload: schemas.EmployeeIn, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.add_employee, db, salon, payload)


@app.delete("/salons/{salon_slug}/employees/{employee_id}")
def remove_employee(
    salon_slug: str, employee_id: int, force: bool = False, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.remove_employee, db, salon, employee_id, force)


@app.put("/salons/{salon_slug}/qualifications")
def set_qualifications(
    salon_slug: str, payload: schemas.QualificationsIn, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.set_qualifications, db, salon, payload.qualifications, payload.force)


@app.put("/salons/{salon_slug}/opening-hours")
def set_opening_hours(
    salon_slug: str, payload: schemas.OpeningHoursUpdate, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    return _run(admin.set_opening_hours, db, salon, payload.opening_hours)
