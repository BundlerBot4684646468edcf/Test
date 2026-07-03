import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from . import admin, auth, models, schemas
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


# ----------------- Auth -----------------

def _session_ok(request: Request, salon_slug: str) -> bool:
    return auth.verify_session_token(request.cookies.get(auth.COOKIE_NAME), salon_slug)


def _require_dashboard(request: Request, salon_slug: str):
    """API guard for salon dashboard endpoints."""
    if not _session_ok(request, salon_slug):
        raise HTTPException(
            status_code=401,
            detail=f"Nicht eingeloggt — bitte unter /salons/{salon_slug}/login anmelden.",
        )


def _page_or_login(request: Request, salon_slug: str, render):
    """HTML pages redirect to the login form instead of returning 401."""
    if not _session_ok(request, salon_slug):
        return RedirectResponse(
            f"/salons/{salon_slug}/login?next={request.url.path}", status_code=303
        )
    return HTMLResponse(render())


def _require_famulor_key(request: Request):
    """If FAMULOR_API_KEY is set, the bot endpoints require it as X-API-Key.
    Unset = open (local development)."""
    required = os.getenv("FAMULOR_API_KEY")
    if required and request.headers.get("x-api-key") != required:
        raise HTTPException(status_code=401, detail="Ungültiger oder fehlender X-API-Key")


@app.get("/salons/{salon_slug}/login", response_class=HTMLResponse)
def login_page(salon_slug: str, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return auth.render_login_page(salon.name, salon.slug)


@app.post("/salons/{salon_slug}/login")
def login(salon_slug: str, payload: schemas.LoginIn, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    if not auth.verify_password(payload.password, salon.admin_password_hash):
        raise HTTPException(status_code=401, detail="Falsches Passwort")
    response = JSONResponse({"ok": True})
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.make_session_token(salon.slug),
        path=f"/salons/{salon.slug}",
        httponly=True,
        samesite="lax",
        max_age=auth.SESSION_TTL_SECONDS,
    )
    return response


@app.get("/salons/{salon_slug}/logout")
def logout(salon_slug: str):
    response = RedirectResponse(f"/salons/{salon_slug}/login", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME, path=f"/salons/{salon_slug}")
    return response


@app.post("/salons")
def create_salon(
    payload: schemas.SalonOnboardIn, request: Request, db: Session = Depends(get_db)
):
    # If PLATFORM_ADMIN_KEY is set, creating salons requires it as X-Admin-Key.
    required = os.getenv("PLATFORM_ADMIN_KEY")
    if required and request.headers.get("x-admin-key") != required:
        raise HTTPException(status_code=401, detail="Ungültiger oder fehlender X-Admin-Key")
    if db.query(models.Salon).filter_by(slug=payload.slug).first():
        raise HTTPException(status_code=409, detail=f"Salon '{payload.slug}' existiert bereits")
    password = payload.admin_password or secrets.token_urlsafe(9)
    salon = _run(onboard_salon, db, payload, auth.hash_password(password))
    result = {"id": salon.id, "slug": salon.slug, "login_url": f"/salons/{salon.slug}/login"}
    if not payload.admin_password:
        # auto-generated: shown exactly once, only the hash is stored
        result["admin_password"] = password
    return result


# ----------------- Famulor function-calling tools -----------------

@app.get("/famulor/tools")
def list_tools():
    return FAMULOR_TOOLS


@app.post("/famulor/tools/get_current_datetime", dependencies=[Depends(_require_famulor_key)])
def get_current_datetime(payload: schemas.CurrentDatetimeQuery, db: Session = Depends(get_db)):
    return _run(handle_get_current_datetime, db, payload.salon_slug)


@app.post("/famulor/tools/get_availability", dependencies=[Depends(_require_famulor_key)])
def get_availability(payload: schemas.AvailabilityQuery, db: Session = Depends(get_db)):
    return _run(handle_get_availability, db, **payload.model_dump())


@app.post("/famulor/tools/book_appointment", dependencies=[Depends(_require_famulor_key)])
def book_appointment(payload: schemas.BookingIn, db: Session = Depends(get_db)):
    return _run(handle_book_appointment, db, **payload.model_dump())


@app.post("/famulor/tools/cancel_appointment", dependencies=[Depends(_require_famulor_key)])
def cancel_appointment(payload: schemas.CancelIn, db: Session = Depends(get_db)):
    return _run(handle_cancel_appointment, db, **payload.model_dump())


# ----------------- Calendar (login required) -----------------

@app.get("/salons/{salon_slug}/bookings")
def salon_bookings(
    salon_slug: str,
    date_from: str,
    date_to: str,
    request: Request,
    db: Session = Depends(get_db),
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return list_salon_bookings(db, salon, date_from, date_to)


@app.delete("/salons/{salon_slug}/bookings/{booking_id}")
def cancel_booking(
    salon_slug: str, booking_id: int, request: Request, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    booking = _run(cancel_booking_by_id, db, salon, booking_id)
    return {"id": booking.id, "status": booking.status}


@app.get("/salons/{salon_slug}/calendar")
def salon_calendar(salon_slug: str, request: Request, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return _page_or_login(request, salon_slug, lambda: render_calendar_page(salon))


# ----------------- Admin: services/employees/qualifications/opening hours (login required) -----------------

@app.get("/salons/{salon_slug}/admin")
def salon_admin(salon_slug: str, request: Request, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    return _page_or_login(request, salon_slug, lambda: render_admin_page(salon))


@app.get("/salons/{salon_slug}/config")
def salon_config(salon_slug: str, request: Request, db: Session = Depends(get_db)):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return admin.get_salon_config(db, salon)


@app.post("/salons/{salon_slug}/services")
def add_service(
    salon_slug: str,
    request: Request, payload: schemas.ServiceIn, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.add_service, db, salon, payload)


@app.patch("/salons/{salon_slug}/services/{service_id}")
def update_service(
    salon_slug: str,
    request: Request,
    service_id: int,
    payload: schemas.ServicePatch,
    db: Session = Depends(get_db),
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.update_service, db, salon, service_id, payload)


@app.delete("/salons/{salon_slug}/services/{service_id}")
def delete_service(
    salon_slug: str,
    request: Request, service_id: int, force: bool = False, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.delete_service, db, salon, service_id, force)


@app.post("/salons/{salon_slug}/employees")
def add_employee(
    salon_slug: str,
    request: Request, payload: schemas.EmployeeIn, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.add_employee, db, salon, payload)


@app.delete("/salons/{salon_slug}/employees/{employee_id}")
def remove_employee(
    salon_slug: str,
    request: Request, employee_id: int, force: bool = False, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.remove_employee, db, salon, employee_id, force)


@app.put("/salons/{salon_slug}/qualifications")
def set_qualifications(
    salon_slug: str,
    request: Request, payload: schemas.QualificationsIn, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.set_qualifications, db, salon, payload.qualifications, payload.force)


@app.put("/salons/{salon_slug}/opening-hours")
def set_opening_hours(
    salon_slug: str,
    request: Request, payload: schemas.OpeningHoursUpdate, db: Session = Depends(get_db)
):
    salon = _find_salon_or_404(db, salon_slug)
    _require_dashboard(request, salon_slug)
    return _run(admin.set_opening_hours, db, salon, payload.opening_hours)
