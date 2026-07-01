from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import schemas
from .cal_client import CalComClient
from .config import CAL_API_KEY
from .db import Base, engine, get_db
from .famulor_tools import FAMULOR_TOOLS, handle_book_appointment, handle_get_availability
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
