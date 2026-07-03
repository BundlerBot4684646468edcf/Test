from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from salon import schemas
from salon.db import Base
from salon.famulor_tools import (
    handle_book_appointment,
    handle_cancel_appointment,
    handle_get_availability,
    handle_get_current_datetime,
    normalize_phone,
)
from salon.onboarding import onboard_salon

# Tests book on a Monday at least a week out: guaranteed in the future and
# an open day under the default opening hours (Mon-Fri 09-18, Sat 09-14).
_d = date.today() + timedelta(days=7)
while _d.weekday() != 0:
    _d += timedelta(days=1)
BOOK_DAY = _d
DATE_FROM = BOOK_DAY.isoformat()
DATE_TO = (BOOK_DAY + timedelta(days=5)).isoformat()  # Mon-Sat
START_AT = f"{DATE_FROM}T09:00:00"


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _onboard_salon_with_two_stylists(db):
    payload = schemas.SalonOnboardIn(
        name="Salon Mueller",
        slug="salon-mueller",
        employees=[
            schemas.EmployeeIn(name="Lena", email="lena@example.com"),
            schemas.EmployeeIn(name="Tom", email="tom@example.com"),
        ],
        services=[schemas.ServiceIn(name="Coloration", duration_min=60)],
        qualifications=[
            # Lena is faster than the service default; Tom is a junior and slower.
            schemas.EmployeeServiceIn(employee_name="Lena", service_name="Coloration", duration_min=45),
            schemas.EmployeeServiceIn(employee_name="Tom", service_name="Coloration", duration_min=90),
        ],
    )
    return onboard_salon(db, payload)


def test_onboard_creates_salon_with_defaults():
    db = make_session()
    salon = _onboard_salon_with_two_stylists(db)

    assert salon.slug == "salon-mueller"
    assert len(salon.employees) == 2
    assert len(salon.services) == 1
    assert len(salon.opening_hours) == 7
    sunday = next(h for h in salon.opening_hours if h.weekday == 6)
    assert sunday.closed


def test_availability_and_booking_roundtrip():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    slots = handle_get_availability(db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO)
    assert f"{DATE_FROM}T09:00" in slots["slots"]

    booking = handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com",
    )
    assert booking["status"] == "confirmed"
    assert booking["employee"] in {"Lena", "Tom"}


def test_named_employee_gets_their_own_duration():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    lena = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Lena"
    )
    tom = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Tom"
    )
    assert lena["duration_min"] == 45
    assert tom["duration_min"] == 90

    booking = handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", employee_name="Tom",
    )
    assert booking["employee"] == "Tom"
    assert booking["start"] == f"{DATE_FROM}T09:00"
    assert booking["end"] == f"{DATE_FROM}T10:30"  # 90 minutes for Tom


def test_unknown_salon_service_employee_raise():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    for args in [
        ("does-not-exist", "Coloration"),
        ("salon-mueller", "Dauerwelle"),
    ]:
        try:
            handle_get_availability(db, args[0], args[1], DATE_FROM, DATE_TO)
            assert False, "expected ValueError"
        except ValueError:
            pass

    try:
        handle_get_availability(
            db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Nina"
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cancel_appointment_frees_the_slot():
    db = make_session()
    _onboard_salon_with_two_stylists(db)
    handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com",
        employee_name="Lena", customer_phone="0176 1234567",
    )

    result = handle_cancel_appointment(
        db, "salon-mueller", START_AT, customer_phone="+491761234567"
    )
    assert result["status"] == "cancelled"

    # Lena is free again at 09:00
    slots = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_FROM, employee_name="Lena"
    )
    assert f"{DATE_FROM}T09:00" in slots["slots"]


def test_cancel_unknown_appointment_raises():
    db = make_session()
    _onboard_salon_with_two_stylists(db)
    try:
        handle_cancel_appointment(db, "salon-mueller", START_AT, customer_name="Niemand")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ----------------- Date awareness ("Samstag war gestern"-Bug) -----------------

def test_get_current_datetime_reports_today_and_weekdays():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    info = handle_get_current_datetime(db, "salon-mueller")

    assert info["today"] == date.today().isoformat()
    assert info["timezone"] == "Europe/Berlin"
    assert len(info["next_days"]) == 7
    assert info["next_days"][0]["date"] == (date.today() + timedelta(days=1)).isoformat()
    assert info["today_weekday_de"] in {"Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"}


def test_availability_range_fully_in_past_is_rejected_with_today_hint():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    past_from = (date.today() - timedelta(days=8)).isoformat()
    past_to = (date.today() - timedelta(days=2)).isoformat()
    try:
        handle_get_availability(db, "salon-mueller", "Coloration", past_from, past_to)
        assert False, "expected ValueError"
    except ValueError as e:
        assert date.today().isoformat() in str(e)  # tells the bot what today is


def test_availability_start_in_past_is_clamped_to_today():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    past_from = (date.today() - timedelta(days=2)).isoformat()
    future_to = (date.today() + timedelta(days=5)).isoformat()
    slots = handle_get_availability(db, "salon-mueller", "Coloration", past_from, future_to)

    assert slots["today"] == date.today().isoformat()
    assert all(s >= date.today().isoformat() for s in slots["slots"])


def test_booking_in_the_past_is_rejected():
    db = make_session()
    salon = _onboard_salon_with_two_stylists(db)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        handle_book_appointment(
            db, "salon-mueller", "Coloration", f"{yesterday}T09:00:00",
            "Max Mustermann", "max@example.com",
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Vergangenheit" in str(e)

    from salon.models import Booking
    assert db.query(Booking).count() == 0


# ----------------- Phone normalization (stored for SMS later) -----------------

def test_booking_stores_normalized_phone():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    booking = handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", customer_phone="0176 123 45678",
    )
    from salon.models import Booking
    stored = db.get(Booking, booking["id"])
    assert stored.customer_phone == "+4917612345678"


def test_phone_normalization_variants():
    assert normalize_phone("+49 176 1234567") == "+491761234567"
    assert normalize_phone("0049 176 1234567") == "+491761234567"
    assert normalize_phone("0176/1234567") == "+491761234567"
    assert normalize_phone(None) is None
    assert normalize_phone("") is None
