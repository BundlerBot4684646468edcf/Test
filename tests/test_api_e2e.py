"""End-to-end tests against the real HTTP endpoints (FastAPI TestClient).
No external services involved — the booking engine is our own."""

from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from salon.db import Base
from salon.main import app, get_db
from tests.test_salon_onboarding import DATE_FROM, DATE_TO, START_AT


def make_client():
    # StaticPool: keep a single connection alive so the in-memory DB isn't
    # wiped between requests, each of which opens its own session.
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def login(client, slug="schoenheitssalon-test", password="geheim123"):
    r = client.post(f"/salons/{slug}/login", json={"password": password})
    assert r.status_code == 200, r.text


TEST_SALON_PAYLOAD = {
    "name": "Schoenheitssalon Test",
    "slug": "schoenheitssalon-test",
    "timezone": "Europe/Berlin",
    "admin_password": "geheim123",
    "employees": [
        {"name": "Lena", "email": "lena@example.com"},
        {"name": "Tom", "email": "tom@example.com"},
    ],
    "services": [
        {"name": "Coloration", "duration_min": 60, "buffer_min": 10, "price_cents": 8000},
        {"name": "Waschen Foenen", "duration_min": 20, "buffer_min": 0, "price_cents": 1500},
    ],
    "qualifications": [
        {"employee_name": "Lena", "service_name": "Coloration", "duration_min": 45},
        {"employee_name": "Tom", "service_name": "Coloration", "duration_min": 90},
        {"employee_name": "Lena", "service_name": "Waschen Foenen"},
        {"employee_name": "Tom", "service_name": "Waschen Foenen"},
    ],
}


def test_full_flow_onboard_availability_book_conflict():
    client = make_client()

    r = client.post("/salons", json=TEST_SALON_PAYLOAD)
    assert r.status_code == 200, r.text
    assert r.json()["slug"] == "schoenheitssalon-test"

    r = client.post("/salons", json=TEST_SALON_PAYLOAD)
    assert r.status_code == 409  # duplicate slug

    r = client.get("/famulor/tools")
    tool_names = {t["name"] for t in r.json()}
    assert tool_names == {
        "get_current_datetime", "get_availability", "book_appointment", "cancel_appointment",
    }

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 200, r.text
    assert r.json()["duration_min"] == 45
    first_slot = r.json()["slots"][0]

    r = client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": first_slot,
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
        "customer_phone": "0176 1234567",
    })
    assert r.status_code == 200, r.text
    booking = r.json()
    assert booking["status"] == "confirmed"
    assert booking["employee"] == "Lena"

    # same slot again for Lena -> conflict, HTTP 409 with a spoken-word reason
    r = client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": first_slot,
        "customer_name": "Kunde Zwei",
        "customer_email": "zwei@example.com",
    })
    assert r.status_code == 409
    assert "vergeben" in r.json()["detail"]


def test_get_current_datetime_endpoint():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.post("/famulor/tools/get_current_datetime", json={
        "salon_slug": "schoenheitssalon-test",
    })
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["today"] == date.today().isoformat()
    assert len(info["next_days"]) == 7


def test_booking_in_past_returns_error_with_today_hint():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    r = client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "start_at": f"{yesterday}T10:00:00",
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
    })
    assert r.status_code == 404
    assert date.today().isoformat() in r.json()["detail"]


def test_unknown_salon_returns_404():
    client = make_client()
    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "does-not-exist",
        "service_name": "Coloration",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 404


# ----------------- Calendar -----------------

def _book(client, **overrides):
    payload = {
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": START_AT,
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
    }
    payload.update(overrides)
    r = client.post("/famulor/tools/book_appointment", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_calendar_page_and_bookings_endpoint():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)
    _book(client)

    r = client.get("/salons/schoenheitssalon-test/calendar")
    assert r.status_code == 200
    assert "Schoenheitssalon Test" in r.text

    r = client.get(
        f"/salons/schoenheitssalon-test/bookings?date_from={DATE_FROM}&date_to={DATE_TO}"
    )
    assert r.status_code == 200, r.text
    bookings = r.json()
    assert len(bookings) == 1
    assert bookings[0]["customer"] == "Max Mustermann"
    assert bookings[0]["employee"] == "Lena"
    assert bookings[0]["title"] == "Coloration"
    assert bookings[0]["start"] == f"{DATE_FROM}T09:00"


def test_cancel_booking_via_calendar_endpoint():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)
    booking = _book(client)

    r = client.delete(f"/salons/schoenheitssalon-test/bookings/{booking['id']}")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"

    r = client.get(
        f"/salons/schoenheitssalon-test/bookings?date_from={DATE_FROM}&date_to={DATE_TO}"
    )
    assert r.json()[0]["status"] == "cancelled"


def test_cancel_appointment_tool():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    _book(client, customer_phone="0176 1234567")

    r = client.post("/famulor/tools/cancel_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "start_at": START_AT,
        "customer_phone": "0176 1234567",
    })
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cancelled"


# ----------------- Admin -----------------

def test_admin_page_and_config():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)

    r = client.get("/salons/schoenheitssalon-test/admin")
    assert r.status_code == 200
    assert "Verwaltung" in r.text
    assert "Öffnungszeiten" in r.text

    r = client.get("/salons/schoenheitssalon-test/config")
    config = r.json()
    assert {e["name"] for e in config["employees"]} == {"Lena", "Tom"}
    assert len(config["qualifications"]) == 4
    assert len(config["opening_hours"]) == 7


def test_admin_service_duration_change_affects_availability():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)
    config = client.get("/salons/schoenheitssalon-test/config").json()
    waschen_id = next(s["id"] for s in config["services"] if s["name"] == "Waschen Foenen")

    r = client.patch(f"/salons/schoenheitssalon-test/services/{waschen_id}", json={"duration_min": 35})
    assert r.status_code == 200, r.text

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Waschen Foenen",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.json()["duration_min"] == 35


def test_admin_remove_employee_with_booking_gives_409_then_force_works():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)
    _book(client)
    config = client.get("/salons/schoenheitssalon-test/config").json()
    lena_id = next(e["id"] for e in config["employees"] if e["name"] == "Lena")

    r = client.delete(f"/salons/schoenheitssalon-test/employees/{lena_id}")
    assert r.status_code == 409
    assert "Termin" in r.json()["detail"]

    r = client.delete(f"/salons/schoenheitssalon-test/employees/{lena_id}?force=true")
    assert r.status_code == 200
    assert {e["name"] for e in r.json()["employees"]} == {"Tom"}

    # the forced removal cancelled Lena's booking -> visible in the calendar
    r = client.get(
        f"/salons/schoenheitssalon-test/bookings?date_from={DATE_FROM}&date_to={DATE_TO}"
    )
    assert r.json()[0]["status"] == "cancelled"


def test_admin_opening_hours_roundtrip():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)

    hours = [
        {"weekday": w, "open_time": "08:00", "close_time": "16:00", "closed": False}
        for w in range(7)
    ]
    r = client.put("/salons/schoenheitssalon-test/opening-hours", json={"opening_hours": hours})
    assert r.status_code == 200, r.text
    assert all(h["open_time"] == "08:00" for h in r.json()["opening_hours"])

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "date_from": DATE_FROM,
        "date_to": DATE_FROM,
    })
    assert r.json()["slots"][0] == f"{DATE_FROM}T08:00"


def test_admin_qualifications_matrix_roundtrip():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)

    r = client.put("/salons/schoenheitssalon-test/qualifications", json={
        "qualifications": [
            {"employee_name": "Lena", "service_name": "Coloration", "duration_min": 45},
            {"employee_name": "Tom", "service_name": "Coloration", "duration_min": 90},
            {"employee_name": "Lena", "service_name": "Waschen Foenen"},
        ],
    })
    assert r.status_code == 200, r.text
    config = r.json()
    assert len(config["qualifications"]) == 3

    # Tom no longer offers Waschen Foenen
    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Waschen Foenen",
        "employee_name": "Tom",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 404
