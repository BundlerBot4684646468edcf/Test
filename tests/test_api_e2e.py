"""End-to-end test against the real HTTP endpoints (FastAPI TestClient),
with the Cal.com client swapped for a fake so no network/API key is needed.
Exercises serialization/routing that the unit tests in
test_salon_onboarding.py don't cover."""

from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from salon.db import Base
from salon.main import app, get_cal_client, get_db
from tests.test_salon_onboarding import DATE_FROM, DATE_TO, START_AT, FakeCalClient


def make_client():
    # StaticPool: keep a single connection alive so the in-memory DB isn't
    # wiped between requests, each of which opens its own session.
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    fake_cal = FakeCalClient()

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_cal_client] = lambda: fake_cal
    client = TestClient(app)
    return client, fake_cal


TEST_SALON_PAYLOAD = {
    "name": "Schoenheitssalon Test",
    "slug": "schoenheitssalon-test",
    "timezone": "Europe/Berlin",
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


def test_full_flow_onboard_then_book_specific_employee():
    client, fake_cal = make_client()

    r = client.post("/salons", json=TEST_SALON_PAYLOAD)
    assert r.status_code == 200, r.text
    salon = r.json()
    assert salon["slug"] == "schoenheitssalon-test"
    assert salon["cal_team_id"] is not None

    r = client.get("/famulor/tools")
    assert r.status_code == 200
    tool_names = {t["name"] for t in r.json()}
    assert tool_names == {"get_current_datetime", "get_availability", "book_appointment"}

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 200, r.text
    lena_event_type_id = r.json()["event_type_id"]
    assert fake_cal.event_types[lena_event_type_id]["length_min"] == 45

    r = client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": START_AT,
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
        "customer_phone": "0176 1234567",
    })
    assert r.status_code == 200, r.text
    booking = r.json()
    assert booking["status"] == "ACCEPTED"
    assert booking["event_type_id"] == lena_event_type_id
    # the phone number must reach Cal.com, otherwise no confirmation/reminder SMS
    assert fake_cal.bookings[-1]["attendee_phone"] == "+491761234567"


def test_full_flow_any_employee_uses_round_robin():
    client, fake_cal = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Waschen Foenen",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 200, r.text
    event_type_id = r.json()["event_type_id"]
    event_type = fake_cal.event_types[event_type_id]
    assert event_type["scheduling_type"] == "ROUND_ROBIN"
    assert len(event_type["host_user_ids"]) == 2


def test_unknown_employee_returns_404_with_clear_message():
    client, _ = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Nina",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 404
    assert "Coloration" in r.json()["detail"]


def test_unknown_salon_returns_404():
    client, _ = make_client()

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "does-not-exist",
        "service_name": "Coloration",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 404


def test_get_current_datetime_endpoint():
    client, _ = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.post("/famulor/tools/get_current_datetime", json={
        "salon_slug": "schoenheitssalon-test",
    })
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["today"] == date.today().isoformat()
    assert info["timezone"] == "Europe/Berlin"
    assert len(info["next_days"]) == 7


def test_booking_in_past_returns_error_with_today_hint():
    client, fake_cal = make_client()
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
    assert fake_cal.bookings == []


# ----------------- Admin UI + endpoints -----------------

def test_admin_page_renders_and_config_endpoint_works():
    client, _ = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.get("/salons/schoenheitssalon-test/admin")
    assert r.status_code == 200
    assert "Verwaltung" in r.text

    r = client.get("/salons/schoenheitssalon-test/config")
    assert r.status_code == 200
    config = r.json()
    assert {e["name"] for e in config["employees"]} == {"Lena", "Tom"}
    assert len(config["qualifications"]) == 4


def test_admin_service_duration_change_updates_availability_slot_length():
    client, fake_cal = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    service_id = client.get("/salons/schoenheitssalon-test/config").json()["services"][0]["id"]

    r = client.patch(f"/salons/schoenheitssalon-test/services/{service_id}", json={"duration_min": 75})
    assert r.status_code == 200, r.text

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    event_type_id = r.json()["event_type_id"]
    assert fake_cal.event_types[event_type_id]["length_min"] == 75


def test_admin_remove_employee_with_booking_gives_409_then_force_works():
    client, _ = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": START_AT,
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
    })
    config = client.get("/salons/schoenheitssalon-test/config").json()
    lena_id = next(e["id"] for e in config["employees"] if e["name"] == "Lena")

    r = client.delete(f"/salons/schoenheitssalon-test/employees/{lena_id}")
    assert r.status_code == 409
    assert "Termin" in r.json()["detail"]

    r = client.delete(f"/salons/schoenheitssalon-test/employees/{lena_id}?force=true")
    assert r.status_code == 200
    assert {e["name"] for e in r.json()["employees"]} == {"Tom"}

    # booking for Lena by name must now fail with a clear message
    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
    })
    assert r.status_code == 404


def test_admin_qualifications_matrix_roundtrip():
    client, fake_cal = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

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
    waschen = next(s for s in config["services"] if s["name"] == "Waschen Foenen")
    assert waschen["bookable_any"] is True
    rr = next(
        et for et in fake_cal.event_types.values()
        if et["scheduling_type"] == "ROUND_ROBIN" and et["title"] == "Waschen Foenen"
    )
    assert len(rr["host_user_ids"]) == 1  # only Lena offers it now


# ----------------- Calendar UI -----------------

def test_calendar_page_renders_for_salon():
    client, _ = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.get("/salons/schoenheitssalon-test/calendar")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Schoenheitssalon Test" in r.text
    assert "/bookings?date_from=" in r.text  # page fetches our bookings endpoint

    r = client.get("/salons/does-not-exist/calendar")
    assert r.status_code == 404


def test_bookings_endpoint_returns_bookings_for_calendar():
    client, _ = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": START_AT,
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
        "customer_phone": "+491761234567",
    })

    r = client.get(
        f"/salons/schoenheitssalon-test/bookings?date_from={DATE_FROM}&date_to={DATE_TO}"
    )
    assert r.status_code == 200, r.text
    bookings = r.json()
    assert len(bookings) == 1
    b = bookings[0]
    assert b["customer"] == "Max Mustermann"
    assert b["start"].startswith(DATE_FROM)
    assert b["status"] == "ACCEPTED"
