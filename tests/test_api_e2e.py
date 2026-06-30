"""End-to-end test against the real HTTP endpoints (FastAPI TestClient),
with the Cal.com client swapped for a fake so no network/API key is needed.
Exercises serialization/routing that the unit tests in
test_salon_onboarding.py don't cover."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from salon.db import Base
from salon.main import app, get_cal_client, get_db
from tests.test_salon_onboarding import FakeCalClient


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
    assert tool_names == {"get_availability", "book_appointment"}

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "date_from": "2026-07-01",
        "date_to": "2026-07-07",
    })
    assert r.status_code == 200, r.text
    lena_event_type_id = r.json()["event_type_id"]
    assert fake_cal.event_types[lena_event_type_id]["length_min"] == 45

    r = client.post("/famulor/tools/book_appointment", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Coloration",
        "employee_name": "Lena",
        "start_at": "2026-07-01T09:00:00",
        "customer_name": "Max Mustermann",
        "customer_email": "max@example.com",
    })
    assert r.status_code == 200, r.text
    booking = r.json()
    assert booking["status"] == "ACCEPTED"
    assert booking["event_type_id"] == lena_event_type_id


def test_full_flow_any_employee_uses_round_robin():
    client, fake_cal = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "schoenheitssalon-test",
        "service_name": "Waschen Foenen",
        "date_from": "2026-07-01",
        "date_to": "2026-07-07",
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
        "date_from": "2026-07-01",
        "date_to": "2026-07-07",
    })
    assert r.status_code == 404
    assert "Coloration" in r.json()["detail"]


def test_unknown_salon_returns_404():
    client, _ = make_client()

    r = client.post("/famulor/tools/get_availability", json={
        "salon_slug": "does-not-exist",
        "service_name": "Coloration",
        "date_from": "2026-07-01",
        "date_to": "2026-07-07",
    })
    assert r.status_code == 404
