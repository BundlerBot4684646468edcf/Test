from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from salon import schemas
from salon.db import Base
from salon.famulor_tools import handle_book_appointment, handle_get_availability
from salon.onboarding import onboard_salon


class FakeCalClient:
    """Stands in for CalComClient so tests don't hit the real Cal.com API."""

    def __init__(self):
        self._next_id = 1

    def _id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def create_team(self, name):
        return {"id": self._id()}

    def invite_team_member(self, team_id, email):
        return {"userId": self._id()}

    def create_event_type(self, team_id, title, slug, length_min, host_user_ids):
        return {"id": self._id()}

    def get_slots(self, event_type_id, date_from, date_to, timezone):
        return {"slots": [f"{date_from}T09:00:00"]}

    def create_booking(self, event_type_id, start_at, attendee_name, attendee_email, timezone):
        return {"id": self._id(), "status": "ACCEPTED"}


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_onboard_creates_salon_with_cal_ids():
    db = make_session()
    cal = FakeCalClient()
    payload = schemas.SalonOnboardIn(
        name="Salon Mueller",
        slug="salon-mueller",
        employees=[schemas.EmployeeIn(name="Lena", email="lena@example.com")],
        services=[schemas.ServiceIn(name="Haarschnitt", duration_min=45)],
    )

    salon = onboard_salon(db, payload, cal)

    assert salon.cal_team_id is not None
    assert len(salon.employees) == 1
    assert salon.employees[0].cal_user_id is not None
    assert len(salon.services) == 1
    assert salon.services[0].cal_event_type_id is not None


def test_famulor_tools_resolve_service_and_call_cal():
    db = make_session()
    cal = FakeCalClient()
    payload = schemas.SalonOnboardIn(
        name="Salon Mueller",
        slug="salon-mueller",
        employees=[schemas.EmployeeIn(name="Lena", email="lena@example.com")],
        services=[schemas.ServiceIn(name="Haarschnitt", duration_min=45)],
    )
    onboard_salon(db, payload, cal)

    slots = handle_get_availability(db, cal, "salon-mueller", "Haarschnitt", "2026-07-01", "2026-07-07")
    assert slots["slots"] == ["2026-07-01T09:00:00"]

    booking = handle_book_appointment(
        db, cal, "salon-mueller", "Haarschnitt", "2026-07-01T09:00:00",
        "Max Mustermann", "max@example.com",
    )
    assert booking["status"] == "ACCEPTED"


def test_famulor_tools_unknown_salon_raises():
    db = make_session()
    cal = FakeCalClient()
    try:
        handle_get_availability(db, cal, "does-not-exist", "Haarschnitt", "2026-07-01", "2026-07-07")
        assert False, "expected ValueError"
    except ValueError:
        pass
