from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from salon import schemas
from salon.db import Base
from salon.famulor_tools import handle_book_appointment, handle_get_availability
from salon.onboarding import onboard_salon


class FakeCalClient:
    """Stands in for CalComClient so tests don't hit the real Cal.com API.

    Records created event types keyed by id so tests can assert which
    duration/hosts ended up on which event type.
    """

    def __init__(self):
        self._next_id = 1
        self.event_types: dict[str, dict] = {}
        self.bookings: list[dict] = []

    def _id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def create_team(self, name):
        return {"id": self._id()}

    def invite_team_member(self, team_id, email):
        return {"userId": self._id()}

    def create_event_type(self, team_id, title, slug, length_min, host_user_ids, scheduling_type=None):
        event_id = self._id()
        self.event_types[event_id] = {
            "title": title,
            "length_min": length_min,
            "host_user_ids": list(host_user_ids),
            "scheduling_type": scheduling_type,
        }
        return {"id": event_id}

    def get_slots(self, event_type_id, date_from, date_to, timezone):
        return {"event_type_id": event_type_id, "slots": [f"{date_from}T09:00:00"]}

    def create_booking(self, event_type_id, start_at, attendee_name, attendee_email, timezone):
        booking = {"id": self._id(), "event_type_id": event_type_id, "status": "ACCEPTED"}
        self.bookings.append(booking)
        return booking


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


def _onboard_salon_with_two_stylists(db, cal):
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
    return onboard_salon(db, payload, cal)


def test_employee_specific_event_type_carries_its_own_duration():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    slots = handle_get_availability(
        db, cal, "salon-mueller", "Coloration", "2026-07-01", "2026-07-07", employee_name="Lena"
    )
    lena_event_type_id = slots["event_type_id"]
    assert cal.event_types[lena_event_type_id]["length_min"] == 45

    slots = handle_get_availability(
        db, cal, "salon-mueller", "Coloration", "2026-07-01", "2026-07-07", employee_name="Tom"
    )
    tom_event_type_id = slots["event_type_id"]
    assert cal.event_types[tom_event_type_id]["length_min"] == 90
    assert tom_event_type_id != lena_event_type_id


def test_no_employee_preference_uses_round_robin_event_type():
    db = make_session()
    cal = FakeCalClient()
    salon = _onboard_salon_with_two_stylists(db, cal)
    any_event_type_id = salon.services[0].cal_event_type_id

    slots = handle_get_availability(db, cal, "salon-mueller", "Coloration", "2026-07-01", "2026-07-07")

    assert slots["event_type_id"] == any_event_type_id
    event_type = cal.event_types[any_event_type_id]
    assert event_type["scheduling_type"] == "ROUND_ROBIN"
    assert len(event_type["host_user_ids"]) == 2


def test_unqualified_employee_raises_clear_error():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    try:
        handle_get_availability(
            db, cal, "salon-mueller", "Coloration", "2026-07-01", "2026-07-07", employee_name="Nina"
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Coloration" in str(e)
