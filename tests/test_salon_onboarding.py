from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from salon import schemas
from salon.db import Base
from salon.famulor_tools import (
    handle_book_appointment,
    handle_get_availability,
    handle_get_current_datetime,
    normalize_phone,
)
from salon.onboarding import onboard_salon

# Bookings/availability must lie in the future now that past dates are
# rejected, so tests derive their dates from today.
DATE_FROM = (date.today() + timedelta(days=7)).isoformat()
DATE_TO = (date.today() + timedelta(days=13)).isoformat()
START_AT = f"{DATE_FROM}T09:00:00"


class FakeCalClient:
    """Stands in for CalComClient so tests don't hit the real Cal.com API.

    Records created event types keyed by id so tests can assert which
    duration/hosts ended up on which event type.
    """

    def __init__(self):
        self._next_id = 1
        self.event_types: dict[str, dict] = {}
        self.bookings: list[dict] = []
        self.deleted_event_types: list[str] = []

    def _id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def create_team(self, name):
        return {"id": self._id()}

    def invite_team_member(self, team_id, email):
        return {"userId": self._id()}

    def create_event_type(self, team_id, title, slug, length_min, host_user_ids, scheduling_type=None, buffer_min=0):
        event_id = self._id()
        self.event_types[event_id] = {
            "title": title,
            "length_min": length_min,
            "host_user_ids": list(host_user_ids),
            "scheduling_type": scheduling_type,
            "buffer_min": buffer_min,
        }
        return {"id": event_id}

    def update_event_type(self, team_id, event_type_id, payload):
        et = self.event_types[event_type_id]
        if "title" in payload:
            et["title"] = payload["title"]
        if "lengthInMinutes" in payload:
            et["length_min"] = payload["lengthInMinutes"]
        if "afterEventBuffer" in payload:
            et["buffer_min"] = payload["afterEventBuffer"]
        if "hosts" in payload:
            et["host_user_ids"] = [str(h["userId"]) for h in payload["hosts"]]
        return {"id": event_type_id}

    def delete_event_type(self, team_id, event_type_id):
        self.event_types.pop(event_type_id, None)
        self.deleted_event_types.append(event_type_id)
        return {}

    def get_slots(self, event_type_id, date_from, date_to, timezone):
        return {"event_type_id": event_type_id, "slots": [f"{date_from}T09:00:00"]}

    def create_booking(self, event_type_id, start_at, attendee_name, attendee_email, timezone, attendee_phone=None):
        booking = {
            "id": self._id(),
            "event_type_id": event_type_id,
            "status": "ACCEPTED",
            "start": start_at,
            "end": start_at,
            "attendee_phone": attendee_phone,
            "attendees": [{"name": attendee_name, "email": attendee_email}],
            "title": self.event_types.get(event_type_id, {}).get("title", ""),
        }
        self.bookings.append(booking)
        return booking

    def get_bookings(self, event_type_ids, after_start, before_end):
        return [b for b in self.bookings if b["event_type_id"] in set(event_type_ids)]


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

    slots = handle_get_availability(db, cal, "salon-mueller", "Haarschnitt", DATE_FROM, DATE_TO)
    assert slots["slots"] == [f"{DATE_FROM}T09:00:00"]

    booking = handle_book_appointment(
        db, cal, "salon-mueller", "Haarschnitt", START_AT,
        "Max Mustermann", "max@example.com",
    )
    assert booking["status"] == "ACCEPTED"


def test_famulor_tools_unknown_salon_raises():
    db = make_session()
    cal = FakeCalClient()
    try:
        handle_get_availability(db, cal, "does-not-exist", "Haarschnitt", DATE_FROM, DATE_TO)
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
        db, cal, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Lena"
    )
    lena_event_type_id = slots["event_type_id"]
    assert cal.event_types[lena_event_type_id]["length_min"] == 45

    slots = handle_get_availability(
        db, cal, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Tom"
    )
    tom_event_type_id = slots["event_type_id"]
    assert cal.event_types[tom_event_type_id]["length_min"] == 90
    assert tom_event_type_id != lena_event_type_id


def test_no_employee_preference_uses_round_robin_event_type():
    db = make_session()
    cal = FakeCalClient()
    salon = _onboard_salon_with_two_stylists(db, cal)
    any_event_type_id = salon.services[0].cal_event_type_id

    slots = handle_get_availability(db, cal, "salon-mueller", "Coloration", DATE_FROM, DATE_TO)

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
            db, cal, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Nina"
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Coloration" in str(e)


# ----------------- Date awareness ("Samstag war gestern"-Bug) -----------------

def test_get_current_datetime_reports_today_and_weekdays():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    info = handle_get_current_datetime(db, "salon-mueller")

    assert info["today"] == date.today().isoformat()
    assert info["timezone"] == "Europe/Berlin"
    assert len(info["next_days"]) == 7
    assert info["next_days"][0]["date"] == (date.today() + timedelta(days=1)).isoformat()
    # weekday names must be consistent with the date, in German and English
    assert info["today_weekday_de"] in {"Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"}


def test_availability_range_fully_in_past_is_rejected_with_today_hint():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    past_from = (date.today() - timedelta(days=8)).isoformat()
    past_to = (date.today() - timedelta(days=2)).isoformat()
    try:
        handle_get_availability(db, cal, "salon-mueller", "Coloration", past_from, past_to)
        assert False, "expected ValueError"
    except ValueError as e:
        assert date.today().isoformat() in str(e)  # tells the bot what today is


def test_availability_start_in_past_is_clamped_to_today():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    past_from = (date.today() - timedelta(days=2)).isoformat()
    future_to = (date.today() + timedelta(days=5)).isoformat()
    slots = handle_get_availability(db, cal, "salon-mueller", "Coloration", past_from, future_to)

    assert slots["slots"] == [f"{date.today().isoformat()}T09:00:00"]
    assert slots["today"] == date.today().isoformat()


def test_booking_in_the_past_is_rejected():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        handle_book_appointment(
            db, cal, "salon-mueller", "Coloration", f"{yesterday}T09:00:00",
            "Max Mustermann", "max@example.com",
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Vergangenheit" in str(e)
    assert cal.bookings == []


# ----------------- SMS: phone number must reach Cal.com -----------------

def test_booking_passes_normalized_phone_to_cal():
    db = make_session()
    cal = FakeCalClient()
    _onboard_salon_with_two_stylists(db, cal)

    handle_book_appointment(
        db, cal, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", customer_phone="0176 123 45678",
    )

    assert cal.bookings[-1]["attendee_phone"] == "+4917612345678"


def test_phone_normalization_variants():
    assert normalize_phone("+49 176 1234567") == "+491761234567"
    assert normalize_phone("0049 176 1234567") == "+491761234567"
    assert normalize_phone("0176/1234567") == "+491761234567"
    assert normalize_phone(None) is None
    assert normalize_phone("") is None
