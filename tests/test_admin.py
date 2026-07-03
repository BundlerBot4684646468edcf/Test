"""Tests for the admin layer: editing services/employees/qualifications and
opening hours takes effect in the booking engine immediately and never
silently orphans upcoming bookings."""

import pytest

from salon import models, schemas
from salon.admin import (
    UpcomingBookingsError,
    add_employee,
    add_service,
    delete_service,
    get_salon_config,
    remove_employee,
    set_opening_hours,
    set_qualifications,
    update_service,
)
from salon.famulor_tools import handle_book_appointment, handle_get_availability
from tests.test_salon_onboarding import (
    DATE_FROM,
    DATE_TO,
    START_AT,
    _onboard_salon_with_two_stylists,
    make_session,
)


def q(employee, service, duration=None):
    return schemas.EmployeeServiceIn(
        employee_name=employee, service_name=service, duration_min=duration
    )


def setup():
    db = make_session()
    salon = _onboard_salon_with_two_stylists(db)
    return db, salon


def test_config_lists_everything():
    db, salon = setup()
    config = get_salon_config(db, salon)

    assert {e["name"] for e in config["employees"]} == {"Lena", "Tom"}
    assert config["services"][0]["bookable_any"] is True
    durations = {x["employee_name"]: x["effective_duration_min"] for x in config["qualifications"]}
    assert durations == {"Lena": 45, "Tom": 90}
    assert len(config["opening_hours"]) == 7


def test_update_service_duration_applies_to_inheriting_employees_only():
    db, salon = setup()
    set_qualifications(db, salon, [q("Lena", "Coloration"), q("Tom", "Coloration", 90)])

    update_service(db, salon, salon.services[0].id, schemas.ServicePatch(duration_min=75))

    lena = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Lena"
    )
    tom = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Tom"
    )
    assert lena["duration_min"] == 75  # inherits the new default
    assert tom["duration_min"] == 90  # keeps his override


def test_new_service_needs_assignment_before_it_is_bookable():
    db, salon = setup()
    config = add_service(db, salon, schemas.ServiceIn(name="Haarschnitt", duration_min=30))
    assert next(s for s in config["services"] if s["name"] == "Haarschnitt")["bookable_any"] is False

    with pytest.raises(ValueError):
        handle_get_availability(db, "salon-mueller", "Haarschnitt", DATE_FROM, DATE_TO)

    config = set_qualifications(db, salon, [
        q("Lena", "Coloration", 45), q("Tom", "Coloration", 90), q("Lena", "Haarschnitt"),
    ])
    assert next(s for s in config["services"] if s["name"] == "Haarschnitt")["bookable_any"] is True
    slots = handle_get_availability(db, "salon-mueller", "Haarschnitt", DATE_FROM, DATE_TO)
    assert slots["slots"]


def test_matrix_removal_with_upcoming_booking_requires_force_and_cancels():
    db, salon = setup()
    handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", employee_name="Lena",
    )

    with pytest.raises(UpcomingBookingsError):
        set_qualifications(db, salon, [q("Tom", "Coloration", 90)])

    config = set_qualifications(db, salon, [q("Tom", "Coloration", 90)], force=True)
    assert [x["employee_name"] for x in config["qualifications"]] == ["Tom"]
    booking = db.query(models.Booking).one()
    assert booking.status == "cancelled"  # explicitly cancelled, not orphaned


def test_remove_employee_soft_deletes_and_keeps_past_attribution():
    db, salon = setup()
    lena = next(e for e in salon.employees if e.name == "Lena")

    config = remove_employee(db, salon, lena.id)
    assert {e["name"] for e in config["employees"]} == {"Tom"}
    assert lena.active is False

    # Lena can no longer be booked by name
    with pytest.raises(ValueError):
        handle_get_availability(
            db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO, employee_name="Lena"
        )
    # "egal wer" still works via Tom
    slots = handle_get_availability(db, "salon-mueller", "Coloration", DATE_FROM, DATE_TO)
    assert slots["slots"]


def test_remove_employee_with_upcoming_booking_requires_force():
    db, salon = setup()
    handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", employee_name="Lena",
    )
    lena = next(e for e in salon.employees if e.name == "Lena")

    with pytest.raises(UpcomingBookingsError):
        remove_employee(db, salon, lena.id)
    assert lena.active

    remove_employee(db, salon, lena.id, force=True)
    assert db.query(models.Booking).one().status == "cancelled"


def test_delete_service_with_upcoming_booking_requires_force():
    db, salon = setup()
    handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com",
    )
    service_id = salon.services[0].id

    with pytest.raises(UpcomingBookingsError):
        delete_service(db, salon, service_id)

    config = delete_service(db, salon, service_id, force=True)
    assert config["services"] == []
    assert config["qualifications"] == []


def test_add_employee_can_be_assigned_and_booked():
    db, salon = setup()
    add_employee(db, salon, schemas.EmployeeIn(name="Nina", email="nina@example.com"))
    config = set_qualifications(db, salon, [
        q("Lena", "Coloration", 45), q("Tom", "Coloration", 90), q("Nina", "Coloration"),
    ])
    nina = next(x for x in config["qualifications"] if x["employee_name"] == "Nina")
    assert nina["effective_duration_min"] == 60  # inherits the service default

    booking = handle_book_appointment(
        db, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", employee_name="Nina",
    )
    assert booking["employee"] == "Nina"


def test_opening_hours_update_affects_availability():
    db, salon = setup()
    hours = [
        schemas.OpeningHoursIn(weekday=w, open_time="10:00", close_time="12:00")
        for w in range(6)
    ] + [schemas.OpeningHoursIn(weekday=6, closed=True)]

    config = set_opening_hours(db, salon, hours)
    assert config["opening_hours"][0]["open_time"] == "10:00"

    slots = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_FROM, employee_name="Lena"
    )
    assert slots["slots"][0] == f"{DATE_FROM}T10:00"
    assert slots["slots"][-1] == f"{DATE_FROM}T11:15"  # 45 min must end by 12:00


def test_opening_hours_validation():
    db, salon = setup()
    with pytest.raises(ValueError):
        set_opening_hours(db, salon, [
            schemas.OpeningHoursIn(weekday=0, open_time="18:00", close_time="09:00")
        ])
