"""Core guarantees of the own booking engine: double-booking protection,
buffers, opening hours, round-robin load balancing."""

import pytest

from salon import booking, schemas
from salon.booking import SlotUnavailableError
from salon.famulor_tools import handle_book_appointment, handle_get_availability
from salon.onboarding import onboard_salon
from tests.test_salon_onboarding import (
    BOOK_DAY,
    DATE_FROM,
    DATE_TO,
    START_AT,
    _onboard_salon_with_two_stylists,
    make_session,
)


def book(db, start_at, customer="Max Mustermann", employee=None, service="Coloration"):
    return handle_book_appointment(
        db, "salon-mueller", service, start_at,
        customer, "max@example.com", employee_name=employee,
    )


def test_double_booking_same_employee_is_rejected():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    book(db, START_AT, employee="Lena")
    with pytest.raises(SlotUnavailableError) as exc:
        book(db, START_AT, customer="Zweiter Kunde", employee="Lena")
    assert "vergeben" in str(exc.value)

    # overlapping (not identical) times are also rejected: Lena 09:00-09:45
    with pytest.raises(SlotUnavailableError):
        book(db, f"{DATE_FROM}T09:30:00", customer="Dritter Kunde", employee="Lena")


def test_round_robin_falls_back_to_free_employee_then_rejects():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    first = book(db, START_AT)
    second = book(db, START_AT, customer="Kunde Zwei")
    assert {first["employee"], second["employee"]} == {"Lena", "Tom"}

    with pytest.raises(SlotUnavailableError):
        book(db, START_AT, customer="Kunde Drei")


def test_round_robin_load_balances_across_the_day():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    first = book(db, START_AT)  # tie -> first employee (Lena)
    assert first["employee"] == "Lena"
    # Lena now has 1 booking that day, Tom 0 -> a later free slot goes to Tom
    second = book(db, f"{DATE_FROM}T14:00:00", customer="Kunde Zwei")
    assert second["employee"] == "Tom"


def test_named_booking_blocks_the_round_robin_slot_too():
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    book(db, START_AT, employee="Lena")
    slots = handle_get_availability(db, "salon-mueller", "Coloration", DATE_FROM, DATE_FROM)
    assert f"{DATE_FROM}T09:00" in slots["slots"]  # Tom still free

    book(db, START_AT, employee="Tom")
    slots = handle_get_availability(db, "salon-mueller", "Coloration", DATE_FROM, DATE_FROM)
    assert f"{DATE_FROM}T09:00" not in slots["slots"]  # nobody left


def test_buffer_blocks_directly_adjacent_slot():
    db = make_session()
    payload = schemas.SalonOnboardIn(
        name="Salon B", slug="salon-mueller",
        employees=[schemas.EmployeeIn(name="Lena", email="lena@example.com")],
        services=[schemas.ServiceIn(name="Coloration", duration_min=45, buffer_min=15)],
    )
    onboard_salon(db, payload)

    book(db, START_AT, employee="Lena")  # 09:00-09:45 + 15 min buffer -> blocked till 10:00

    slots = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_FROM, employee_name="Lena"
    )
    assert f"{DATE_FROM}T09:45" not in slots["slots"]
    assert f"{DATE_FROM}T10:00" in slots["slots"]

    with pytest.raises(SlotUnavailableError):
        book(db, f"{DATE_FROM}T09:45:00", customer="Kunde Zwei", employee="Lena")


def test_no_slots_on_closed_day_and_booking_rejected():
    db = make_session()
    _onboard_salon_with_two_stylists(db)
    sunday = BOOK_DAY.fromordinal(BOOK_DAY.toordinal() + 6)  # BOOK_DAY is a Monday

    slots = handle_get_availability(
        db, "salon-mueller", "Coloration", sunday.isoformat(), sunday.isoformat()
    )
    assert slots["slots"] == []

    with pytest.raises(SlotUnavailableError) as exc:
        book(db, f"{sunday.isoformat()}T10:00:00")
    assert "geschlossen" in str(exc.value)


def test_slots_respect_opening_and_closing_time():
    db = make_session()
    salon = _onboard_salon_with_two_stylists(db)

    saturday = BOOK_DAY.fromordinal(BOOK_DAY.toordinal() + 5)  # Sat: 09:00-14:00
    slots = handle_get_availability(
        db, "salon-mueller", "Coloration", saturday.isoformat(), saturday.isoformat(),
        employee_name="Tom",  # 90 minutes
    )
    assert slots["slots"][0] == f"{saturday.isoformat()}T09:00"
    # last possible start so that 90 min still fit before 14:00 is 12:30
    assert slots["slots"][-1] == f"{saturday.isoformat()}T12:30"

    with pytest.raises(SlotUnavailableError) as exc:
        book(db, f"{saturday.isoformat()}T13:00:00", employee="Tom")
    assert "Öffnungszeiten" in str(exc.value)


def test_booking_outside_opening_hours_rejected():
    db = make_session()
    _onboard_salon_with_two_stylists(db)
    with pytest.raises(SlotUnavailableError):
        book(db, f"{DATE_FROM}T07:00:00")


def test_slot_taken_between_availability_and_booking_is_rejected():
    """The classic race: bot reads slots, someone books it meanwhile, bot books."""
    db = make_session()
    _onboard_salon_with_two_stylists(db)

    slots = handle_get_availability(
        db, "salon-mueller", "Coloration", DATE_FROM, DATE_FROM, employee_name="Lena"
    )
    target = slots["slots"][0]

    book(db, target, customer="Schnellerer Kunde", employee="Lena")  # someone else

    with pytest.raises(SlotUnavailableError):
        book(db, target, employee="Lena")
