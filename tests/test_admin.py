"""Tests for the admin layer: editing services/employees/qualifications after
onboarding must keep the Cal.com event types (durations, buffers, hosts)
consistent and must not orphan upcoming bookings."""

import pytest

from salon import schemas
from salon.admin import (
    UpcomingBookingsError,
    add_employee,
    add_service,
    delete_service,
    get_salon_config,
    remove_employee,
    set_qualifications,
    update_service,
)
from salon.famulor_tools import handle_book_appointment
from tests.test_salon_onboarding import (
    DATE_FROM,
    START_AT,
    FakeCalClient,
    _onboard_salon_with_two_stylists,
    make_session,
)


def q(employee, service, duration=None):
    return schemas.EmployeeServiceIn(
        employee_name=employee, service_name=service, duration_min=duration
    )


def setup():
    db = make_session()
    cal = FakeCalClient()
    salon = _onboard_salon_with_two_stylists(db, cal)
    return db, cal, salon


def _rr_event_type(cal, salon, service_name="Coloration"):
    service = next(s for s in salon.services if s.name == service_name)
    return service.cal_event_type_id, cal.event_types.get(service.cal_event_type_id)


def test_config_lists_everything_with_effective_durations():
    db, cal, salon = setup()
    config = get_salon_config(db, salon)

    assert {e["name"] for e in config["employees"]} == {"Lena", "Tom"}
    assert config["services"][0]["bookable_any"] is True
    durations = {qual["employee_name"]: qual["effective_duration_min"] for qual in config["qualifications"]}
    assert durations == {"Lena": 45, "Tom": 90}


def test_update_service_duration_syncs_inheriting_event_types_only():
    db, cal, salon = setup()
    # give Lena an inherited duration (no override) alongside Tom's override
    set_qualifications(db, cal, salon, [q("Lena", "Coloration"), q("Tom", "Coloration", 90)])

    update_service(db, cal, salon, salon.services[0].id, schemas.ServicePatch(duration_min=75))

    _, rr = _rr_event_type(cal, salon)
    assert rr["length_min"] == 75  # round-robin follows the service default
    lengths = sorted(
        et["length_min"] for et in cal.event_types.values() if et["scheduling_type"] is None
    )
    assert lengths == [75, 90]  # Lena inherits the new default, Tom keeps his override


def test_update_service_buffer_reaches_cal():
    db, cal, salon = setup()
    update_service(db, cal, salon, salon.services[0].id, schemas.ServicePatch(buffer_min=15))
    assert all(et["buffer_min"] == 15 for et in cal.event_types.values())


def test_onboarding_passes_buffer_to_cal():
    db = make_session()
    cal = FakeCalClient()
    payload = schemas.SalonOnboardIn(
        name="Salon B", slug="salon-b",
        employees=[schemas.EmployeeIn(name="Mia", email="mia@example.com")],
        services=[schemas.ServiceIn(name="Coloration", duration_min=60, buffer_min=10)],
    )
    from salon.onboarding import onboard_salon
    onboard_salon(db, payload, cal)
    assert all(et["buffer_min"] == 10 for et in cal.event_types.values())


def test_matrix_add_and_remove_qualification_syncs_round_robin_hosts():
    db, cal, salon = setup()
    add_service(db, cal, salon, schemas.ServiceIn(name="Haarschnitt", duration_min=30))
    service = next(s for s in salon.services if s.name == "Haarschnitt")
    assert service.cal_event_type_id is None  # nobody assigned yet -> not bookable

    config = set_qualifications(db, cal, salon, [
        q("Lena", "Coloration", 45), q("Tom", "Coloration", 90),
        q("Lena", "Haarschnitt"),
    ])
    assert next(s for s in config["services"] if s["name"] == "Haarschnitt")["bookable_any"] is True
    rr = cal.event_types[service.cal_event_type_id]
    assert rr["scheduling_type"] == "ROUND_ROBIN"
    assert len(rr["host_user_ids"]) == 1

    # Tom learns Haarschnitt too -> becomes a round-robin host
    set_qualifications(db, cal, salon, [
        q("Lena", "Coloration", 45), q("Tom", "Coloration", 90),
        q("Lena", "Haarschnitt"), q("Tom", "Haarschnitt", 40),
    ])
    assert len(cal.event_types[service.cal_event_type_id]["host_user_ids"]) == 2

    # Lena stops offering it -> host removed, her event type deleted
    set_qualifications(db, cal, salon, [
        q("Lena", "Coloration", 45), q("Tom", "Coloration", 90),
        q("Tom", "Haarschnitt", 40),
    ])
    assert len(cal.event_types[service.cal_event_type_id]["host_user_ids"]) == 1
    assert len(cal.deleted_event_types) == 1


def test_matrix_duration_change_patches_event_type():
    db, cal, salon = setup()
    config = set_qualifications(db, cal, salon, [
        q("Lena", "Coloration", 50), q("Tom", "Coloration", 90),
    ])
    lena = next(x for x in config["qualifications"] if x["employee_name"] == "Lena")
    assert lena["effective_duration_min"] == 50
    lena_link_et = next(
        et for et in cal.event_types.values()
        if et["scheduling_type"] is None and et["length_min"] == 50
    )
    assert lena_link_et["title"] == "Coloration mit Lena"


def test_remove_last_employee_of_service_removes_round_robin():
    db, cal, salon = setup()
    set_qualifications(db, cal, salon, [q("Tom", "Coloration", 90)])
    config = set_qualifications(db, cal, salon, [])

    assert config["services"][0]["bookable_any"] is False
    assert all(et["scheduling_type"] != "ROUND_ROBIN" for et in cal.event_types.values())


def test_remove_employee_updates_round_robin_and_deletes_event_types():
    db, cal, salon = setup()
    lena = next(e for e in salon.employees if e.name == "Lena")

    config = remove_employee(db, cal, salon, lena.id)

    assert {e["name"] for e in config["employees"]} == {"Tom"}
    rr_id, rr = _rr_event_type(cal, salon)
    assert rr["host_user_ids"] == [next(e for e in salon.employees if e.name == "Tom").cal_user_id]
    assert config["qualifications"][0]["employee_name"] == "Tom"


def test_remove_employee_with_upcoming_booking_requires_force():
    db, cal, salon = setup()
    handle_book_appointment(
        db, cal, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com", employee_name="Lena",
    )
    lena = next(e for e in salon.employees if e.name == "Lena")

    with pytest.raises(UpcomingBookingsError):
        remove_employee(db, cal, salon, lena.id)
    assert lena.active  # nothing was changed

    config = remove_employee(db, cal, salon, lena.id, force=True)
    assert {e["name"] for e in config["employees"]} == {"Tom"}


def test_delete_service_with_upcoming_booking_requires_force():
    db, cal, salon = setup()
    handle_book_appointment(
        db, cal, "salon-mueller", "Coloration", START_AT,
        "Max Mustermann", "max@example.com",
    )
    service_id = salon.services[0].id

    with pytest.raises(UpcomingBookingsError):
        delete_service(db, cal, salon, service_id)

    config = delete_service(db, cal, salon, service_id, force=True)
    assert config["services"] == []
    assert config["qualifications"] == []


def test_add_employee_invites_to_cal_team_and_can_be_assigned():
    db, cal, salon = setup()
    add_employee(db, cal, salon, schemas.EmployeeIn(name="Nina", email="nina@example.com"))

    config = set_qualifications(db, cal, salon, [
        q("Lena", "Coloration", 45), q("Tom", "Coloration", 90), q("Nina", "Coloration"),
    ])
    assert len(config["qualifications"]) == 3
    _, rr = _rr_event_type(cal, salon)
    assert len(rr["host_user_ids"]) == 3
    nina = next(x for x in config["qualifications"] if x["employee_name"] == "Nina")
    assert nina["effective_duration_min"] == 60  # inherits the service default
