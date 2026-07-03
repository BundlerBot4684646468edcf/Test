"""Dashboard login: password check, session cookies, per-salon isolation,
and the optional API keys for the Famulor endpoints and salon creation."""

from salon.auth import COOKIE_NAME, hash_password, verify_password
from tests.test_api_e2e import TEST_SALON_PAYLOAD, login, make_client


def test_password_hash_roundtrip():
    stored = hash_password("geheim123")
    assert verify_password("geheim123", stored)
    assert not verify_password("falsch", stored)
    assert not verify_password("geheim123", None)


def test_dashboard_pages_redirect_to_login_without_session():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    for page in ["admin", "calendar"]:
        r = client.get(f"/salons/schoenheitssalon-test/{page}", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"].startswith("/salons/schoenheitssalon-test/login")

    r = client.get("/salons/schoenheitssalon-test/login")
    assert r.status_code == 200
    assert "Anmeldung" in r.text


def test_dashboard_api_returns_401_without_session():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    assert client.get("/salons/schoenheitssalon-test/config").status_code == 401
    assert client.get(
        "/salons/schoenheitssalon-test/bookings?date_from=2030-01-01&date_to=2030-01-07"
    ).status_code == 401
    assert client.put(
        "/salons/schoenheitssalon-test/opening-hours", json={"opening_hours": []}
    ).status_code == 401


def test_wrong_password_rejected_then_correct_login_works():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)

    r = client.post("/salons/schoenheitssalon-test/login", json={"password": "falsch"})
    assert r.status_code == 401

    login(client)
    r = client.get("/salons/schoenheitssalon-test/config")
    assert r.status_code == 200
    r = client.get("/salons/schoenheitssalon-test/admin")
    assert r.status_code == 200
    assert "Verwaltung" in r.text


def test_generated_password_is_returned_once_and_works():
    client = make_client()
    payload = dict(TEST_SALON_PAYLOAD)
    payload.pop("admin_password")

    r = client.post("/salons", json=payload)
    assert r.status_code == 200, r.text
    generated = r.json()["admin_password"]
    assert len(generated) >= 8

    login(client, password=generated)
    assert client.get("/salons/schoenheitssalon-test/config").status_code == 200


def test_session_of_one_salon_does_not_open_another():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    second = dict(TEST_SALON_PAYLOAD)
    second.update({"name": "Salon Zwei", "slug": "salon-zwei", "admin_password": "anderes"})
    client.post("/salons", json=second)

    login(client)  # session for schoenheitssalon-test
    token = client.cookies.get(COOKIE_NAME)
    assert token

    r = client.get("/salons/salon-zwei/config", headers={"Cookie": f"{COOKIE_NAME}={token}"})
    assert r.status_code == 401


def test_logout_invalidates_the_browser_session():
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    login(client)
    assert client.get("/salons/schoenheitssalon-test/config").status_code == 200

    client.get("/salons/schoenheitssalon-test/logout", follow_redirects=False)
    assert client.get("/salons/schoenheitssalon-test/config").status_code == 401


def test_famulor_api_key_enforced_when_configured(monkeypatch):
    client = make_client()
    client.post("/salons", json=TEST_SALON_PAYLOAD)
    monkeypatch.setenv("FAMULOR_API_KEY", "bot-key-1")

    body = {"salon_slug": "schoenheitssalon-test"}
    r = client.post("/famulor/tools/get_current_datetime", json=body)
    assert r.status_code == 401

    r = client.post(
        "/famulor/tools/get_current_datetime", json=body, headers={"X-API-Key": "bot-key-1"}
    )
    assert r.status_code == 200


def test_platform_admin_key_guards_salon_creation(monkeypatch):
    client = make_client()
    monkeypatch.setenv("PLATFORM_ADMIN_KEY", "platform-key-1")

    r = client.post("/salons", json=TEST_SALON_PAYLOAD)
    assert r.status_code == 401

    r = client.post("/salons", json=TEST_SALON_PAYLOAD, headers={"X-Admin-Key": "platform-key-1"})
    assert r.status_code == 200
