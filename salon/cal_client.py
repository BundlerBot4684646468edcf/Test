import httpx

from .config import CAL_API_BASE, CAL_API_KEY


class CalComClient:
    """Thin wrapper around the Cal.com v2 API.

    Endpoint shapes follow Cal.com's documented v2 API as of writing; verify
    against your Cal.com plan/API version before going live, since the API
    surface evolves (see https://cal.com/docs/api-reference).
    """

    def __init__(self, api_key: str = CAL_API_KEY, base_url: str = CAL_API_BASE):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        with httpx.Client(timeout=20) as client:
            resp = client.request(method, f"{self.base_url}{path}", headers=self._headers, **kwargs)
            resp.raise_for_status()
            return resp.json()

    def create_team(self, name: str) -> dict:
        return self._request("POST", "/teams", json={"name": name})

    def invite_team_member(self, team_id: str, email: str) -> dict:
        return self._request("POST", f"/teams/{team_id}/memberships", json={"email": email})

    def create_event_type(
        self, team_id: str, title: str, slug: str, length_min: int, host_user_ids: list[str]
    ) -> dict:
        payload = {
            "title": title,
            "slug": slug,
            "lengthInMinutes": length_min,
            "teamId": team_id,
            "hosts": [{"userId": uid} for uid in host_user_ids],
        }
        return self._request("POST", "/event-types", json=payload)

    def get_slots(self, event_type_id: str, date_from: str, date_to: str, timezone: str) -> dict:
        params = {"eventTypeId": event_type_id, "start": date_from, "end": date_to, "timeZone": timezone}
        return self._request("GET", "/slots", params=params)

    def create_booking(
        self, event_type_id: str, start_at: str, attendee_name: str, attendee_email: str, timezone: str
    ) -> dict:
        payload = {
            "eventTypeId": event_type_id,
            "start": start_at,
            "attendee": {"name": attendee_name, "email": attendee_email, "timeZone": timezone},
        }
        return self._request("POST", "/bookings", json=payload)
