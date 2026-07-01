import httpx

from .config import CAL_API_BASE, CAL_API_KEY

CAL_API_VERSION = "2026-02-25"


class CalComClient:
    """Thin wrapper around the Cal.com v2 API.

    All responses are unwrapped from Cal.com's standard envelope:
      {"status": "success", "data": {...}}  →  returns the inner dict directly.
    """

    def __init__(self, api_key: str = CAL_API_KEY, base_url: str = CAL_API_BASE):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "cal-api-version": CAL_API_VERSION,
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        with httpx.Client(timeout=20) as client:
            resp = client.request(method, f"{self.base_url}{path}", headers=self._headers, **kwargs)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Cal.com {method} {path} returned {exc.response.status_code}: {exc.response.text}"
                ) from exc
            payload = resp.json()
            # Unwrap Cal.com v2 envelope: {"status": "success", "data": {...}}
            if isinstance(payload, dict) and "data" in payload:
                return payload["data"]
            return payload

    def create_team(self, name: str) -> dict:
        return self._request("POST", "/teams", json={"name": name})

    def invite_team_member(self, team_id: str, email: str) -> dict:
        return self._request(
            "POST",
            f"/teams/{team_id}/memberships",
            json={"email": email, "role": "MEMBER"},
        )

    def create_event_type(
        self,
        team_id: str,
        title: str,
        slug: str,
        length_min: int,
        host_user_ids: list[str],
        scheduling_type: str | None = None,
    ) -> dict:
        """Create a team event type.

        scheduling_type="ROUND_ROBIN" with multiple hosts lets Cal.com pick
        any available host automatically ("egal wer"). Omit for a single
        fixed-host event type. userId must be int for the Cal.com API.
        """
        payload = {
            "title": title,
            "slug": slug,
            "lengthInMinutes": length_min,
            "hosts": [
                {"userId": int(uid), "isFixed": scheduling_type != "ROUND_ROBIN"}
                for uid in host_user_ids
                if uid
            ],
        }
        if scheduling_type:
            payload["schedulingType"] = scheduling_type
        # Correct v2 endpoint: /teams/{teamId}/event-types  (not /event-types with teamId in body)
        return self._request("POST", f"/teams/{team_id}/event-types", json=payload)

    def get_slots(self, event_type_id: str, date_from: str, date_to: str, timezone: str) -> dict:
        params = {
            "eventTypeId": event_type_id,
            "start": date_from,
            "end": date_to,
            "timeZone": timezone,
        }
        return self._request("GET", "/slots", params=params)

    def create_booking(
        self, event_type_id: str, start_at: str, attendee_name: str, attendee_email: str, timezone: str
    ) -> dict:
        payload = {
            "eventTypeId": int(event_type_id),
            "start": start_at,
            "attendee": {"name": attendee_name, "email": attendee_email, "timeZone": timezone},
        }
        return self._request("POST", "/bookings", json=payload)
