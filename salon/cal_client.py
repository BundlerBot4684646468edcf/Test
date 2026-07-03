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
            if not resp.content:
                return {}
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
        buffer_min: int = 0,
    ) -> dict:
        """Create a team event type.

        scheduling_type="ROUND_ROBIN" with multiple hosts lets Cal.com pick
        any available host automatically ("egal wer"). Omit for a single
        fixed-host event type. userId must be int for the Cal.com API.
        buffer_min becomes Cal.com's afterEventBuffer, so cleanup time
        between appointments is part of the conflict check.
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
        if buffer_min:
            payload["afterEventBuffer"] = buffer_min
        # Correct v2 endpoint: /teams/{teamId}/event-types  (not /event-types with teamId in body)
        return self._request("POST", f"/teams/{team_id}/event-types", json=payload)

    def update_event_type(self, team_id: str, event_type_id: str, payload: dict) -> dict:
        return self._request(
            "PATCH", f"/teams/{team_id}/event-types/{event_type_id}", json=payload
        )

    def delete_event_type(self, team_id: str, event_type_id: str) -> dict:
        return self._request("DELETE", f"/teams/{team_id}/event-types/{event_type_id}")

    def get_slots(self, event_type_id: str, date_from: str, date_to: str, timezone: str) -> dict:
        params = {
            "eventTypeId": event_type_id,
            "start": date_from,
            "end": date_to,
            "timeZone": timezone,
        }
        return self._request("GET", "/slots", params=params)

    def create_booking(
        self,
        event_type_id: str,
        start_at: str,
        attendee_name: str,
        attendee_email: str,
        timezone: str,
        attendee_phone: str | None = None,
    ) -> dict:
        attendee = {"name": attendee_name, "email": attendee_email, "timeZone": timezone}
        # Cal.com's SMS workflows (confirmation, reminder, review) send to the
        # attendee's phoneNumber — without it no SMS ever goes out.
        if attendee_phone:
            attendee["phoneNumber"] = attendee_phone
        payload = {
            "eventTypeId": int(event_type_id),
            "start": start_at,
            "attendee": attendee,
        }
        return self._request("POST", "/bookings", json=payload)

    def get_bookings(self, event_type_ids: list[str], after_start: str, before_end: str) -> list[dict]:
        """List bookings for the given event types within a time window.
        Cal.com v2 returns the list directly inside the response envelope."""
        params = {
            "eventTypeIds": ",".join(str(i) for i in event_type_ids if i),
            "afterStart": after_start,
            "beforeEnd": before_end,
            "take": 250,
        }
        data = self._request("GET", "/bookings", params=params)
        if isinstance(data, dict):
            return data.get("bookings", [])
        return data
