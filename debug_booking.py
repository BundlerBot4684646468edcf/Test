#!/usr/bin/env python3
"""
Local booking debug tester.
Run this to simulate the full booking flow and see detailed error messages.

Usage:
  python debug_booking.py
"""

import os
import sys
from pprint import pprint

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from salon.cal_client import CalComClient
from salon.config import CAL_API_KEY, CAL_API_BASE

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    # Check API key
    print_section("1. Checking CAL_API_KEY")
    if not CAL_API_KEY:
        print("❌ CAL_API_KEY is NOT set!")
        print("   Set it with: export CAL_API_KEY='your_key_here'")
        return False
    print(f"✅ CAL_API_KEY is set (first 10 chars: {CAL_API_KEY[:10]}...)")
    print(f"   Base URL: {CAL_API_BASE}")

    # Test connectivity
    print_section("2. Testing Cal.com API connectivity (GET /me)")
    cal = CalComClient()
    try:
        me = cal._request("GET", "/me")
        print("✅ Connected to Cal.com")
        print(f"   Authenticated as: {me}")
    except RuntimeError as e:
        print(f"❌ Cal.com connection failed:")
        print(f"   {e}")
        return False

    # Test team creation
    print_section("3. Testing team creation (POST /teams)")
    try:
        team = cal.create_team("Debug Test Salon")
        print("✅ Team created")
        print(f"   Team ID: {team.get('id')}")
        team_id = str(team.get("id"))
    except RuntimeError as e:
        print(f"❌ Team creation failed:")
        print(f"   {e}")
        return False

    # Test member invite
    print_section("4. Testing member invite (POST /teams/{teamId}/memberships)")
    try:
        membership = cal.invite_team_member(team_id, "test-employee@example.com")
        print("✅ Member invited")
        print(f"   User ID: {membership.get('userId')}")
        user_id = str(membership.get("userId"))
    except RuntimeError as e:
        print(f"❌ Member invite failed:")
        print(f"   {e}")
        return False

    # Test event type creation (fixed-host)
    print_section("5. Testing event-type creation (POST /teams/{teamId}/event-types)")
    try:
        event_type = cal.create_event_type(
            team_id,
            title="Test Service",
            slug="test-service",
            length_min=60,
            host_user_ids=[user_id],
            scheduling_type=None,  # fixed-host, not round-robin
        )
        print("✅ Event type created")
        print(f"   Event Type ID: {event_type.get('id')}")
        event_type_id = str(event_type.get("id"))
    except RuntimeError as e:
        print(f"❌ Event type creation failed:")
        print(f"   {e}")
        return False

    # Test get slots
    print_section("6. Testing get slots (GET /slots)")
    try:
        slots = cal.get_slots(event_type_id, "2026-07-01", "2026-07-07", "Europe/Berlin")
        print("✅ Slots retrieved")
        pprint(slots)
    except RuntimeError as e:
        print(f"❌ Get slots failed:")
        print(f"   {e}")
        return False

    # Test booking creation
    print_section("7. Testing booking creation (POST /bookings)")
    try:
        booking = cal.create_booking(
            event_type_id,
            "2026-07-01T09:00:00",
            "Test Customer",
            "test-customer@example.com",
            "Europe/Berlin",
        )
        print("✅ Booking created")
        pprint(booking)
    except RuntimeError as e:
        print(f"❌ Booking creation failed:")
        print(f"   {e}")
        return False

    print_section("✅ ALL TESTS PASSED")
    print("Your backend is ready to receive bookings from Famulor!\n")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
