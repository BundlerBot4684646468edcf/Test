---
tags: [decision]
status: accepted
date: 2026-07-10
---

# Google Places as first review source

## Context
The MVP needs guest reviews. Candidates: Google Places, Booking.com, TripAdvisor, manual CSV import.

## Decision
Start with **Google Places only**.

## Rationale
- Official API, straightforward key setup, no scraping risk.
- Covers nearly every hotel and includes DE/IT/EN reviews.
- Fastest path to "hotel name → insights" for pilots.

## Consequences
- Limited review depth per hotel (Places API caps returned reviews) — may understate trends.
- Revisit once pilots ask for Booking/TripAdvisor coverage → tracked on the [[03-Product/Product Overview|roadmap]].
