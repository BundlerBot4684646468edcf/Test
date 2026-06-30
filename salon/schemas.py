from typing import Optional

from pydantic import BaseModel, EmailStr


class EmployeeIn(BaseModel):
    name: str
    email: EmailStr


class ServiceIn(BaseModel):
    name: str
    duration_min: int
    buffer_min: int = 0
    price_cents: int = 0


class SalonOnboardIn(BaseModel):
    name: str
    slug: str
    timezone: str = "Europe/Berlin"
    employees: list[EmployeeIn]
    services: list[ServiceIn]


class AvailabilityQuery(BaseModel):
    salon_slug: str
    service_name: str
    date_from: str
    date_to: str
    employee_name: Optional[str] = None


class BookingIn(BaseModel):
    salon_slug: str
    service_name: str
    start_at: str
    customer_name: str
    customer_email: EmailStr
    employee_name: Optional[str] = None
    customer_phone: Optional[str] = None
