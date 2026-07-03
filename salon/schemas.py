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


class EmployeeServiceIn(BaseModel):
    employee_name: str
    service_name: str
    duration_min: Optional[int] = None  # overrides the service default for this employee


class SalonOnboardIn(BaseModel):
    name: str
    slug: str
    timezone: str = "Europe/Berlin"
    employees: list[EmployeeIn]
    services: list[ServiceIn]
    # who can perform what, and how long it takes them; if omitted, every
    # employee is assumed to offer every service at the service's default duration
    qualifications: list[EmployeeServiceIn] = []


class ServicePatch(BaseModel):
    name: Optional[str] = None
    duration_min: Optional[int] = None
    buffer_min: Optional[int] = None
    price_cents: Optional[int] = None


class QualificationsIn(BaseModel):
    # full desired matrix; the backend reconciles additions/removals/changes
    qualifications: list[EmployeeServiceIn]
    force: bool = False


class CurrentDatetimeQuery(BaseModel):
    salon_slug: str


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
