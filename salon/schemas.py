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
    # dashboard login; auto-generated and returned once if omitted
    admin_password: Optional[str] = None
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


class OpeningHoursIn(BaseModel):
    weekday: int  # 0 = Montag .. 6 = Sonntag
    open_time: str = "09:00"
    close_time: str = "18:00"
    closed: bool = False


class OpeningHoursUpdate(BaseModel):
    opening_hours: list[OpeningHoursIn]


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


class LoginIn(BaseModel):
    password: str


class CancelIn(BaseModel):
    salon_slug: str
    start_at: str
    customer_phone: Optional[str] = None
    customer_name: Optional[str] = None
