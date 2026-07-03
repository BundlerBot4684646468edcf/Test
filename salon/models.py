from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .db import Base


class Salon(Base):
    __tablename__ = "salons"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    timezone = Column(String, default="Europe/Berlin")
    # PBKDF2 hash (salon/auth.py); guards the admin/calendar dashboard
    admin_password_hash = Column(String, nullable=True)

    employees = relationship("Employee", back_populates="salon")
    services = relationship("Service", back_populates="salon")
    opening_hours = relationship("OpeningHours", back_populates="salon")


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    salon_id = Column(Integer, ForeignKey("salons.id"), nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    active = Column(Boolean, default=True)

    salon = relationship("Salon", back_populates="employees")


class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True)
    salon_id = Column(Integer, ForeignKey("salons.id"), nullable=False)
    name = Column(String, nullable=False)
    duration_min = Column(Integer, nullable=False)
    buffer_min = Column(Integer, default=0)  # cleanup time after the appointment
    price_cents = Column(Integer, default=0)

    salon = relationship("Salon", back_populates="services")


class EmployeeService(Base):
    """Qualifies one employee to perform one service, optionally with their
    own duration (NULL = inherits the service default)."""

    __tablename__ = "employee_services"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    duration_min = Column(Integer, nullable=True)

    employee = relationship("Employee")
    service = relationship("Service")

    @property
    def effective_duration_min(self) -> int:
        return self.duration_min or self.service.duration_min


class OpeningHours(Base):
    """Weekly opening hours of a salon; slots are only offered inside them.
    One row per weekday (0 = Monday .. 6 = Sunday)."""

    __tablename__ = "opening_hours"

    id = Column(Integer, primary_key=True)
    salon_id = Column(Integer, ForeignKey("salons.id"), nullable=False)
    weekday = Column(Integer, nullable=False)
    open_time = Column(String, default="09:00")   # "HH:MM" local salon time
    close_time = Column(String, default="18:00")
    closed = Column(Boolean, default=False)

    salon = relationship("Salon", back_populates="opening_hours")


class Booking(Base):
    """An appointment. All times are naive datetimes in the salon's local
    timezone. buffer_min is snapshotted at booking time so later service
    edits don't change what an existing appointment blocks."""

    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True)
    salon_id = Column(Integer, ForeignKey("salons.id"), nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    customer_name = Column(String, nullable=False)
    customer_email = Column(String, nullable=False)
    customer_phone = Column(String, nullable=True)  # E.164, for future SMS
    start = Column(DateTime, nullable=False)
    end = Column(DateTime, nullable=False)
    buffer_min = Column(Integer, default=0)
    status = Column(String, default="confirmed")  # confirmed | cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee")
    service = relationship("Service")
