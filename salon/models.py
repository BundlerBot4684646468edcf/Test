from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .db import Base


class Salon(Base):
    __tablename__ = "salons"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    timezone = Column(String, default="Europe/Berlin")
    cal_team_id = Column(String, nullable=True)

    employees = relationship("Employee", back_populates="salon")
    services = relationship("Service", back_populates="salon")


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    salon_id = Column(Integer, ForeignKey("salons.id"), nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    cal_user_id = Column(String, nullable=True)
    active = Column(Boolean, default=True)

    salon = relationship("Salon", back_populates="employees")


class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True)
    salon_id = Column(Integer, ForeignKey("salons.id"), nullable=False)
    name = Column(String, nullable=False)
    duration_min = Column(Integer, nullable=False)
    buffer_min = Column(Integer, default=0)
    price_cents = Column(Integer, default=0)
    # Round-Robin team event type used when the customer has no employee preference.
    cal_event_type_id = Column(String, nullable=True)

    salon = relationship("Salon", back_populates="services")


class EmployeeService(Base):
    """Qualifies one employee to perform one service, with its own duration
    and a dedicated fixed-host Cal.com event type for direct booking."""

    __tablename__ = "employee_services"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    # NULL = inherits the service default; changing the service duration then
    # automatically re-syncs this employee's event type too.
    duration_min = Column(Integer, nullable=True)
    cal_event_type_id = Column(String, nullable=True)

    employee = relationship("Employee")
    service = relationship("Service")

    @property
    def effective_duration_min(self) -> int:
        return self.duration_min or self.service.duration_min
