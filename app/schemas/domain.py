from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum

from pydantic import BaseModel, Field


class ActorType(str, Enum):
    PATIENT = "patient"
    ATTENDANT = "attendant"
    DOCTOR = "doctor"
    IT_ADMIN = "it_admin"


class AppointmentStatus(str, Enum):
    AVAILABLE = "available"
    HELD = "held"
    BOOKED = "booked"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class RecordEntryType(str, Enum):
    VISIT_NOTE = "visit_note"
    HISTORY_SUMMARY = "history_summary"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    PROCEDURE = "procedure"
    ADMIN_NOTE = "admin_note"
    UPLOADED_DOCUMENT = "uploaded_document"


class SourceReference(BaseModel):
    source_type: str
    source_path: str


class Patient(BaseModel):
    patient_id: str
    full_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    phone: str | None = None
    address: str | None = None
    primary_conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    preferred_doctors: list[str] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)


class DoctorAvailabilityWindow(BaseModel):
    days_of_week: list[int] = Field(default_factory=list)
    start_time: time
    end_time: time
    source_text: str | None = None


class Doctor(BaseModel):
    doctor_id: str
    full_name: str
    specialty: str
    clinic_name: str
    location_id: str
    phone: str | None = None
    email: str | None = None
    gender: str | None = None
    active: bool = True
    availability: list[DoctorAvailabilityWindow] = Field(default_factory=list)


class Appointment(BaseModel):
    appointment_id: str
    patient_id: str
    doctor_id: str
    specialty: str
    status: AppointmentStatus
    appointment_date: date
    appointment_time: time
    location_id: str
    booked_by_actor: ActorType
    booking_reason: str
    created_at: datetime


class MedicalRecordEntry(BaseModel):
    record_id: str
    patient_id: str
    entry_type: RecordEntryType
    visit_date: date | None = None
    title: str
    subjective: str | None = None
    objective: str | None = None
    assessment: str | None = None
    plan: str | None = None
    doctor_id: str | None = None
    source_type: str
    source_path: str
    structured_fields: dict[str, object] = Field(default_factory=dict)


class Diagnosis(BaseModel):
    diagnosis_id: str
    patient_id: str
    record_id: str
    name: str
    code_system: str | None = None
    code: str | None = None
    status: str
    diagnosed_on: date | None = None


class PatientMemory(BaseModel):
    memory_id: str
    patient_id: str
    summary_type: str
    summary_text: str
    derived_from_record_ids: list[str] = Field(default_factory=list)
    key_facts: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class RetrievedEvidence(BaseModel):
    evidence_id: str
    source_group: str
    source_type: str
    source_label: str
    source_uri: str
    patient_id: str | None = None
    chunk_id: str | None = None
    text: str
    score: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
