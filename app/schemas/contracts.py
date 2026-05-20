from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.domain import ActorType


class WorkflowType(str, Enum):
    SYMPTOM_TO_APPOINTMENT = 'symptom_to_appointment'
    PATIENT_APPOINTMENT_LOOKUP = 'patient_appointment_lookup'
    PATIENT_APPOINTMENT_RESCHEDULE = 'patient_appointment_reschedule'
    PATIENT_APPOINTMENT_CANCEL = 'patient_appointment_cancel'
    PATIENT_HISTORY_SUMMARY = 'patient_history_summary'
    ATTENDANT_HISTORY_SUMMARY = 'attendant_history_summary'
    ATTENDANT_APPOINTMENT_SCHEDULE = 'attendant_appointment_schedule'
    ATTENDANT_APPOINTMENT_RESCHEDULE = 'attendant_appointment_reschedule'
    ATTENDANT_APPOINTMENT_CANCEL = 'attendant_appointment_cancel'
    ATTENDANT_BATCH_APPOINTMENT_RESCHEDULE = 'attendant_batch_appointment_reschedule'
    PATIENT_RESEARCH = 'patient_research'
    ATTENDANT_RESEARCH = 'attendant_research'
    DOCTOR_APPOINTMENT_BOARD = 'doctor_appointment_board'
    DOCTOR_PATIENT_LOOKUP = 'doctor_patient_lookup'
    DOCTOR_RESEARCH = 'doctor_research'
    DOCTOR_WRITEBACK = 'doctor_writeback'
    ADMIN_TRACE_REVIEW = 'admin_trace_review'


class TraceStatus(str, Enum):
    STARTED = 'started'
    SUCCESS = 'success'
    FAILURE = 'failure'
    SKIPPED = 'skipped'
    AWAITING_CONFIRMATION = 'awaiting_confirmation'


class RouteType(str, Enum):
    HISTORICAL_MATCH = 'historical_match'
    NEW_SYMPTOM = 'new_symptom'
    MIXED_OR_UNCERTAIN = 'mixed_or_uncertain'
    URGENT_ESCALATION = 'urgent_escalation'


class PlannerStep(BaseModel):
    step_id: str
    action: str
    tool_name: str | None = None
    requires_confirmation: bool = False
    depends_on: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    plan_id: str
    actor: ActorType
    user_goal: str
    workflow_type: WorkflowType
    active_patient_id: str | None = None
    steps: list[PlannerStep] = Field(default_factory=list)


class HistoryMatchResult(BaseModel):
    patient_id: str
    has_historical_match: bool
    match_confidence: float = 0.0
    matched_records: list[str] = Field(default_factory=list)
    matched_diagnoses: list[str] = Field(default_factory=list)
    matched_doctors: list[str] = Field(default_factory=list)
    prior_treatments: list[str] = Field(default_factory=list)
    explanation: str = ''


class SymptomRoutingDecision(BaseModel):
    decision_id: str
    patient_id: str
    symptom_text: str
    route_type: RouteType
    history_confidence: float = 0.0
    history_match_record_ids: list[str] = Field(default_factory=list)
    external_search_used: bool = False
    recommended_specialty: str | None = None
    recommended_doctor_ids: list[str] = Field(default_factory=list)
    continuity_treatment_summary: str | None = None
    novelty_reason: str | None = None
    safety_note: str


class MedlinePlusResult(BaseModel):
    title: str
    url: str
    snippet: str | None = None
    source_group: str = 'medlineplus_web'


class SearchMedlinePlusTopicsInput(BaseModel):
    query: str
    topic_type: str = 'symptom_or_disease'
    actor: ActorType


class SearchMedlinePlusTopicsOutput(BaseModel):
    results: list[MedlinePlusResult] = Field(default_factory=list)


class ConfirmationRequest(BaseModel):
    confirmation_id: str
    session_id: str
    action_type: str
    actor: ActorType
    target_ref: str
    prompt: str
    status: str = 'pending'


class StatusUpdate(BaseModel):
    stage: str
    message: str


class WorkflowError(BaseModel):
    stage: str
    message: str
    retryable: bool = False


class ToolTraceMetadata(BaseModel):
    trace_event_id: str
    session_id: str
    langsmith_run_id: str | None = None
    timestamp: datetime
    actor: ActorType
    active_patient_id: str | None = None
    workflow_type: WorkflowType
    step_id: str | None = None
    tool_name: str | None = None
    status: TraceStatus
    latency_ms: int | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    tags: list[str] = Field(default_factory=list)
