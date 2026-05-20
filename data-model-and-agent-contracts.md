# Data Model And Agent Contracts

## 1. Purpose

This document describes the current domain models, persistence contracts, retrieval contracts, and actor-workflow contracts used by the refactored healthcare assistant.

It reflects the active implementation rather than the retired shared-graph design.

## 2. Canonical Domain Models

Primary schema files:

- [domain.py](G:/workspace/playground/healthcare-assistant/app/schemas/domain.py)
- [contracts.py](G:/workspace/playground/healthcare-assistant/app/schemas/contracts.py)
- [raw.py](G:/workspace/playground/healthcare-assistant/app/schemas/raw.py)

## 2.1 ActorType

Implemented values:

- `patient`
- `attendant`
- `doctor`
- `it_admin`

## 2.2 Patient

Implemented fields:

- `patient_id`
- `full_name`
- `date_of_birth`
- `gender`
- `phone`
- `address`
- `primary_conditions`
- `allergies`
- `chronic_conditions`
- `preferred_doctors`
- `source_refs`

Usage notes:

- patient lookup in the Patient workflow is phone-first
- doctor and attendant workflows can resolve patients by id, name, phone, or appointment context
- patient summaries from workbook parsing populate `primary_conditions`, `allergies`, and `chronic_conditions`

## 2.3 Appointment

Implemented fields:

- `appointment_id`
- `patient_id`
- `doctor_id`
- `specialty`
- `status`
- `appointment_date`
- `appointment_time`
- `location_id`
- `booked_by_actor`
- `booking_reason`
- `created_at`

Implemented statuses:

- `available`
- `held`
- `booked`
- `cancelled`
- `completed`

Operational semantics:

- `available` = inventory or open slot
- `booked` and `held` = user-facing scheduled appointments
- patient, attendant, and doctor workflows default to booked appointment operations unless the request explicitly asks for open or available slots

## 2.4 MedicalRecordEntry

Implemented fields:

- `record_id`
- `patient_id`
- `entry_type`
- `visit_date`
- `title`
- `subjective`
- `objective`
- `assessment`
- `plan`
- `doctor_id`
- `source_type`
- `source_path`
- `structured_fields`

Important usage:

- PDF-derived visits are normalized into `visit_note`
- workbook-derived summaries are normalized into `history_summary`
- doctor writeback currently adds manual entries with `source_type="manual_doctor_entry"`

## 2.5 Diagnosis

Implemented fields:

- `diagnosis_id`
- `patient_id`
- `record_id`
- `name`
- `code_system`
- `code`
- `status`
- `diagnosed_on`

Usage:

- diagnosis codes support MedlinePlus Connect enrichment

## 2.6 PatientMemory

Implemented fields:

- `memory_id`
- `patient_id`
- `summary_type`
- `summary_text`
- `derived_from_record_ids`
- `key_facts`
- `created_at`
- `updated_at`

Usage:

- Chroma stores patient-level semantic memory summaries for retrieval grounding

## 2.7 RetrievedEvidence

Implemented fields:

- `evidence_id`
- `source_group`
- `source_type`
- `source_label`
- `source_uri`
- `patient_id`
- `chunk_id`
- `text`
- `score`
- `metadata`

Common source groups:

- `patient_history`
- `patient_memory`
- `medlineplus_web`
- `medlineplus_connect`

## 3. WorkflowType Contract

The shared workflow enum in [contracts.py](G:/workspace/playground/healthcare-assistant/app/schemas/contracts.py) includes:

- `symptom_to_appointment`
- `patient_appointment_lookup`
- `patient_appointment_reschedule`
- `patient_appointment_cancel`
- `patient_history_summary`
- `attendant_history_summary`
- `attendant_appointment_schedule`
- `attendant_appointment_reschedule`
- `attendant_appointment_cancel`
- `attendant_batch_appointment_reschedule`
- `patient_research`
- `attendant_research`
- `doctor_appointment_board`
- `doctor_patient_lookup`
- `doctor_research`
- `doctor_writeback`
- `admin_trace_review`

This enum remains as cross-cutting vocabulary from the earlier shared-graph design, but the active implementation now uses actor-specific intent names and workflow modules under `app/workflows/`.

## 4. Actor Workflow Contracts

The runtime workflows are now actor-specific and may use more task-oriented intent names internally.

## 4.1 Patient workflow contract

Core responsibilities:

- identify patient
- register patient
- show past appointments
- show current appointments
- show open appointments
- research symptoms
- book appointment
- amend appointment
- cancel appointment

Typical patient state includes:

- `session_id`
- `actor`
- `user_query`
- `patient_phone`
- `active_patient_id`
- `patient_profile`
- `conversation_context`
- `workflow_type`
- `intent_decision`
- `history_rag_results`
- `memory_rag_results`
- `medline_payload`
- `external_summary`
- `final_response`
- `langsmith_run_id`
- `langsmith_run_url`

## 4.2 Attendant workflow contract

Core responsibilities:

- show open appointments
- show booked appointments
- show active patients
- view patient history
- edit patient details
- delete patient
- schedule, reschedule, and cancel patient appointments
- bulk doctor/date reschedule operations for booked appointments

Typical attendant state includes:

- `session_id`
- `actor`
- `user_query`
- `active_patient_id`
- `selected_appointment_id`
- `conversation_context`
- `workflow_type`
- `patient_history_summary`
- `appointment_rows`
- `open_appointment_rows`
- `batch_appointment_updates`
- `final_response`

## 4.3 Doctor workflow contract

Core responsibilities:

- show booked schedule
- resolve patient context
- view patient history
- research symptoms and treatment
- reschedule or cancel appointments

Typical doctor state includes:

- `session_id`
- `actor`
- `user_query`
- `selected_appointment_id`
- `active_patient_id`
- `patient_name_query`
- `patient_phone_query`
- `schedule_date_query`
- `reschedule_date`
- `conversation_context`
- `workflow_type`
- `selected_patient`
- `patient_history_summary`
- `history_rag_results`
- `memory_rag_results`
- `external_summary`
- `appointment_rows`
- `final_response`

## 4.4 Admin workflow contract

Core responsibilities:

- view interaction logs
- view planner traces
- view system errors
- view LangSmith runs

Typical admin state includes:

- `session_id`
- `actor`
- `user_query`
- `selected_view`
- `actor_filter`
- `conversation_context`
- `workflow_type`
- `interaction_rows`
- `planner_trace_rows`
- `system_error_rows`
- `langsmith_run_rows`
- `final_response`

## 5. Planning Contract

The refactored design is LLM-led but actor-scoped.

Planning responsibilities:

1. Use the actor-specific intent classifier.
2. Consider short-term conversation context.
3. Return an intent and extracted entities.
4. Map the result into a legal actor-specific workflow path.
5. Persist planner traces in SQLite.

Persisted planner trace fields include:

- actor
- session id
- user message
- context
- planner output
- final workflow type
- timestamp

## 6. Retrieval Contracts

## 6.1 SQLite responsibility

SQLite is the system of record for:

- patients
- medical records
- appointments
- interaction logs
- planner traces
- system errors
- LangSmith run references

## 6.2 Chroma responsibility

ChromaDB is the semantic retrieval layer for:

- patient summaries
- historical medical records
- visit summaries
- memory summaries

Chroma should support:

- patient symptom synthesis
- doctor research grounding
- contextual history retrieval

Chroma should not be treated as the transactional source of truth for appointments or identity.

## 6.3 MedlinePlus responsibility

MedlinePlus is the approved external knowledge source for:

- symptom topics
- disease topics
- treatment education

It is used in conjunction with patient history rather than as a standalone answer engine.

## 7. Confirmation And Update Contracts

The shared `ConfirmationRequest` model includes:

- `confirmation_id`
- `session_id`
- `action_type`
- `actor`
- `target_ref`
- `prompt`
- `status`

Common action patterns:

- `book_appointment`
- `offer_booking`

Operational appointment updates are now persisted directly through the appointment repository for:

- booking
- reschedule
- cancel
- open-slot move
- open-slot cancel

## 8. Persistence Contracts

## 8.1 Local database repository

The SQLite repository now supports:

- patient upsert and lookup
- medical record upsert and lookup
- workbook snapshot tables
- appointment upsert and listing
- interaction logging
- planner trace logging
- system error logging
- LangSmith run logging
- snapshot cleanup methods used during bootstrap

## 8.2 Sample data repository

The repository layer exposes:

- patient lookup by id, phone, and name
- search by partial name
- record lookup
- diagnosis lookup
- recent-record lookup
- patient registration
- clinical note writeback
- interaction, planner, error, and LangSmith logging wrappers

## 8.3 Appointment repository

The appointment repository is SQLite-backed and currently supports:

- seeded demo schedule reset
- open appointment listing
- patient appointment listing
- current patient appointment listing
- date and date-range schedule lookup
- doctor/date schedule lookup
- booking
- reschedule
- cancel
- bulk doctor/date reschedule

## 9. Observability Contract

Observability has two layers:

### Application persistence

Stored locally in SQLite:

- interactions
- planner traces
- errors
- LangSmith run references

### LangSmith runtime tracing

When enabled, workflows store:

- `langsmith_run_id`
- `langsmith_run_url`

and persist those references for later IT Admin inspection.

## 10. Current Gaps

- Doctor writeback persistence is lighter than the main record bootstrap path.
- Automatic memory refresh after doctor writeback is not yet complete.
- Some actor prompt/output contracts still need tuning for better consistency.
- Langflow artifacts are still not checked in as runtime definitions.

## 11. Recommended Contract Follow-ups

1. Add stronger automated test fixtures for each actor workflow.
2. Add a post-writeback memory refresh contract.
3. Continue tightening actor-specific synthesis contracts.
4. Add explicit versioned prompt metadata for auditability.
