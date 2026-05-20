# Agentic Healthcare Assistant

## 1. Project Summary

This project is a prototype healthcare assistant built around four actor-specific experiences:

- Patient
- Attendant
- Doctor
- IT Admin

The system uses:

- Streamlit for the demo UI
- SQLite as the transactional system of record
- ChromaDB as the semantic memory and retrieval layer over patient summaries and historical medical records
- MedlinePlus as the approved external medical-information source
- LangSmith-compatible tracing and locally persisted observability logs

The current design is intentionally prototype-oriented. It is meant to demonstrate safe administrative support, contextual medical-information support, patient-history retrieval, schedule operations, and workflow observability. It is not intended to make autonomous clinical decisions.

## 2. Confirmed Product Decisions

1. This is a self-contained prototype using local sample data.
2. The supported actors are `Patient`, `Attendant`, `Doctor`, and `IT Admin`.
3. Patient identification in the patient experience is phone-first.
4. New patient registration is form-based and persists to SQLite.
5. SQLite is the source of truth for patients, medical records, appointments, interaction logs, planner traces, system errors, and LangSmith run references.
6. ChromaDB is used for semantic retrieval over patient summaries and past medical history.
7. MedlinePlus is the approved external research source.
8. OpenAI embeddings use `text-embedding-3-small`.
9. The refactored application uses actor-specific workflows instead of a single shared routing graph.
10. Prompts are being externalized by actor and by task for easier tuning.

## 3. Scope

### In scope

- Local workbook and PDF ingestion
- Unified bootstrap into SQLite and ChromaDB
- Patient lookup and registration
- Patient appointment lookup, booking, reschedule, and cancel
- Patient symptom research grounded in patient history plus MedlinePlus
- Attendant patient lookup, patient maintenance, and appointment operations
- Attendant bulk appointment changes through natural language
- Doctor schedule review, patient lookup, research, and appointment changes
- IT Admin review of interaction logs, planner traces, system errors, and LangSmith runs
- Persisted interaction and planner telemetry for debugging

### Out of scope

- External EHR integration
- Production authentication and authorization
- Real prescription transmission
- Autonomous diagnosis or treatment decisions
- Production security and compliance hardening
- Production-grade scheduling integrations

## 4. Target User Experience

### 4.1 Landing page

The app presents a default landing page with role selection for demo use. In a final product this would be replaced by actor-specific login and auth.

### 4.2 Patient experience

The Patient page is a conversational workspace with:

- a personal-information pane
- a chat pane
- history and appointment results rendered as structured tables

Expected behavior:

1. Ask for the patient phone number first.
2. If found, populate profile information from SQLite.
3. If not found, collect first name, last name, phone, and optional address in a registration form.
4. Allow requests for:
   - past appointment details
   - current appointments
   - open appointments
   - symptom assistance
   - booking, amending, and cancelling appointments
5. Use ChromaDB retrieval over past medical records and summaries when the patient describes symptoms or asks for medical context.
6. Use MedlinePlus only for explicit research or symptom-support situations, not for historical-record lookup.
7. Synthesize a short, coherent patient-facing response instead of stitching tool output directly.

### 4.3 Attendant experience

The Attendant page is a chat-first operations workspace with:

- open appointments preloaded at the top
- a single attendant chat interface for patient and appointment operations
- structured result panels rendered below the chat when a workflow returns patient, history, or appointment data

Expected behavior:

1. View all active patients.
2. Edit or delete patients.
3. Pull patient medical history and appointment history.
4. Show all booked appointments across the system.
5. Schedule, amend, and cancel patient appointments.
6. Perform bulk updates through typed requests such as doctor/date rescheduling.
7. Treat open appointments as inventory only. Open-slot management is not a primary attendant BAU workflow in the current implementation.

### 4.4 Doctor experience

The Doctor page is a chat-first schedule and patient-context workspace.

Expected behavior:

1. Show today’s appointments first.
2. If none exist for today, fall back to the current week.
3. Interpret generic appointment and schedule requests as booked or scheduled appointments, not open-slot inventory.
4. Allow lookup by date, patient name, phone, patient id, or selected appointment.
5. Allow appointment amend and cancel.
6. Allow patient-history review.
7. Allow symptom and treatment research using Chroma-grounded patient history plus MedlinePlus.
8. Keep doctor responses more structured and evidence-rich than patient responses.

### 4.5 IT Admin experience

The IT Admin page is the observability workspace.

Expected behavior:

1. View interaction logs for the day.
2. View planner traces for the day.
3. View system errors for the day.
4. View LangSmith run references for the day.
5. Filter by actor.
6. Use typed follow-up questions to inspect stored operational data.

## 5. Functional Requirements

### FR1. Actor-specific workflows

The system must maintain separate workflow paths for each actor. Shared services are allowed, but planning, response synthesis, and legal workflow transitions must remain actor-aware.

Current status:

- implemented for Patient, Attendant, Doctor, and IT Admin

### FR2. Context continuity

The system must preserve short-term conversation context for up to 15 interactions and use that context in LLM intent interpretation and response synthesis.

Current status:

- implemented as rolling recent-message context in actor workflows
- can still be strengthened with richer transcript summarization

### FR3. Patient identity and registration

The system must:

- identify patients by phone
- create new patients through a form
- generate a unique `patient_id`
- persist the result in SQLite

Current status:

- implemented

### FR4. Medical-history retrieval

The system must retrieve and synthesize patient history from:

- structured records in SQLite
- semantic retrieval from ChromaDB over patient summaries and historical records

Current status:

- implemented

### FR5. Appointment management

The system must support:

- patient lookup of appointments
- open appointment lookup
- booking
- reschedule
- cancel
- attendant schedule operations
- doctor schedule operations

Appointment semantics:

- open appointments = available inventory
- booked appointments = user-facing scheduled visits
- patient, attendant, and doctor flows default to booked or scheduled appointment operations unless the request explicitly asks for open or available slots

Current status:

- implemented in the SQLite-backed appointment repository

### FR6. Symptom research

The system must combine:

- prior patient medical history from ChromaDB
- structured patient/record context from SQLite
- MedlinePlus results where appropriate

Current status:

- implemented for patient and doctor flows

### FR7. Response synthesis

The system must synthesize coherent actor-specific responses instead of concatenating raw outputs from tools.

Current status:

- implemented in the refactored patient, attendant, doctor, and admin response generators
- still open to tuning for tone and brevity

### FR8. Persistence and traceability

The system must persist:

- patients
- records
- appointments
- interaction logs
- planner traces
- system errors
- LangSmith run references

Current status:

- implemented in SQLite

### FR9. Safe scope control

The system must treat medical-information answers as supportive and informational, not as autonomous medical judgment.

Current status:

- implemented as a design and response constraint

## 6. Current Architecture Direction

The current codebase has moved away from the original shared graph. The active architecture is now:

1. Actor-specific Streamlit UI pages
2. Actor-specific workflow modules under `app/workflows/`
3. Shared repositories and services underneath
4. SQLite persistence for operational state
5. ChromaDB retrieval for semantic medical context
6. MedlinePlus for external medical evidence
7. Persisted logs and LangSmith-aware tracing for observability

UI simplification status:

- Patient remains a profile-plus-chat workspace
- Attendant is now open-appointments plus chat
- Doctor is now schedule-board plus chat
- IT Admin remains an observability workspace

## 7. Data and Retrieval Design

### Source data

- [records.xlsx](G:/workspace/playground/healthcare-assistant/samples/records.xlsx)
- `sample_*.pdf` reports in [samples](G:/workspace/playground/healthcare-assistant/samples)

### Bootstrapped persistent data

The unified bootstrap seeds:

- canonical patients
- workbook-backed patient snapshot rows
- normalized medical records
- seeded demo appointments
- Chroma collections for semantic retrieval

### Data split by responsibility

#### SQLite

Used for:

- patient details
- medical records
- appointment state
- interaction logs
- planner traces
- system errors
- LangSmith run references

#### ChromaDB

Used for:

- patient summaries
- historical medical record retrieval
- semantic grounding for symptom synthesis and patient-context retrieval

#### MedlinePlus

Used for:

- symptom research
- disease research
- treatment research

## 8. Safety And Guardrails

- Patient flows should not drift into doctor, attendant, or admin workflows.
- Attendant flows should not drift into patient, doctor, or admin workflows.
- Doctor flows should stay within doctor-owned operational and research capabilities.
- Admin flows are observability-focused.
- Historical lookup should not trigger MedlinePlus unless the user is explicitly asking for medical research or symptom understanding.
- Symptom guidance should be contextual and supportive, not a definitive diagnosis.

## 9. Current Known Limitations

- Some prompt and response tuning still needs refinement for naturalness and brevity.
- Doctor clinical writeback is lighter than the rest of the persistence model.
- Chroma retrieval ranking and summarization can still be improved.
- Langflow assets are not yet checked into the repo as executable artifacts.
- Conversation context is stronger than before but can still be enhanced with richer summarization.

## 10. Recommended Next Steps

1. Finish updating all project docs and specs to the actor-specific architecture.
2. Add stronger automated tests around patient and appointment flows.
3. Improve Chroma retrieval ranking and memory refresh after writeback.
4. Continue tightening actor-specific prompt behavior.
5. Add Langflow assets that mirror the current workflows.
