# Agentic Healthcare Assistant

A role-based prototype healthcare assistant built with Streamlit, SQLite, ChromaDB, MedlinePlus, and LangSmith-aware tracing.

It demonstrates:

- patient phone-based lookup and registration
- persistent sample patient and medical-record bootstrapping into SQLite
- semantic retrieval over patient summaries and medical history with ChromaDB
- symptom and disease research using patient context plus MedlinePlus
- patient, attendant, doctor, and IT Admin actor-specific workflows
- appointment booking, amend, cancel, and operational management
- persisted interaction logs, planner traces, and admin observability views

## Features

### Patient

- identify the patient by phone number
- register a new patient with a form if the phone number is not found
- review medical history and recent follow-up guidance
- ask symptom and disease questions grounded in patient history and MedlinePlus
- request appointment booking

### Attendant

- use the standard workflow form
- run history and symptom-support requests against a selected patient ID

### Doctor

- view the doctor appointment board and open slots
- look up a patient by appointment, patient name, or patient ID
- research symptoms and treatments with patient-aware context
- add clinical notes and treatment plans to a patient record

### IT Admin

- inspect plan and routing outputs
- inspect retrieval evidence and MedlinePlus evidence
- inspect LangSmith trace metadata and direct trace URL

## Tech Stack

- Python
- Streamlit
- LangSmith
- ChromaDB
- OpenAI embeddings (`text-embedding-3-small`)
- MedlinePlus Web Service
- MedlinePlus Connect
- Pydantic
- pypdf

## Project Structure

```text
healthcare-assistant/
  app/
    prompts/
    repositories/
    schemas/
    services/
    ui/
    workflows/
    utils/
  data/
    chroma/
  docs/
  samples/
  requirements.txt
  app.py
```

## Prerequisites

- Python 3.11+ recommended
- A working virtual environment
- OpenAI API key for embeddings
- LangSmith API key if you want trace links and live tracing

## Environment Variables

Create a `.env` file in the repo root.

Required:

```env
OPENAI_API_KEY=your_openai_key
```

Optional but recommended:

```env
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=healthcare-assistant
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Notes:

- `LANGSMITH_API_KEY` enables the LangSmith client.
- `LANGSMITH_TRACING=true` is required to actually ingest traces.
- `LANGSMITH_PROJECT` controls the LangSmith project where traces are written.
- `LANGSMITH_ENDPOINT` should match your LangSmith region or deployment.
- After changing `.env`, restart the Streamlit app so the running process reloads the environment.

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Data Preparation

The app expects local sample files under:

- [samples](G:/workspace/playground/healthcare-assistant/samples)

These include the workbook and patient PDFs used for ingestion and retrieval.

## Bootstrap SQLite + Chroma

For first-time setup, use the unified bootstrap. It seeds:

- canonical patients into SQLite
- workbook-backed patient snapshot tables
- normalized medical records
- demo appointment data
- Chroma collections for patient summaries and medical history

```powershell
.\.venv\Scripts\python.exe -m app.services.bootstrap_sample_database
```

Optional:

```powershell
.\.venv\Scripts\python.exe -m app.services.bootstrap_sample_database --skip-chroma
.\.venv\Scripts\python.exe -m app.services.bootstrap_sample_database --workbook C:\path\to\records.xlsx --pdf-dir C:\path\to\pdfs
```

This populates:

- [healthcare_assistant.db](G:/workspace/playground/healthcare-assistant/data/healthcare_assistant.db)
- [data/chroma](G:/workspace/playground/healthcare-assistant/data/chroma)

## Run the App

```powershell
streamlit run app.py
```

## Documentation

- [Requirements And Design Plan](G:/workspace/playground/healthcare-assistant/docs/requirements-and-design-plan.md)
- [Solution Design](G:/workspace/playground/healthcare-assistant/docs/solution-design.md)
- [Data Model And Agent Contracts](G:/workspace/playground/healthcare-assistant/docs/data-model-and-agent-contracts.md)

## How to Use the App

### Landing Page

When the app opens, it shows a default landing page with an actor selector dropdown.

- `Patient`
- `Attendant`
- `Doctor`
- `IT Admin`

Select the actor you want to use from the dropdown.

### Patient

1. Open the `Patient` tab.
2. In the chat box, enter the patient's phone number first.
3. If the patient is new, complete the registration form with:
   - first name
   - last name
   - optional residential address
4. If the phone matches an existing patient, the details section populates automatically.
5. Continue the conversation in the chat area.

Example requests:

- `Hello`
- `Can you show my medical history and recent follow up?`
- `I have dry cough and mild fever.`
- `I have dry cough and mild fever and need an appointment.`

### Attendant

1. Open the `Attendant` tab.
2. Optionally select a doctor to filter the open appointment inventory table.
3. Use the chat area for patient, doctor, and appointment operations, including bulk changes.

### Doctor

1. Open the `Doctor` tab.
2. Select a doctor profile to start the doctor session.
3. Review the schedule board filtered to that doctor.
4. Use the chat area for schedule review, patient lookup, research, and appointment actions.

Example requests:

- `Show all open appointments for today.`
- `Pull records for patient pat_david_thompson.`
- `Research symptoms and treatment for diabetes with increased thirst.`
- `Add note and prescribe treatment.`

### IT Admin

1. Open the `IT Admin` tab.
2. Select an admin menu option such as logs, traces, errors, or LangSmith runs.
3. Optionally filter by actor.
4. Use chat for follow-up observability questions.

## Current Workflow Coverage

### Implemented

- patient phone lookup
- new-patient registration form
- actor-specific patient, attendant, doctor, and admin workflows
- patient phone lookup and registration
- patient appointment history, open appointment lookup, booking, amend, and cancel
- attendant patient management, appointment operations, and bulk schedule updates
- doctor schedule review, patient lookup, research, and appointment updates
- admin interaction-log, planner-trace, system-error, and LangSmith-run review
- unified SQLite + Chroma bootstrap

### Prototype limitations

- appointments are demo-seeded during bootstrap and intended for prototype use
- doctor clinical writeback is still lighter than the rest of the persistence model
- LangSmith live availability still depends on local network and credentials
- Langflow assets are not yet checked into the repo

## Key Files

### Workflows

- [patient workflow](G:/workspace/playground/healthcare-assistant/app/workflows/patient/workflow.py)
- [attendant workflow](G:/workspace/playground/healthcare-assistant/app/workflows/attendant/workflow.py)
- [doctor workflow](G:/workspace/playground/healthcare-assistant/app/workflows/doctor/workflow.py)
- [admin workflow](G:/workspace/playground/healthcare-assistant/app/workflows/admin/workflow.py)

### Data and retrieval

- [sample_data_loader.py](G:/workspace/playground/healthcare-assistant/app/services/sample_data_loader.py)
- [sample_data_repository.py](G:/workspace/playground/healthcare-assistant/app/repositories/sample_data_repository.py)
- [local_database.py](G:/workspace/playground/healthcare-assistant/app/repositories/local_database.py)
- [bootstrap_sample_database.py](G:/workspace/playground/healthcare-assistant/app/services/bootstrap_sample_database.py)
- [appointment_repository.py](G:/workspace/playground/healthcare-assistant/app/repositories/appointment_repository.py)
- [chroma_ingestion.py](G:/workspace/playground/healthcare-assistant/app/services/chroma_ingestion.py)
- [chroma_retrieval.py](G:/workspace/playground/healthcare-assistant/app/services/chroma_retrieval.py)
- [medlineplus_service.py](G:/workspace/playground/healthcare-assistant/app/services/medlineplus_service.py)

### UI

- [streamlit_app.py](G:/workspace/playground/healthcare-assistant/app/ui/streamlit_app.py)
- [components.py](G:/workspace/playground/healthcare-assistant/app/ui/components.py)

## Troubleshooting

### LangSmith trace URL is missing

Make sure `.env` includes:

```env
LANGSMITH_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=healthcare-assistant
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Additional notes:

- The app now stores LangSmith URLs only after the run is verified in LangSmith.
- If tracing is disabled or ingestion is not working, `LangSmith Runs` may show no URL instead of a broken link.
- Existing stale rows written before this fix can still contain invalid URLs.
- Restart Streamlit after updating `.env`.

### Chroma retrieval is empty

Re-run the unified bootstrap:

```powershell
.\.venv\Scripts\python.exe -m app.services.bootstrap_sample_database
```

### MedlinePlus or LangSmith requests fail

Check:

- internet access from the environment
- firewall or Windows socket restrictions
- API keys in `.env`

## Next Recommended Improvements

1. Expand automated tests around the new actor workflows.
2. Refresh patient-memory summaries after doctor writeback.
3. Continue tuning MedlinePlus synthesis per actor.
4. Add Langflow workflow artifacts to the repo.
