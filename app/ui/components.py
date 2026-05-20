from __future__ import annotations

import html
import re

import streamlit as st

from app.schemas.contracts import SearchMedlinePlusTopicsOutput
from app.schemas.domain import MedicalRecordEntry, Patient, RecordEntryType, RetrievedEvidence

_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')


def render_evidence_section(title: str, evidence_items: list[RetrievedEvidence]) -> None:
    st.subheader(title)
    if not evidence_items:
        st.caption('No evidence available.')
        return
    for item in evidence_items:
        score = f'{item.score:.2f}' if item.score is not None else 'n/a'
        label = item.source_label or item.chunk_id or item.evidence_id
        with st.expander(f'{label} | score: {score}'):
            if item.metadata:
                st.caption(', '.join(f'{key}={value}' for key, value in item.metadata.items() if value not in (None, '')))
            st.write(item.text)
            if item.source_uri:
                st.code(item.source_uri)


def render_medlineplus_section(results: SearchMedlinePlusTopicsOutput | None, summary: str | None) -> None:
    st.subheader('MedlinePlus Evidence')
    if summary:
        st.markdown('**Synthesized summary**')
        st.write(clean_text(summary))
    if not results or not results.results:
        st.caption('No MedlinePlus results available.')
        return
    for item in results.results:
        with st.expander(clean_text(item.title)):
            if item.snippet:
                st.write(clean_text(item.snippet))
            st.markdown(f'Source: [{clean_text(item.url)}]({item.url})')
            st.caption(f'Group: {item.source_group}')


def render_doctor_research_summary(summary: dict[str, object] | None) -> None:
    st.subheader('Doctor Research Summary')
    if not summary:
        st.caption('No doctor-specific summary available.')
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Patient context**')
        st.write(summary.get('patient_name') or 'Unknown patient')
        if summary.get('patient_id'):
            st.caption(f"Patient id: {summary['patient_id']}")
        st.markdown('**Historical diagnosis**')
        st.write(summary.get('historical_diagnosis') or 'No historical diagnosis identified')
    with col2:
        st.markdown('**Suggested specialty lens**')
        st.write(summary.get('suggested_specialty_lens') or 'General medicine')
        st.markdown('**Prior treatment context**')
        st.write(summary.get('prior_treatment_context') or 'No prior treatment context available')
    if summary.get('medlineplus_summary'):
        st.markdown('**MedlinePlus synthesis**')
        st.write(clean_text(str(summary['medlineplus_summary'])))
    follow_ups = summary.get('follow_up_prompts') or []
    if follow_ups:
        st.markdown('**Suggested follow-up prompts**')
        for item in follow_ups:
            st.write(f'- {item}')


def render_doctor_appointments(appointments: list[dict[str, str]]) -> None:
    st.subheader('Doctor Appointments')
    if not appointments:
        st.caption('No appointments available.')
        return
    for appointment in appointments:
        patient_id = appointment.get('patient_id') or 'unassigned'
        st.write(
            f"{appointment.get('appointment_id')}: {appointment.get('appointment_date')} {appointment.get('appointment_time')} | "
            f"{appointment.get('specialty')} | patient {patient_id} | status {appointment.get('status')}"
        )


def render_doctor_writeback_summary(summary: dict[str, object] | None) -> None:
    st.subheader('Doctor Writeback')
    if not summary:
        st.caption('No note or treatment writeback recorded.')
        return
    st.write(f"Patient: {summary.get('patient_name')} ({summary.get('patient_id')})")
    st.write(f"Record id: {summary.get('record_id')}")
    if summary.get('visit_date'):
        st.caption(f"Visit date: {summary['visit_date']}")
    if summary.get('note'):
        st.markdown('**Clinical note**')
        st.write(summary['note'])
    if summary.get('treatment_plan'):
        st.markdown('**Treatment plan**')
        st.write(summary['treatment_plan'])


def render_admin_trace_summary(summary: dict[str, object] | None) -> None:
    st.subheader('Admin Observability Summary')
    if not summary:
        st.caption('No admin summary available.')
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Plan steps', summary.get('plan_steps', 0))
    c2.metric('History hits', summary.get('history_rag_hits', 0))
    c3.metric('Memory hits', summary.get('memory_rag_hits', 0))
    c4.metric('MedlinePlus hits', summary.get('medlineplus_hits', 0))
    c5, c6, c7 = st.columns(3)
    c5.metric('Errors', summary.get('errors', 0))
    c6.metric('Status events', summary.get('status_count', 0))
    c7.metric('Pending confirmation', 'Yes' if summary.get('pending_confirmation') else 'No')
    st.markdown('**Trace context**')
    st.write(f"Workflow type: {summary.get('workflow_type')}")
    st.write(f"Patient id: {summary.get('patient_id') or 'n/a'}")
    st.write(f"Patient name: {summary.get('patient_name') or 'n/a'}")
    st.write(f"Route type: {summary.get('route_type') or 'n/a'}")
    st.write(f"LangSmith enabled: {'Yes' if summary.get('langsmith_enabled') else 'No'}")
    if summary.get('langsmith_run_id'):
        st.caption('Run ID')
        st.code(str(summary['langsmith_run_id']))
    if summary.get('langsmith_run_url'):
        st.link_button('Open LangSmith Trace', str(summary['langsmith_run_url']))


def render_patient_directory(patients: list[Patient]) -> None:
    st.subheader('Registered Patients')
    if not patients:
        st.caption('No registered patients available.')
        return
    rows = [
        {
            'Patient ID': patient.patient_id,
            'Name': patient.full_name,
            'Phone': patient.phone or '',
            'Address': patient.address or '',
            'Primary Conditions': ', '.join(patient.primary_conditions),
            'Chronic Conditions': ', '.join(patient.chronic_conditions),
            'Allergies': ', '.join(patient.allergies),
        }
        for patient in patients
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_patient_profile_summary(patient: Patient | None) -> None:
    st.subheader('Patient Profile')
    if patient is None:
        st.caption('No patient selected.')
        return
    left, right = st.columns(2)
    with left:
        st.write(f'Name: {patient.full_name}')
        st.write(f'Patient ID: {patient.patient_id}')
        st.write(f'Phone: {patient.phone or "n/a"}')
        st.write(f'Address: {patient.address or "n/a"}')
    with right:
        st.write(f'Primary conditions: {", ".join(patient.primary_conditions) or "n/a"}')
        st.write(f'Chronic conditions: {", ".join(patient.chronic_conditions) or "n/a"}')
        st.write(f'Allergies: {", ".join(patient.allergies) or "n/a"}')


def render_attendant_history_summary(summary: dict[str, object] | None) -> None:
    st.subheader('History Summary')
    if not summary:
        st.caption('No patient history summary available.')
        return
    st.write(f"Patient: {summary.get('patient_name') or 'n/a'}")
    if summary.get('summary_text'):
        st.write(summary['summary_text'])
    if summary.get('primary_conditions'):
        st.write(f"Primary conditions: {', '.join(summary['primary_conditions'])}")
    if summary.get('chronic_conditions'):
        st.write(f"Chronic conditions: {', '.join(summary['chronic_conditions'])}")
    if summary.get('allergies'):
        st.write(f"Allergies: {', '.join(summary['allergies'])}")
    if summary.get('latest_visit_date'):
        st.caption(f"Latest visit: {summary['latest_visit_date']}")
    if summary.get('latest_visit_summary'):
        st.write(summary['latest_visit_summary'])


def render_patient_medical_history(records: list[MedicalRecordEntry]) -> None:
    visit_records = [record for record in records if record.entry_type != RecordEntryType.HISTORY_SUMMARY]
    summary_records = [record for record in records if record.entry_type == RecordEntryType.HISTORY_SUMMARY]

    st.subheader('Medical History')
    if not visit_records:
        st.caption('No dated medical-history visits available.')
    else:
        rows = [
            {
                'Visit Date': record.visit_date.isoformat() if record.visit_date else '',
                'Title': record.title,
                'Type': record.entry_type.value,
                'Visit Summary': str(record.structured_fields.get('visit_summary') or record.assessment or record.title),
                'Source': record.source_type,
            }
            for record in visit_records
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        for record in visit_records:
            heading = record.visit_date.isoformat() if record.visit_date else record.title
            summary = str(record.structured_fields.get('visit_summary') or record.assessment or record.title)
            with st.expander(f'{heading} | {summary}'):
                st.caption(f'Title: {record.title}')
                if record.subjective:
                    st.markdown('**Subjective**')
                    st.write(record.subjective)
                if record.objective:
                    st.markdown('**Objective**')
                    st.write(record.objective)
                if record.assessment:
                    st.markdown('**Diagnosis / Assessment**')
                    st.write(record.assessment)
                if record.plan:
                    st.markdown('**Detailed Plan**')
                    st.write(record.plan)

    if summary_records:
        st.subheader('Imported Workbook Summary')
        for record in summary_records:
            with st.expander(record.title):
                st.write(record.subjective or 'No summary text available.')


def render_patient_appointment_history(appointments: list[dict[str, str]]) -> None:
    st.subheader('Appointment History')
    if not appointments:
        st.caption('No appointment history available.')
        return
    st.dataframe(appointments, use_container_width=True, hide_index=True)


def clean_text(value: str) -> str:
    cleaned = html.unescape(value)
    cleaned = _TAG_RE.sub('', cleaned)
    return _WS_RE.sub(' ', cleaned).strip()
