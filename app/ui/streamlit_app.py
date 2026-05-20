from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

from pypdf import PdfReader
import streamlit as st

from app.repositories.appointment_repository import get_appointment_repository
from app.repositories.sample_data_repository import get_sample_repository
from app.schemas.domain import ActorType, Patient
from app.ui.components import (
    render_attendant_history_summary,
    render_doctor_appointments,
    render_medlineplus_section,
    render_patient_medical_history,
    render_patient_profile_summary,
)
from app.workflows.admin import AdminAssistantWorkflow
from app.workflows.attendant import AttendantAssistantWorkflow
from app.workflows.doctor import DoctorAssistantWorkflow
from app.workflows.patient import PatientAssistantWorkflow

_PATIENT_WORKFLOW = PatientAssistantWorkflow()
_ATTENDANT_WORKFLOW = AttendantAssistantWorkflow()
_DOCTOR_WORKFLOW = DoctorAssistantWorkflow()
_ADMIN_WORKFLOW = AdminAssistantWorkflow()
_APPOINTMENTS = get_appointment_repository()
_PATIENTS = get_sample_repository()

_DEFAULT_QUERIES = {
    'attendant': 'Please show the patient history and any recent follow-up guidance.',
}


def _reset_application_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _appointment_rows_for_open_slots(doctor_query: str | None = None) -> list[dict[str, str]]:
    appointments = _APPOINTMENTS.list_open_appointments()
    if doctor_query:
        doctor_id = _APPOINTMENTS.resolve_doctor_id(doctor_query)
        appointments = [appointment for appointment in appointments if appointment.doctor_id == doctor_id]
    return [
        {
            'Appointment ID': appointment.appointment_id,
            'Date': appointment.appointment_date.isoformat(),
            'Time': appointment.appointment_time.strftime('%H:%M'),
            'Doctor': _APPOINTMENTS.get_doctor_display_name(appointment.doctor_id),
            'Specialty': appointment.specialty.replace('_', ' '),
            'Status': appointment.status.value,
        }
        for appointment in appointments
    ]


def _default_doctor_schedule_rows(doctor_query: str | None = None) -> list[dict[str, str]]:
    today = date.today()
    if doctor_query:
        appointments = _APPOINTMENTS.list_doctor_appointments_for_date(doctor_query, today)
    else:
        appointments = _APPOINTMENTS.list_booked_appointments_for_date(today)
    if not appointments:
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=5)
        if doctor_query:
            appointments = _APPOINTMENTS.list_doctor_appointments_for_range(doctor_query, week_start, week_end)
        else:
            appointments = _APPOINTMENTS.list_booked_appointments_for_range(week_start, week_end)
    return [
        {
            'Appointment ID': appointment.appointment_id,
            'Date': appointment.appointment_date.isoformat(),
            'Time': appointment.appointment_time.strftime('%H:%M'),
            'Doctor': _APPOINTMENTS.get_doctor_display_name(appointment.doctor_id),
            'Patient ID': appointment.patient_id or 'unassigned',
            'Specialty': appointment.specialty.replace('_', ' '),
            'Status': appointment.status.value,
            'Reason': appointment.booking_reason,
        }
        for appointment in appointments
    ]


def _workflow_name(result: dict[str, object] | None) -> str | None:
    if not result:
        return None
    workflow = result.get('workflow_type')
    if hasattr(workflow, 'value'):
        return workflow.value
    return str(workflow) if workflow else None


def _normalize_result_for_display(actor: str, result: dict[str, object] | None) -> dict[str, object] | None:
    if not result:
        return result
    normalized = dict(result)
    list_keys = {
        'appointment_rows',
        'open_appointment_rows',
        'visit_history_rows',
        'patient_directory_rows',
        'doctor_directory_rows',
        'batch_appointment_updates',
        'patient_records',
        'interaction_rows',
        'planner_trace_rows',
        'system_error_rows',
        'langsmith_run_rows',
    }
    object_keys = {
        'selected_patient',
        'selected_doctor',
        'patient_history_summary',
        'external_summary',
    }
    for key in list_keys:
        normalized[key] = []
    for key in object_keys:
        normalized[key] = None
    normalized['appointment_table_title'] = None

    workflow_type = _workflow_name(result)
    completed_actions = set(result.get('completed_actions') or [])

    if actor == 'patient':
        if workflow_type == 'show_medical_history' and result.get('visit_history_rows'):
            normalized['visit_history_rows'] = result['visit_history_rows']
        if workflow_type in {'show_past_appointments'} and result.get('appointment_rows'):
            normalized['appointment_rows'] = result['appointment_rows']
            normalized['appointment_table_title'] = 'Appointments on Record'
        if workflow_type in {'book_appointment', 'amend_appointment'} and result.get('appointment_rows'):
            normalized['appointment_rows'] = result['appointment_rows']
            normalized['appointment_table_title'] = 'Scheduled Appointment Details'
        if workflow_type in {'show_open_appointments', 'book_appointment'} and result.get('open_appointment_rows'):
            normalized['open_appointment_rows'] = result['open_appointment_rows']
        if result.get('visit_history_rows'):
            normalized['visit_history_rows'] = result['visit_history_rows']
        return normalized

    if actor == 'attendant':
        if result.get('selected_patient'):
            normalized['selected_patient'] = result['selected_patient']
        if result.get('selected_doctor'):
            normalized['selected_doctor'] = result['selected_doctor']
        if result.get('patient_history_summary'):
            normalized['patient_history_summary'] = result['patient_history_summary']
        if result.get('patient_records'):
            normalized['patient_records'] = result['patient_records']
        if completed_actions.intersection({'show_active_patients', 'delete_patient'}) and result.get('patient_directory_rows'):
            normalized['patient_directory_rows'] = result['patient_directory_rows']
        if completed_actions.intersection({'show_doctors', 'register_doctor', 'edit_doctor_details'}) and result.get('doctor_directory_rows'):
            normalized['doctor_directory_rows'] = result['doctor_directory_rows']
        if result.get('appointment_rows'):
            normalized['appointment_rows'] = result['appointment_rows']
            normalized['appointment_table_title'] = result.get('appointment_table_title')
        if 'show_open_appointments' in completed_actions and result.get('open_appointment_rows'):
            normalized['open_appointment_rows'] = result['open_appointment_rows']
        if result.get('batch_appointment_updates'):
            normalized['batch_appointment_updates'] = result['batch_appointment_updates']
        return normalized

    if actor == 'doctor':
        if workflow_type == 'show_schedule' and result.get('appointment_rows'):
            normalized['appointment_rows'] = result['appointment_rows']
        if workflow_type in {'view_patient_history', 'amend_appointment', 'cancel_appointment'} and result.get('selected_patient'):
            normalized['selected_patient'] = result['selected_patient']
        if workflow_type == 'view_patient_history' and result.get('patient_history_summary'):
            normalized['patient_history_summary'] = result['patient_history_summary']
        if workflow_type == 'view_patient_history' and result.get('patient_records'):
            normalized['patient_records'] = result['patient_records']
        if workflow_type in {'view_patient_history', 'amend_appointment', 'cancel_appointment'} and result.get('appointment_rows'):
            normalized['appointment_rows'] = result['appointment_rows']
        if result.get('external_summary'):
            normalized['external_summary'] = result['external_summary']
        return normalized

    if actor == 'admin':
        if workflow_type == 'view_interaction_logs' and result.get('interaction_rows'):
            normalized['interaction_rows'] = result['interaction_rows']
        if workflow_type == 'view_planner_traces' and result.get('planner_trace_rows'):
            normalized['planner_trace_rows'] = result['planner_trace_rows']
        if workflow_type == 'view_system_errors' and result.get('system_error_rows'):
            normalized['system_error_rows'] = result['system_error_rows']
        if workflow_type == 'view_langsmith_runs' and result.get('langsmith_run_rows'):
            normalized['langsmith_run_rows'] = result['langsmith_run_rows']
        return normalized

    return normalized


def _run_attendant_workflow(selected_patient_id: str | None, user_query: str) -> dict[str, object]:
    return _ATTENDANT_WORKFLOW.run(
        {
            'session_id': 'sess_attendant',
            'actor': ActorType.ATTENDANT,
            'active_patient_id': selected_patient_id,
            'conversation_context': _build_attendant_conversation_context(),
            'uploaded_report_text': st.session_state.get('attendant_uploaded_report_text'),
            'uploaded_report_name': st.session_state.get('attendant_uploaded_report_name'),
            'user_query': user_query,
        }
    )


def _run_doctor_workflow(
    *,
    user_query: str,
    selected_doctor_id: str | None,
    selected_appointment_id: str | None,
    active_patient_id: str | None,
    patient_name_query: str | None,
    patient_phone_query: str | None,
    schedule_date_query: str | None,
    reschedule_date: str | None,
) -> dict[str, object]:
    return _DOCTOR_WORKFLOW.run(
        {
            'session_id': 'sess_doctor',
            'actor': ActorType.DOCTOR,
            'selected_doctor_id': selected_doctor_id,
            'selected_appointment_id': selected_appointment_id,
            'active_patient_id': active_patient_id,
            'patient_name_query': patient_name_query,
            'patient_phone_query': patient_phone_query,
            'schedule_date_query': schedule_date_query,
            'reschedule_date': reschedule_date,
            'conversation_context': _build_doctor_conversation_context(),
            'user_query': user_query,
        }
    )


def _run_admin_workflow(*, user_query: str, selected_view: str | None, actor_filter: str | None) -> dict[str, object]:
    return _ADMIN_WORKFLOW.run(
        {
            'session_id': 'sess_admin',
            'actor': ActorType.IT_ADMIN,
            'selected_view': selected_view,
            'actor_filter': actor_filter,
            'conversation_context': _build_admin_conversation_context(),
            'user_query': user_query,
        }
    )


def _split_name(profile: Patient | None) -> tuple[str, str]:
    if profile is None or not profile.full_name.strip():
        return '', ''
    parts = profile.full_name.strip().split()
    if len(parts) == 1:
        return parts[0], ''
    return parts[0], ' '.join(parts[1:])


def _init_patient_state() -> None:
    st.session_state.setdefault(
        'patient_messages',
        [{'role': 'assistant', 'content': "Please share the patient's phone number to continue."}],
    )
    st.session_state.setdefault('patient_phone', '')
    st.session_state.setdefault('patient_profile', None)
    st.session_state.setdefault('patient_last_result', None)
    st.session_state.setdefault('patient_pending_registration', False)
    st.session_state.setdefault('patient_registration_first_name', '')
    st.session_state.setdefault('patient_registration_last_name', '')
    st.session_state.setdefault('patient_registration_address', '')
    st.session_state.setdefault('patient_pending_query', 'Hello')
    st.session_state.setdefault('patient_pending_user_query', None)


def _append_patient_message(role: str, content: str) -> None:
    st.session_state.patient_messages.append({'role': role, 'content': content})


def _build_patient_conversation_context() -> dict[str, object]:
    last_result = st.session_state.get('patient_last_result') or {}
    messages = st.session_state.get('patient_messages', [])
    recent_messages = [
        {'role': message.get('role'), 'content': str(message.get('content') or '')}
        for message in messages[-15:]
    ]
    last_user_message = next((item['content'] for item in reversed(recent_messages) if item['role'] == 'user'), '')
    previous_workflow = last_result.get('workflow_type')
    if hasattr(previous_workflow, 'value'):
        previous_workflow = previous_workflow.value
    return {
        'previous_workflow_type': previous_workflow,
        'previous_user_query': last_result.get('user_query') or last_user_message,
        'previous_final_response': last_result.get('final_response'),
        'active_patient_id': last_result.get('active_patient_id') or getattr(st.session_state.get('patient_profile'), 'patient_id', None),
        'last_appointment_slots': last_result.get('appointment_rows', []) or last_result.get('appointment_slots', []),
        'booking_context': last_result.get('booking_context'),
        'last_pending_action': getattr(last_result.get('pending_confirmation'), 'action_type', None),
        'recent_messages': recent_messages,
    }
def _run_patient_workflow(user_query: str, *, use_registration_fields: bool = False) -> None:
    result = _PATIENT_WORKFLOW.run(
        {
            'session_id': 'sess_patient_chat',
            'actor': ActorType.PATIENT,
            'active_patient_id': getattr(st.session_state.patient_profile, 'patient_id', None),
            'patient_phone': st.session_state.patient_phone or None,
            'conversation_context': _build_patient_conversation_context(),
            'registration_first_name': st.session_state.patient_registration_first_name if use_registration_fields else None,
            'registration_last_name': st.session_state.patient_registration_last_name if use_registration_fields else None,
            'registration_address': st.session_state.patient_registration_address if use_registration_fields else None,
            'user_query': user_query,
        }
    )
    result = _normalize_result_for_display('patient', result)
    st.session_state.patient_last_result = result
    profile = result.get('patient_profile')
    if profile is None:
        patient_id = result.get('active_patient_id')
        if patient_id:
            profile = _PATIENTS.get_patient_by_id(patient_id)
        elif st.session_state.patient_phone:
            profile = _PATIENTS.get_patient_by_phone(st.session_state.patient_phone)
    if profile is not None:
        st.session_state.patient_profile = profile
        st.session_state.patient_phone = profile.phone or st.session_state.patient_phone
    if result.get('patient_resolution_status') == 'awaiting_registration':
        st.session_state.patient_pending_registration = True
        if result.get('patient_phone'):
            st.session_state.patient_phone = result['patient_phone']
        st.session_state.patient_pending_query = user_query
    else:
        st.session_state.patient_pending_registration = False
    if result.get('final_response'):
        _append_patient_message('assistant', result['final_response'])
    if result.get('pending_confirmation') and result['pending_confirmation'].status != 'confirmed':
        _append_patient_message('assistant', result['pending_confirmation'].prompt)


def _build_attendant_conversation_context() -> dict[str, object]:
    last_result = st.session_state.get('attendant_last_result') or {}
    messages = st.session_state.get('attendant_messages', [])
    recent_messages = [
        {'role': message.get('role'), 'content': str(message.get('content') or '')}
        for message in messages[-15:]
    ]
    return {
        'previous_workflow_type': last_result.get('workflow_type'),
        'previous_final_response': last_result.get('final_response'),
        'active_patient_id': last_result.get('active_patient_id'),
        'recent_messages': recent_messages,
        'selected_appointment_rows': last_result.get('appointment_rows', []),
        'uploaded_report_name': st.session_state.get('attendant_uploaded_report_name'),
        'uploaded_report_present': bool(st.session_state.get('attendant_uploaded_report_text')),
    }


def _read_uploaded_report(uploaded_file) -> tuple[str | None, str | None]:
    if uploaded_file is None:
        return None, None
    try:
        if uploaded_file.name.lower().endswith('.pdf'):
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            text = '\n'.join((page.extract_text() or '') for page in reader.pages).strip()
            return (text or None), uploaded_file.name
        text = uploaded_file.getvalue().decode('utf-8', errors='ignore').strip()
        return (text or None), uploaded_file.name
    except Exception:
        return None, uploaded_file.name


def _build_doctor_conversation_context() -> dict[str, object]:
    last_result = st.session_state.get('doctor_last_result') or {}
    messages = st.session_state.get('doctor_messages', [])
    recent_messages = [
        {'role': message.get('role'), 'content': str(message.get('content') or '')}
        for message in messages[-15:]
    ]
    return {
        'previous_workflow_type': last_result.get('workflow_type'),
        'previous_final_response': last_result.get('final_response'),
        'active_patient_id': last_result.get('active_patient_id'),
        'selected_appointment_id': last_result.get('selected_appointment_id'),
        'selected_doctor_id': st.session_state.get('doctor_selected_doctor_id'),
        'selected_doctor_name': st.session_state.get('doctor_selected_doctor_name'),
        'pending_confirmation': last_result.get('pending_confirmation'),
        'last_pending_action': (last_result.get('pending_confirmation') or {}).get('action_type'),
        'recent_messages': recent_messages,
    }


def _build_admin_conversation_context() -> dict[str, object]:
    last_result = st.session_state.get('admin_last_result') or {}
    messages = st.session_state.get('admin_messages', [])
    recent_messages = [
        {'role': message.get('role'), 'content': str(message.get('content') or '')}
        for message in messages[-15:]
    ]
    return {
        'previous_workflow_type': last_result.get('workflow_type'),
        'previous_final_response': last_result.get('final_response'),
        'recent_messages': recent_messages,
    }


def _render_patient_tab() -> None:
    _init_patient_state()
    st.subheader('Patient View')

    left_pane, right_pane = st.columns([0.9, 1.6], vertical_alignment='top')

    with left_pane:
        details = st.container(border=True)
        with details:
            st.markdown('**Patient Details**')
            profile = st.session_state.patient_profile
            profile_first_name, profile_last_name = _split_name(profile)
            profile_address = profile.address if profile is not None else ''
            display_suffix = profile.patient_id if profile is not None else 'empty'

            st.text_input('Phone Number', value=st.session_state.patient_phone, disabled=True, key=f'patient_phone_display_{display_suffix}')
            if st.session_state.patient_pending_registration:
                st.text_input('First Name', key='patient_registration_first_name')
                st.text_input('Last Name', key='patient_registration_last_name')
                st.text_area('Residential Address', key='patient_registration_address', height=100)
            else:
                st.text_input('First Name', value=profile_first_name, disabled=True, key=f'patient_first_name_display_{display_suffix}')
                st.text_input('Last Name', value=profile_last_name, disabled=True, key=f'patient_last_name_display_{display_suffix}')
                st.text_area('Residential Address', value=profile_address or '', disabled=True, height=100, key=f'patient_address_display_{display_suffix}')

            if st.session_state.patient_pending_registration:
                st.caption('Patient not found. Complete the details and continue registration.')
                if st.button('Complete Patient Registration', key='patient_registration_button'):
                    _run_patient_workflow(st.session_state.patient_pending_query or 'Hello', use_registration_fields=True)
                    st.rerun()
            elif profile is None:
                st.caption('Patient details will populate here after phone lookup.')

    with right_pane:
        chat = st.container(border=True)
        with chat:
            st.markdown('**Healthcare Assistant Chat**')
            transcript = st.container()
            with transcript:
                for message in st.session_state.patient_messages:
                    with st.chat_message(message['role']):
                        st.markdown(message['content'])
            user_message = st.chat_input('Type your message here', key='patient_chat_input')
            if user_message:
                _append_patient_message('user', user_message)
                if not st.session_state.patient_phone:
                    st.session_state.patient_phone = user_message
                st.session_state.patient_pending_user_query = user_message
                st.rerun()

        pending_user_query = st.session_state.get('patient_pending_user_query')
        if pending_user_query:
            st.session_state.patient_pending_user_query = None
            with st.spinner('Thinking...'):
                _run_patient_workflow(pending_user_query)
            st.rerun()

    result = st.session_state.patient_last_result
    if result:
        workflow_type = _workflow_name(result)
        if result.get('visit_history_rows'):
            title = 'Medical History'
            if workflow_type != 'show_medical_history':
                title = 'Past Visit Details'
            st.markdown(f'**{title}**')
            st.dataframe(result['visit_history_rows'], use_container_width=True, hide_index=True)
        if result.get('appointment_rows'):
            title = 'Appointments on Record'
            if workflow_type in {'book_appointment', 'amend_appointment'}:
                title = 'Scheduled Appointment Details'
            st.markdown(f'**{title}**')
            st.dataframe(result['appointment_rows'], use_container_width=True, hide_index=True)
        if result.get('open_appointment_rows'):
            st.markdown('**Open Appointments**')
            st.dataframe(result['open_appointment_rows'], use_container_width=True, hide_index=True)


def _render_attendant_tab() -> None:
    st.subheader('Attendant View')
    st.caption('Hello, Attendant. Open appointments are loaded below, and you can manage patient and appointment operations through chat.')
    st.session_state.setdefault('attendant_last_result', None)
    st.session_state.setdefault('attendant_request_text', _DEFAULT_QUERIES['attendant'])
    st.session_state.setdefault('attendant_pending_user_query', None)
    st.session_state.setdefault('attendant_uploaded_report_text', None)
    st.session_state.setdefault('attendant_uploaded_report_name', None)
    st.session_state.setdefault('attendant_report_uploader_nonce', 0)
    doctor_options = [doctor.full_name for doctor in _PATIENTS.list_doctors()]
    default_doctor = doctor_options[0] if doctor_options else None
    st.session_state.setdefault('attendant_selected_doctor', default_doctor)
    st.session_state.setdefault(
        'attendant_messages',
        [
            {
                'role': 'assistant',
                'content': (
                    'How may I help with patient or appointment operations today?\n\n'
                    'Examples:\n'
                    '- Show patient history for Rahul Negi\n'
                    '- Update Anjali Mehra history: doctor cleared the upper respiratory infection and symptoms have resolved\n'
                    '- Update Anjali Mehra history from the uploaded report\n'
                    '- Show all doctors\n'
                    '- Schedule an appointment for patient pat_rahul_negi\n'
                    '- Cancel appointments for Robert Nickle. Re-book the cancelled appointment for 2026-04-16 with James C\n'
                    '- Delete patient pat_rahul_negi'
                ),
            }
        ],
    )
    selected_doctor = st.selectbox(
        'Doctor',
        options=doctor_options,
        key='attendant_selected_doctor',
        placeholder='Select a doctor',
    )
    current_open_rows = (st.session_state.attendant_last_result or {}).get('open_appointment_rows')
    st.markdown('**Open Appointments**')
    if current_open_rows and 'show_open_appointments' in set((st.session_state.attendant_last_result or {}).get('completed_actions') or []):
        open_rows = current_open_rows
    else:
        open_rows = _appointment_rows_for_open_slots(selected_doctor)
    st.dataframe(open_rows, use_container_width=True, hide_index=True)

    uploaded_report = st.file_uploader(
        'Upload patient follow-up report',
        type=['pdf', 'txt'],
        key=f"attendant_report_uploader_{st.session_state.attendant_report_uploader_nonce}",
        help='Optional: upload a doctor report or follow-up note before asking the assistant to update patient history.',
    )
    if uploaded_report is not None:
        report_text, report_name = _read_uploaded_report(uploaded_report)
        st.session_state.attendant_uploaded_report_text = report_text
        st.session_state.attendant_uploaded_report_name = report_name
    if st.session_state.attendant_uploaded_report_name:
        st.caption(f"Report ready: {st.session_state.attendant_uploaded_report_name}")

    chat_container = st.container(border=True)
    with chat_container:
        st.markdown('**Attendant Operations Chat**')
        for message in st.session_state.attendant_messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        user_query = st.chat_input(
            'Type an attendant request here',
            key='request_attendant',
        )
        if user_query:
            st.session_state.attendant_request_text = user_query
            st.session_state.attendant_messages.append({'role': 'user', 'content': user_query})
            st.session_state.attendant_pending_user_query = user_query
            st.rerun()

    pending_user_query = st.session_state.get('attendant_pending_user_query')
    if pending_user_query:
        st.session_state.attendant_pending_user_query = None
        with st.spinner('Thinking...'):
            result = _normalize_result_for_display('attendant', _run_attendant_workflow(None, pending_user_query))
            st.session_state.attendant_last_result = result
            st.session_state.attendant_messages.append({'role': 'assistant', 'content': result.get('final_response') or ''})
            st.session_state.attendant_uploaded_report_text = None
            st.session_state.attendant_uploaded_report_name = None
            st.session_state.attendant_report_uploader_nonce += 1
        st.rerun()

    result = st.session_state.attendant_last_result
    if result:
        completed_actions = set(result.get('completed_actions') or [])
        if result.get('selected_patient'):
            render_patient_profile_summary(result.get('selected_patient'))
        if result.get('selected_doctor'):
            st.markdown('**Doctor Profile**')
            st.dataframe(
                [
                    {
                        'Doctor ID': result['selected_doctor'].doctor_id,
                        'Name': result['selected_doctor'].full_name,
                        'Specialty': result['selected_doctor'].specialty,
                        'Phone': result['selected_doctor'].phone or '',
                        'Email': result['selected_doctor'].email or '',
                        'Gender': result['selected_doctor'].gender or '',
                        'Clinic': result['selected_doctor'].clinic_name,
                    }
                ],
                use_container_width=True,
                hide_index=True,
            )
        if result.get('patient_directory_rows') and completed_actions.intersection({'show_active_patients', 'delete_patient'}):
            st.markdown('**Registered Patients**')
            st.dataframe(result['patient_directory_rows'], use_container_width=True, hide_index=True)
        if result.get('doctor_directory_rows') and completed_actions.intersection({'show_doctors', 'register_doctor', 'edit_doctor_details'}):
            st.markdown('**Doctor Registry**')
            st.dataframe(result['doctor_directory_rows'], use_container_width=True, hide_index=True)
        if result.get('patient_history_summary'):
            render_attendant_history_summary(result.get('patient_history_summary'))
        if result.get('patient_records'):
            render_patient_medical_history(result.get('patient_records', []))
        if result.get('appointment_rows'):
            title = result.get('appointment_table_title') or 'Appointments'
            st.markdown(f'**{title}**')
            st.dataframe(result['appointment_rows'], use_container_width=True, hide_index=True)
        if result.get('open_appointment_rows') and 'show_open_appointments' in completed_actions:
            st.markdown('**Current Open Appointments**')
            st.dataframe(result['open_appointment_rows'], use_container_width=True, hide_index=True)
        if result.get('batch_appointment_updates'):
            st.markdown('**Updated Doctor Schedule**')
            st.dataframe(result['batch_appointment_updates'], use_container_width=True, hide_index=True)


def _render_doctor_tab() -> None:
    st.subheader('Doctor View')
    st.session_state.setdefault('doctor_last_result', None)
    st.session_state.setdefault('doctor_selected_doctor_id', None)
    st.session_state.setdefault('doctor_selected_doctor_name', None)
    st.session_state.setdefault('doctor_pending_user_query', None)
    st.session_state.setdefault(
        'doctor_messages',
        [
            {
                'role': 'assistant',
                'content': (
                    'How may I help with schedule review, patient records, or symptom research today?\n\n'
                    'Examples:\n'
                    '- Show my appointments for today\n'
                    '- Show patient history for Rahul Negi\n'
                    '- Research treatment options for diabetes\n'
                    '- Please reschedule appointment appt_001 to 2026-04-20'
                ),
            }
        ],
    )
    if st.session_state.doctor_selected_doctor_id is None:
        st.caption('For demo, please select one of the available Doctor profiles. This can later be integrated with Identity management system to SSO into correct logged in user')
        doctor_options = [doctor.full_name for doctor in _PATIENTS.list_doctors()]
        selected_doctor_name = st.selectbox(
            'Doctor Profile',
            options=doctor_options,
            index=None,
            placeholder='Select a doctor profile',
            key='doctor_profile_selector',
        )
        if selected_doctor_name:
            selected_doctor_id = _APPOINTMENTS.resolve_doctor_id(selected_doctor_name)
            if selected_doctor_id:
                st.session_state.doctor_selected_doctor_id = selected_doctor_id
                st.session_state.doctor_selected_doctor_name = selected_doctor_name
                st.session_state.doctor_last_result = None
                st.session_state.doctor_messages = [
                    {
                        'role': 'assistant',
                        'content': (
                            f'Welcome, Dr. {selected_doctor_name}. How may I assist you with schedule review, patient records, or symptom research today?\n\n'
                            'Examples:\n'
                            '- Show my appointments for today\n'
                            '- Show patient history for Rahul Negi\n'
                            '- Research treatment options for diabetes\n'
                            '- Please reschedule appointment appt_001 to 2026-04-20'
                        ),
                    }
                ]
                st.rerun()
        return

    selected_doctor_id = st.session_state.doctor_selected_doctor_id
    selected_doctor_name = st.session_state.doctor_selected_doctor_name or _APPOINTMENTS.get_doctor_display_name(selected_doctor_id)
    st.caption(f'Logged in user: Dr. {selected_doctor_name}')
    st.caption(f'Dr. {selected_doctor_name}, the schedule board and appointment operations below are limited to your profile.')

    current_result = st.session_state.doctor_last_result or {}
    current_workflow = _workflow_name(current_result)
    st.markdown('**Schedule Board**')
    schedule_rows = current_result.get('appointment_rows') if current_workflow == 'show_schedule' else None
    if not schedule_rows:
        schedule_rows = _default_doctor_schedule_rows(selected_doctor_id)
    if schedule_rows:
        st.dataframe(schedule_rows, use_container_width=True, hide_index=True)
    else:
        st.caption('No appointments available.')

    chat = st.container(border=True)
    with chat:
        st.markdown('**Doctor Assistant Chat**')
        for message in st.session_state.doctor_messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        user_query = st.chat_input('Type a doctor request here', key='doctor_chat_input')
        if user_query:
            st.session_state.doctor_messages.append({'role': 'user', 'content': user_query})
            st.session_state.doctor_pending_user_query = user_query
            st.rerun()

    pending_user_query = st.session_state.get('doctor_pending_user_query')
    if pending_user_query:
        st.session_state.doctor_pending_user_query = None
        with st.spinner('Thinking...'):
            result = _normalize_result_for_display(
                'doctor',
                _run_doctor_workflow(
                    user_query=pending_user_query,
                    selected_doctor_id=selected_doctor_id,
                    selected_appointment_id=None,
                    active_patient_id=None,
                    patient_name_query=None,
                    patient_phone_query=None,
                    schedule_date_query=None,
                    reschedule_date=None,
                ),
            )
            st.session_state.doctor_last_result = result
            st.session_state.doctor_messages.append({'role': 'assistant', 'content': result.get('final_response') or ''})
        st.rerun()

    result = st.session_state.doctor_last_result
    if result:
        workflow_type = _workflow_name(result)
        lower_left, lower_right = st.columns([1, 1])
        with lower_left:
            if result.get('selected_patient') and workflow_type in {'view_patient_history', 'amend_appointment', 'cancel_appointment'}:
                render_patient_profile_summary(result.get('selected_patient'))
            if result.get('patient_history_summary'):
                render_attendant_history_summary(result.get('patient_history_summary'))
            if result.get('patient_records'):
                render_patient_medical_history(result.get('patient_records', []))
        with lower_right:
            if result.get('appointment_rows') and workflow_type in {'show_schedule', 'view_patient_history', 'amend_appointment', 'cancel_appointment'}:
                if workflow_type == 'show_schedule':
                    st.markdown('**Appointment Details**')
                render_doctor_appointments(
                    [
                        {
                            'appointment_id': row.get('Appointment ID'),
                            'appointment_date': row.get('Date'),
                            'appointment_time': row.get('Time'),
                            'specialty': row.get('Specialty'),
                            'patient_id': row.get('Patient ID'),
                            'status': row.get('Status'),
                        }
                        for row in result.get('appointment_rows', [])
                    ]
                )
            if result.get('external_summary'):
                render_medlineplus_section(None, result.get('external_summary'))


def _render_admin_tab() -> None:
    st.subheader('IT Admin View')
    st.caption('Use the menu below to inspect logs, planner traces, system errors, and LangSmith runs for today.')
    st.session_state.setdefault('admin_last_result', None)
    st.session_state.setdefault('admin_messages', [{'role': 'assistant', 'content': 'Select an admin view or ask for logs, traces, errors, or LangSmith runs.'}])
    st.session_state.setdefault('admin_pending_user_query', None)

    left, right = st.columns([0.8, 1.2])
    with left:
        selected_view = st.selectbox(
            'Admin menu',
            options=['Interaction Logs', 'Planner Traces', 'System Errors', 'LangSmith Runs'],
            key='admin_selected_view',
        )
        actor_filter = st.selectbox(
            'Actor filter',
            options=['', 'patient', 'doctor', 'attendant'],
            format_func=lambda value: 'All actors' if value == '' else value,
            key='admin_actor_filter',
        )
        if st.button('Load selected view', key='admin_load_view'):
            result = _normalize_result_for_display(
                'admin',
                _run_admin_workflow(
                user_query=f'Please show {selected_view.lower()}.',
                selected_view=selected_view,
                actor_filter=actor_filter or None,
                ),
            )
            st.session_state.admin_last_result = result
            st.session_state.admin_messages.append({'role': 'assistant', 'content': result.get('final_response') or ''})
            st.rerun()
    with right:
        chat = st.container(border=True)
        with chat:
            st.markdown('**Admin Assistant Chat**')
            for message in st.session_state.admin_messages:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])
            user_query = st.chat_input('Type an admin request here', key='admin_chat_input')
            if user_query:
                st.session_state.admin_messages.append({'role': 'user', 'content': user_query})
                st.session_state.admin_pending_user_query = user_query
                st.rerun()

    pending_user_query = st.session_state.get('admin_pending_user_query')
    if pending_user_query:
        st.session_state.admin_pending_user_query = None
        with st.spinner('Thinking...'):
            result = _normalize_result_for_display(
                'admin',
                _run_admin_workflow(
                    user_query=pending_user_query,
                    selected_view=st.session_state.get('admin_selected_view'),
                    actor_filter=st.session_state.get('admin_actor_filter') or None,
                ),
            )
            st.session_state.admin_last_result = result
            st.session_state.admin_messages.append({'role': 'assistant', 'content': result.get('final_response') or ''})
        st.rerun()

    result = st.session_state.admin_last_result
    if result:
        if result.get('interaction_rows'):
            st.markdown('**Interaction Logs**')
            st.dataframe(
                [{k: v for k, v in row.items() if k != 'Context'} for row in result['interaction_rows']],
                use_container_width=True,
                hide_index=True,
            )
        if result.get('planner_trace_rows'):
            st.markdown('**Planner Traces**')
            st.dataframe(
                [{k: v for k, v in row.items() if k not in {'Context', 'Planner Output'}} for row in result['planner_trace_rows']],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander('Planner Trace Details'):
                st.json(result['planner_trace_rows'])
        if result.get('system_error_rows'):
            st.markdown('**System Errors**')
            st.dataframe(
                [{k: v for k, v in row.items() if k != 'Stack Trace'} for row in result['system_error_rows']],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander('System Error Details'):
                st.json(result['system_error_rows'])
        if result.get('langsmith_run_rows'):
            st.markdown('**LangSmith Runs**')
            st.dataframe(result['langsmith_run_rows'], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title='Healthcare Assistant', layout='wide')
    title_col, action_col = st.columns([0.85, 0.15], vertical_alignment='center')
    with title_col:
        st.title('Agentic Healthcare Assistant')
    with action_col:
        if st.button('Reset Session', key='reset_application_session', use_container_width=True):
            _reset_application_session()
    st.caption('Role-based workflow demo with Chroma RAG and MedlinePlus integration.')
    actor = st.selectbox(
        'Select Actor',
        options=['Patient', 'Attendant', 'Doctor', 'IT Admin'],
        index=None,
        placeholder='Choose an actor profile',
        key='selected_actor_view',
    )

    if actor is None:
        st.subheader('Landing Page')
        st.write(
            'This is a default landing page, In final product the application will have role based login and Auth. '
            'Please select a role based tab to interact with the Healthcare Assistant'
        )
        return

    if actor == 'Patient':
        _render_patient_tab()
    elif actor == 'Attendant':
        _render_attendant_tab()
    elif actor == 'Doctor':
        _render_doctor_tab()
    elif actor == 'IT Admin':
        _render_admin_tab()


if __name__ == '__main__':
    main()
