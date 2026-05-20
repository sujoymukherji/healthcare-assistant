from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import date, time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.repositories.appointment_repository import get_appointment_repository
from app.repositories.sample_data_repository import get_sample_repository
from app.schemas.domain import ActorType, Doctor, MedicalRecordEntry, Patient
from app.services.attendant_intent_service import AttendantIntentDecision, AttendantTask, attendant_intent_service
from app.services.patient_history_update_service import patient_history_update_service
from app.services.attendant_response_generator import attendant_response_generator
from app.services.langsmith_service import langsmith_service


class AttendantWorkflowState(TypedDict, total=False):
    session_id: str
    actor: ActorType
    user_query: str
    active_patient_id: str | None
    conversation_context: dict[str, object] | None
    selected_patient: Patient | None
    selected_doctor: Doctor | None
    patient_records: list[MedicalRecordEntry]
    patient_history_summary: dict[str, object] | None
    patient_directory_rows: list[dict[str, str]]
    doctor_directory_rows: list[dict[str, str]]
    appointment_rows: list[dict[str, str]]
    appointment_table_title: str | None
    open_appointment_rows: list[dict[str, str]]
    batch_appointment_updates: list[dict[str, str]]
    task_outcomes: list[dict[str, object]]
    final_response: str | None
    workflow_type: str | None
    intent_decision: AttendantIntentDecision | None
    planner_source: str | None
    completed_actions: list[str]
    validation_errors: list[str]
    uploaded_report_text: str | None
    uploaded_report_name: str | None
    langsmith_enabled: bool
    langsmith_run_id: str | None
    langsmith_run_url: str | None


class AttendantAssistantWorkflow:
    def __init__(self) -> None:
        self.repository = get_sample_repository()
        self.appointments = get_appointment_repository()
        self.graph = self._build_graph()

    def run(self, state: AttendantWorkflowState) -> AttendantWorkflowState:
        state.setdefault('actor', ActorType.ATTENDANT)
        state['langsmith_enabled'] = langsmith_service.enabled
        try:
            run_id: str | None = None
            with langsmith_service.trace_context(
                'attendant_workflow',
                run_type='chain',
                inputs={'query': state.get('user_query'), 'patient_id': state.get('active_patient_id')},
                metadata={'session_id': state.get('session_id')},
                tags=['healthcare-assistant', 'attendant-workflow'],
            ) as run:
                if run is not None:
                    run_id = str(run.id)
                result = self.graph.invoke(state)
            if run_id:
                result['langsmith_run_id'] = run_id
                langsmith_service.flush()
                result['langsmith_run_url'] = langsmith_service.get_verified_run_url(run_id)
                self.repository.log_interaction(
                    actor='attendant',
                    session_id=result.get('session_id', 'sess_attendant'),
                    user_message=result.get('user_query', ''),
                    assistant_message=result.get('final_response'),
                    workflow_type=result.get('workflow_type'),
                    context=result.get('conversation_context'),
                )
                if result.get('langsmith_run_id') and result.get('langsmith_run_url'):
                    self.repository.log_langsmith_run(
                        session_id=result.get('session_id', 'sess_attendant'),
                        actor='attendant',
                        workflow_type=result.get('workflow_type'),
                        run_id=result['langsmith_run_id'],
                        trace_url=result.get('langsmith_run_url'),
                    )
            return result
        except Exception as error:
            self.repository.log_system_error(
                session_id=state.get('session_id', 'sess_attendant'),
                actor='attendant',
                stage='attendant_workflow',
                error=error,
            )
            raise

    def stream(self, state: AttendantWorkflowState) -> Iterator[AttendantWorkflowState]:
        yield from self.graph.stream(state, stream_mode='values')

    def _build_graph(self):
        builder = StateGraph(AttendantWorkflowState)
        builder.add_node('initialize', self._initialize)
        builder.add_node('classify_intent', self._classify_intent)
        builder.add_node('execute_tasks', self._execute_tasks)

        builder.add_edge(START, 'initialize')
        builder.add_edge('initialize', 'classify_intent')
        builder.add_edge('classify_intent', 'execute_tasks')
        builder.add_edge('execute_tasks', END)
        return builder.compile()

    def _initialize(self, state: AttendantWorkflowState) -> AttendantWorkflowState:
        state.setdefault('conversation_context', {})
        state['selected_patient'] = None
        state['selected_doctor'] = None
        state['patient_records'] = []
        state['patient_history_summary'] = None
        state['patient_directory_rows'] = []
        state['doctor_directory_rows'] = []
        state['appointment_rows'] = []
        state['appointment_table_title'] = None
        state['open_appointment_rows'] = self._build_open_appointment_rows()
        state['batch_appointment_updates'] = []
        state['task_outcomes'] = []
        state['workflow_type'] = None
        state['planner_source'] = None
        state['completed_actions'] = []
        state['validation_errors'] = []
        state.setdefault('uploaded_report_text', None)
        state.setdefault('uploaded_report_name', None)
        return state

    def _classify_intent(self, state: AttendantWorkflowState) -> AttendantWorkflowState:
        query = state.get('user_query', '')
        context = state.get('conversation_context') or {}
        decision = attendant_intent_service.classify(query, json.dumps(context, sort_keys=True))
        decision = AttendantIntentDecision(
            tasks=[self._normalize_task(task, user_query=query) for task in decision.tasks],
            rationale=decision.rationale,
        )
        state['intent_decision'] = decision
        state['workflow_type'] = 'execute_attendant_tasks'
        state['planner_source'] = 'llm_or_fallback'
        self.repository.log_planner_trace(
            actor='attendant',
            session_id=state.get('session_id', 'sess_attendant'),
            user_message=query,
            context=context,
            planner_output=decision.model_dump(mode='json'),
            final_workflow_type='execute_attendant_tasks',
        )
        return state

    def _normalize_task(self, task: AttendantTask, *, user_query: str) -> AttendantTask:
        updates: dict[str, object] = {}
        if (
            task.action in {'cancel_appointments', 'reschedule_appointments', 'book_appointments'}
            and not task.doctor_query
            and not task.target_doctor_query
            and task.patient_name
            and self.appointments.resolve_doctor_id(task.patient_name)
            and not self.repository.search_patients_by_name(task.patient_name)
        ):
            updates['doctor_query'] = task.patient_name
            updates['patient_name'] = None
            if task.action in {'cancel_appointments', 'reschedule_appointments'}:
                updates['target_scope'] = 'batch'
        if not re.search(r'\b20\d{2}\b', user_query):
            max_year = max((appointment.appointment_date.year for appointment in self.appointments.list_doctor_schedule()), default=2026)
            for field in ('target_date', 'target_end_date', 'source_date'):
                value = getattr(task, field)
                if value:
                    parsed = date.fromisoformat(value)
                    if parsed.year < max_year:
                        updates[field] = parsed.replace(year=max_year).isoformat()
        return task.model_copy(update=updates)

    def _execute_tasks(self, state: AttendantWorkflowState) -> AttendantWorkflowState:
        decision = state.get('intent_decision') or AttendantIntentDecision()
        ordered_tasks = self._order_tasks(decision.tasks)
        validation_errors = self._validate_tasks(ordered_tasks)
        if validation_errors:
            state['validation_errors'] = validation_errors
            state['final_response'] = attendant_response_generator.synthesize(
                workflow_type='execute_attendant_tasks',
                payload_json=json.dumps({'validation_errors': validation_errors, 'task_outcomes': []}),
            )
            return state

        prior_cancelled: list[dict[str, str]] = []

        for task in ordered_tasks:
            if task.action == 'show_open_appointments':
                state['open_appointment_rows'] = self._build_open_appointment_rows(
                    target_date=task.target_date,
                    target_end_date=task.target_end_date,
                    target_time=task.target_time,
                    doctor_query=task.doctor_query,
                )
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'count': len(state['open_appointment_rows'])})
                continue

            if task.action == 'show_booked_appointments':
                state['selected_patient'] = None
                state['selected_doctor'] = None
                state['patient_history_summary'] = None
                state['patient_records'] = []
                state['appointment_table_title'] = 'Booked Appointments'
                state['appointment_rows'] = self._build_booked_appointment_rows(
                    target_date=task.target_date,
                    target_end_date=task.target_end_date,
                    doctor_query=task.doctor_query,
                )
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'count': len(state['appointment_rows'])})
                continue

            if task.action == 'show_active_patients':
                state['selected_patient'] = None
                state['selected_doctor'] = None
                state['patient_directory_rows'] = self._build_patient_directory_rows()
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'count': len(state['patient_directory_rows'])})
                continue

            if task.action == 'show_doctors':
                state['selected_patient'] = None
                state['doctor_directory_rows'] = self._build_doctor_directory_rows()
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'count': len(state['doctor_directory_rows'])})
                continue

            if task.action == 'view_patient_history':
                patient = self._resolve_patient(task, state)
                if patient is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'Please specify which patient to review by sharing the patient_id, phone number, or full name.'})
                    continue
                state['selected_patient'] = patient
                state['active_patient_id'] = patient.patient_id
                state['patient_records'] = self.repository.get_records_for_patient(patient.patient_id)
                state['patient_history_summary'] = self.repository.build_patient_history_summary(patient.patient_id)
                state['appointment_table_title'] = 'Patient Appointments'
                state['appointment_rows'] = self._build_patient_appointment_rows(patient.patient_id)
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'patient_name': patient.full_name})
                continue

            if task.action == 'update_patient_history':
                patient = self._resolve_patient(task, state)
                if patient is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'Please specify which patient history to update by sharing the patient_id, phone number, or full name.'})
                    continue
                if not (task.history_update_text or state.get('uploaded_report_text')):
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'Please provide a follow-up note or upload a patient report to update the history.'})
                    continue
                updated_patient, _, updated_summary = self._update_patient_history(task, state, patient)
                if updated_patient is None or updated_summary is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'I could not update the patient history right now.'})
                    continue
                state['selected_patient'] = updated_patient
                state['active_patient_id'] = updated_patient.patient_id
                state['patient_records'] = self.repository.get_records_for_patient(updated_patient.patient_id)
                state['patient_history_summary'] = updated_summary
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append(
                    {
                        'action': task.action,
                        'status': 'success',
                        'patient_name': updated_patient.full_name,
                        'latest_visit_summary': updated_summary.get('latest_visit_summary'),
                        'summary_text': updated_summary.get('summary_text'),
                        'report_name': state.get('uploaded_report_name'),
                    }
                )
                continue

            if task.action == 'edit_patient_details':
                patient = self._resolve_patient(task, state)
                if patient is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'Please specify which patient to update by sharing the patient_id, phone number, or full name.'})
                    continue
                updated = self.repository.update_patient_details(
                    patient.patient_id,
                    first_name=task.edit_first_name,
                    last_name=task.edit_last_name,
                    address=task.edit_address,
                    phone=task.edit_phone,
                )
                if updated is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'I could not update the patient details right now.'})
                    continue
                state['selected_patient'] = updated
                state['active_patient_id'] = updated.patient_id
                state['patient_directory_rows'] = self._build_patient_directory_rows()
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'patient_name': updated.full_name})
                continue

            if task.action == 'delete_patient':
                patient = self._resolve_patient(task, state)
                if patient is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'Please specify which patient to delete by sharing the patient_id, phone number, or full name.'})
                    continue
                self.repository.delete_patient(patient.patient_id)
                state['selected_patient'] = None
                state['active_patient_id'] = None
                state['patient_directory_rows'] = self._build_patient_directory_rows()
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'patient_name': patient.full_name})
                continue

            if task.action == 'register_doctor':
                doctor = self.repository.register_doctor(
                    full_name=task.doctor_name or '',
                    specialty=task.doctor_specialty or 'General Medicine',
                    phone=task.doctor_phone,
                    email=task.doctor_email,
                    gender=task.doctor_gender,
                )
                state['selected_doctor'] = doctor
                state['doctor_directory_rows'] = self._build_doctor_directory_rows()
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'doctor_name': doctor.full_name, 'doctor_specialty': doctor.specialty})
                continue

            if task.action == 'edit_doctor_details':
                doctor = self._resolve_doctor(task)
                if doctor is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'Please specify which doctor to update by sharing the doctor name, doctor_id, phone number, or email.'})
                    continue
                updated = self.repository.update_doctor_details(
                    doctor.doctor_id,
                    full_name=task.doctor_name,
                    specialty=task.doctor_specialty,
                    phone=task.doctor_phone,
                    email=task.doctor_email,
                    gender=task.doctor_gender,
                )
                if updated is None:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'I could not update the doctor details right now.'})
                    continue
                state['selected_doctor'] = updated
                state['doctor_directory_rows'] = self._build_doctor_directory_rows()
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append({'action': task.action, 'status': 'success', 'doctor_name': updated.full_name})
                continue

            if task.action == 'cancel_appointments':
                cancelled_rows = self._cancel_appointments(task, state)
                if not cancelled_rows:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'I could not find any matching booked appointments to cancel.'})
                    continue
                prior_cancelled = cancelled_rows
                state['batch_appointment_updates'] = cancelled_rows if task.target_scope == 'batch' else []
                state['appointment_table_title'] = 'Updated Appointments'
                state['appointment_rows'] = cancelled_rows if task.target_scope == 'single' else []
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append(
                    {
                        'action': task.action,
                        'status': 'success',
                        'count': len(cancelled_rows),
                        'date': cancelled_rows[0]['Date'],
                        'time': cancelled_rows[0]['Time'],
                    }
                )
                continue

            if task.action == 'reschedule_appointments':
                updated_rows, option_rows = self._reschedule_appointments(task, state)
                if option_rows:
                    state['completed_actions'].append(task.action)
                    state['task_outcomes'].append(
                        {
                            'action': task.action,
                            'status': 'options',
                            'count': len(option_rows),
                            'target_date': task.target_date,
                            'target_end_date': task.target_end_date,
                            'target_time': task.target_time,
                            'doctor_query': task.doctor_query or task.target_doctor_query,
                        }
                    )
                    continue
                if not updated_rows:
                    state['task_outcomes'].append({'action': task.action, 'status': 'failed', 'error': 'I could not find any matching booked appointments to reschedule.'})
                    continue
                state['batch_appointment_updates'] = updated_rows if task.target_scope == 'batch' else []
                state['appointment_table_title'] = 'Updated Appointments'
                state['appointment_rows'] = updated_rows if task.target_scope == 'single' else []
                state['completed_actions'].append(task.action)
                state['task_outcomes'].append(
                    {
                        'action': task.action,
                        'status': 'success',
                        'count': len(updated_rows),
                        'target_date': task.target_date,
                        'target_end_date': task.target_end_date,
                        'target_time': task.target_time,
                    }
                )
                continue

            if task.action == 'book_appointments':
                booking_outcome = self._book_appointments(task, state, prior_cancelled=prior_cancelled)
                option_rows = booking_outcome.get('option_rows') or []
                if option_rows:
                    state['completed_actions'].append(task.action)
                    state['task_outcomes'].append(
                        {
                            'action': task.action,
                            'status': 'options',
                            'count': len(option_rows),
                            'target_date': task.target_date,
                            'target_end_date': task.target_end_date,
                            'target_time': task.target_time,
                            'doctor_query': task.doctor_query or task.target_doctor_query,
                        }
                    )
                    continue
                booked_rows = booking_outcome['booked_rows']
                if not booked_rows:
                    suggestion = self._format_future_open_slot_suggestion(
                        preferred_doctor=task.doctor_query or task.target_doctor_query
                    )
                    state['task_outcomes'].append(
                        {
                            'action': task.action,
                            'status': 'failed',
                            'requested_count': booking_outcome['requested_count'],
                            'booked_count': 0,
                            'unbooked_count': booking_outcome['requested_count'],
                            'target_date': task.target_date,
                            'target_end_date': task.target_end_date,
                            'target_time': task.target_time,
                            'doctor_query': task.doctor_query or task.target_doctor_query,
                            'suggestion': suggestion,
                        }
                    )
                    continue
                state['batch_appointment_updates'] = booked_rows if task.target_scope == 'batch' else []
                state['appointment_table_title'] = 'Booked Appointment Details'
                state['appointment_rows'] = booked_rows if task.target_scope == 'single' else []
                state['open_appointment_rows'] = self._build_open_appointment_rows()
                state['completed_actions'].append(task.action)
                unbooked_count = max(0, booking_outcome['requested_count'] - len(booked_rows))
                suggestion = self._format_future_open_slot_suggestion(
                    preferred_doctor=task.doctor_query or task.target_doctor_query
                )
                state['task_outcomes'].append(
                    {
                        'action': task.action,
                        'status': 'partial' if unbooked_count else 'success',
                        'requested_count': booking_outcome['requested_count'],
                        'booked_count': len(booked_rows),
                        'unbooked_count': unbooked_count,
                        'target_date': task.target_date,
                        'target_end_date': task.target_end_date,
                        'target_time': task.target_time,
                        'doctor_query': task.doctor_query or task.target_doctor_query,
                        'suggestion': suggestion,
                        'time': booked_rows[0]['Time'],
                    }
                )
                continue

        state['final_response'] = attendant_response_generator.synthesize(
            workflow_type='execute_attendant_tasks',
            payload_json=json.dumps({'task_outcomes': state['task_outcomes'], 'validation_errors': state['validation_errors']}),
        )
        return state

    def _update_patient_history(
        self,
        task: AttendantTask,
        state: AttendantWorkflowState,
        patient: Patient,
    ) -> tuple[Patient | None, MedicalRecordEntry | None, dict[str, object] | None]:
        current_summary = self.repository.build_patient_history_summary(patient.patient_id)
        report_text = state.get('uploaded_report_text') or ''
        report_name = state.get('uploaded_report_name') or ''
        update = patient_history_update_service.extract_update(
            patient_json=patient.model_dump_json(),
            current_summary_json=json.dumps(current_summary, default=str),
            user_query=task.history_update_text or state.get('user_query', ''),
            report_text=report_text,
            report_name=report_name,
        )
        doctor = self._resolve_doctor(task)
        source_type = 'uploaded_pdf' if report_text else 'manual_history_update'
        source_path = report_name or f'in-memory://history/{patient.patient_id}'
        visit_date = date.fromisoformat(update.visit_date) if update.visit_date else None
        return self.repository.apply_patient_history_update(
            patient.patient_id,
            title=update.title,
            subjective=update.subjective,
            objective=update.objective,
            assessment=update.assessment,
            plan=update.plan,
            visit_date=visit_date,
            source_type=source_type,
            source_path=source_path,
            doctor_id=doctor.doctor_id if doctor else None,
            primary_conditions=update.primary_conditions,
            chronic_conditions=update.chronic_conditions,
            allergies=update.allergies,
            cleared_conditions=update.cleared_conditions,
            latest_visit_summary=update.latest_visit_summary,
            summary_text=update.summary_text,
        )

    def _order_tasks(self, tasks: list[AttendantTask]) -> list[AttendantTask]:
        priority = {
            'show_open_appointments': 100,
            'show_booked_appointments': 100,
            'show_active_patients': 100,
            'show_doctors': 100,
            'view_patient_history': 90,
            'update_patient_history': 50,
            'edit_patient_details': 40,
            'delete_patient': 40,
            'register_doctor': 40,
            'edit_doctor_details': 40,
            'cancel_appointments': 10,
            'reschedule_appointments': 20,
            'book_appointments': 30,
            'general_help': 200,
        }
        indexed = list(enumerate(tasks))
        ordered = sorted(indexed, key=lambda item: (priority.get(item[1].action, 150), item[0]))
        return [task for _, task in ordered]

    def _validate_tasks(self, tasks: list[AttendantTask]) -> list[str]:
        errors: list[str] = []
        for task in tasks:
            for field in ('target_date', 'target_end_date', 'source_date'):
                value = getattr(task, field)
                parsed = date.fromisoformat(value) if value else None
                if parsed is not None and parsed.weekday() == 6:
                    errors.append(f'{value} is a Sunday. Please choose a date from Monday to Saturday.')
            doctor_query = task.doctor_query or task.target_doctor_query
            if doctor_query and self.appointments.resolve_doctor_id(doctor_query) is None:
                errors.append(f'I could not find doctor "{doctor_query}" in the registry.')
            if task.action == 'register_doctor':
                if not task.doctor_name or not task.doctor_specialty:
                    errors.append('Please provide both doctor name and specialty to register a doctor.')
            if task.action == 'edit_doctor_details':
                if not task.doctor_query:
                    errors.append('Please specify which doctor to update.')
            if task.action in {'book_appointments', 'reschedule_appointments'} and not task.target_date:
                errors.append('Please provide the target appointment date in YYYY-MM-DD format.')
            if task.action == 'reschedule_appointments' and task.target_scope == 'batch' and not task.doctor_query:
                errors.append('Please specify which doctor schedule should be rescheduled.')
        return errors

    def _resolve_patient(self, task: AttendantTask, state: AttendantWorkflowState) -> Patient | None:
        if task.patient_id:
            patient = self.repository.get_patient_by_id(task.patient_id)
            if patient is not None:
                return patient
        if task.patient_phone:
            patient = self.repository.get_patient_by_phone(task.patient_phone)
            if patient is not None:
                return patient
        if task.patient_name:
            matches = self.repository.search_patients_by_name(task.patient_name)
            if matches:
                return matches[0]
        active_patient_id = state.get('active_patient_id')
        return self.repository.get_patient_by_id(active_patient_id) if active_patient_id else None

    def _resolve_doctor(self, task: AttendantTask) -> Doctor | None:
        query = task.doctor_query or task.target_doctor_query or task.doctor_name or task.doctor_phone or task.doctor_email
        if not query:
            return None
        doctor_id = self.appointments.resolve_doctor_id(query)
        return self.repository.get_doctor_by_id(doctor_id) if doctor_id else None

    def _cancel_appointments(self, task: AttendantTask, state: AttendantWorkflowState) -> list[dict[str, str]]:
        appointments = self._matching_booked_appointments(task, state)
        cancelled_rows: list[dict[str, str]] = []
        for appointment in appointments:
            reopened = self.appointments.cancel_appointment(
                appointment.appointment_id,
                patient_id=appointment.patient_id,
                booked_by_actor=ActorType.ATTENDANT,
            )
            if reopened is None:
                continue
            cancelled_rows.append(
                {
                    'Appointment ID': appointment.appointment_id,
                    'Patient ID': appointment.patient_id,
                    'Date': appointment.appointment_date.isoformat(),
                    'Time': appointment.appointment_time.strftime('%H:%M'),
                    'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                    'Status': 'cancelled',
                }
            )
        return cancelled_rows

    def _reschedule_appointments(self, task: AttendantTask, state: AttendantWorkflowState) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        target = date.fromisoformat(task.target_date) if task.target_date else None
        if target is None:
            return [], []
        target_end = date.fromisoformat(task.target_end_date) if task.target_end_date else None
        target_time = self._parse_requested_time(task.target_time)
        if task.target_scope == 'batch' and task.doctor_query:
            if target_time is not None:
                matches = self.appointments.find_matching_open_slots(
                    target_date=target,
                    end_date=target_end,
                    doctor_query=task.doctor_query,
                    target_time=target_time,
                )
                rows = self._build_open_rows_from_slots(matches)
                state['appointment_table_title'] = 'Available Rebooking Options'
                state['appointment_rows'] = rows
                return [], rows
            if target_end is not None:
                matches = self.appointments.find_matching_open_slots(
                    target_date=target,
                    end_date=target_end,
                    doctor_query=task.doctor_query,
                    target_time=target_time,
                )
                rows = self._build_open_rows_from_slots(matches)
                state['appointment_table_title'] = 'Available Rebooking Options'
                state['appointment_rows'] = rows
                return [], rows
            source = date.fromisoformat(task.source_date) if task.source_date else target
            updated = self.appointments.bulk_reschedule_doctor_appointments(
                doctor_query=task.doctor_query,
                source_date=source,
                new_date=target,
                booked_by_actor=ActorType.ATTENDANT,
            )
        else:
            updated: list = []
            for appointment in self._matching_booked_appointments(task, state):
                preferred_doctor = task.doctor_query or task.target_doctor_query or self.appointments.get_doctor_display_name(appointment.doctor_id)
                if target_end is not None:
                    reopened, item = self.appointments.rebook_appointment_to_matching_available(
                        appointment.appointment_id,
                        patient_id=appointment.patient_id,
                        booking_reason=appointment.booking_reason,
                        booked_by_actor=ActorType.ATTENDANT,
                        target_date=target,
                        target_end_date=target_end,
                        doctor_query=preferred_doctor,
                        target_time=target_time,
                    )
                    if item is not None:
                        updated.append(item)
                        continue
                    matches = self.appointments.find_matching_open_slots(
                        target_date=target,
                        end_date=target_end,
                        doctor_query=preferred_doctor,
                        target_time=target_time,
                    )
                    rows = self._build_open_rows_from_slots(matches)
                    state['appointment_table_title'] = 'Available Rebooking Options'
                    state['appointment_rows'] = rows
                    return [], rows
                reopened, item = self.appointments.rebook_appointment(
                    appointment.appointment_id,
                    patient_id=appointment.patient_id,
                    booking_reason=appointment.booking_reason,
                    booked_by_actor=ActorType.ATTENDANT,
                    target_date=target,
                    doctor_query=preferred_doctor,
                    target_time=target_time,
                )
                if item is not None:
                    updated.append(item)
        return [
            {
                'Appointment ID': appointment.appointment_id,
                'Patient ID': appointment.patient_id,
                'Date': appointment.appointment_date.isoformat(),
                'Time': appointment.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                'Status': appointment.status.value,
            }
            for appointment in updated
        ], []

    def _book_appointments(
        self,
        task: AttendantTask,
        state: AttendantWorkflowState,
        *,
        prior_cancelled: list[dict[str, str]],
    ) -> dict[str, object]:
        target = date.fromisoformat(task.target_date) if task.target_date else None
        if target is None:
            return {'booked_rows': [], 'requested_count': 0}
        target_end = date.fromisoformat(task.target_end_date) if task.target_end_date else None
        target_time = self._parse_requested_time(task.target_time)

        patient_ids: list[str] = []
        if task.depends_on_previous and prior_cancelled:
            patient_ids = [row['Patient ID'] for row in prior_cancelled if row.get('Patient ID')]
        else:
            patient = self._resolve_patient(task, state)
            if patient is not None:
                patient_ids = [patient.patient_id]
        if target_end is not None:
            matches = self.appointments.find_matching_open_slots(
                target_date=target,
                end_date=target_end,
                doctor_query=task.doctor_query or task.target_doctor_query,
                target_time=target_time,
            )
            if not matches:
                self.appointments.ensure_open_slots_for_range(target, target_end)
                matches = self.appointments.find_matching_open_slots(
                    target_date=target,
                    end_date=target_end,
                    doctor_query=task.doctor_query or task.target_doctor_query,
                    target_time=target_time,
                )
            option_rows = self._build_open_rows_from_slots(matches)
            state['appointment_table_title'] = 'Available Booking Options'
            state['appointment_rows'] = option_rows
            return {'booked_rows': [], 'requested_count': len(patient_ids), 'option_rows': option_rows}
        booked_rows: list[dict[str, str]] = []
        for patient_id in patient_ids:
            booked = self.appointments.schedule_matching_available_for_patient(
                patient_id=patient_id,
                booking_reason=state.get('user_query', 'Attendant booked appointment'),
                booked_by_actor=ActorType.ATTENDANT,
                target_date=target,
                doctor_query=task.doctor_query or task.target_doctor_query,
                target_time=target_time,
            )
            if booked is None:
                continue
            booked_rows.append(self._build_booked_row(booked))
        return {'booked_rows': booked_rows, 'requested_count': len(patient_ids)}

    def _format_future_open_slot_suggestion(self, *, preferred_doctor: str | None) -> str:
        future_slots = self.appointments.find_matching_open_slots(doctor_query=preferred_doctor)
        if not future_slots:
            future_slots = self.appointments.list_open_appointments()
        if not future_slots:
            return 'Please try a different date once more slots are available.'
        unique_slots: list[str] = []
        for slot in future_slots[:3]:
            unique_slots.append(
                f"{slot.appointment_date.isoformat()} at {slot.appointment_time.strftime('%H:%M')} with {self.appointments.get_doctor_display_name(slot.doctor_id)}"
            )
        slot_text = '; '.join(unique_slots)
        return f'Please try a different date, or use one of the next available slots: {slot_text}.'

    def _matching_booked_appointments(self, task: AttendantTask, state: AttendantWorkflowState):
        if task.appointment_id:
            appointment = self.appointments.get_appointment_by_id(task.appointment_id)
            return [appointment] if appointment and appointment.status.value in {'booked', 'held'} else []
        if task.target_scope == 'batch' and task.doctor_query:
            if task.source_date:
                return self.appointments.list_doctor_appointments_for_date(task.doctor_query, date.fromisoformat(task.source_date))
            return self.appointments.list_doctor_appointments(task.doctor_query)
        patient = self._resolve_patient(task, state)
        if patient is not None:
            state['selected_patient'] = patient
            state['active_patient_id'] = patient.patient_id
            return self.appointments.list_current_patient_appointments(patient.patient_id)
        return []

    def _build_open_appointment_rows(
        self,
        *,
        target_date: str | None = None,
        target_end_date: str | None = None,
        target_time: str | None = None,
        doctor_query: str | None = None,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        parsed_target_time = self._parse_requested_time(target_time)
        if target_date and target_end_date:
            appointments = self.appointments.find_matching_open_slots(
                target_date=date.fromisoformat(target_date),
                end_date=date.fromisoformat(target_end_date),
                doctor_query=doctor_query,
                target_time=parsed_target_time,
            )
        elif target_date:
            appointments = self.appointments.find_matching_open_slots(
                target_date=date.fromisoformat(target_date),
                doctor_query=doctor_query,
                target_time=parsed_target_time,
            )
        else:
            appointments = self.appointments.find_matching_open_slots(doctor_query=doctor_query, target_time=parsed_target_time)
        for appointment in appointments:
            rows.append(
                {
                    'Appointment ID': appointment.appointment_id,
                    'Date': appointment.appointment_date.isoformat(),
                    'Time': appointment.appointment_time.strftime('%H:%M'),
                    'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                    'Specialty': appointment.specialty.replace('_', ' '),
                    'Status': appointment.status.value,
                }
            )
        return rows

    def _build_booked_appointment_rows(
        self,
        *,
        target_date: str | None = None,
        target_end_date: str | None = None,
        doctor_query: str | None = None,
    ) -> list[dict[str, str]]:
        if doctor_query and target_date and target_end_date:
            appointments = self.appointments.list_doctor_appointments_for_range(
                doctor_query,
                date.fromisoformat(target_date),
                date.fromisoformat(target_end_date),
            )
        elif doctor_query and target_date:
            appointments = self.appointments.list_doctor_appointments_for_date(
                doctor_query,
                date.fromisoformat(target_date),
            )
        elif target_date and target_end_date:
            appointments = self.appointments.list_booked_appointments_for_range(
                date.fromisoformat(target_date),
                date.fromisoformat(target_end_date),
            )
        elif target_date:
            appointments = self.appointments.list_booked_appointments_for_date(date.fromisoformat(target_date))
        elif doctor_query:
            appointments = self.appointments.list_doctor_appointments(doctor_query)
        else:
            appointments = self.appointments.list_booked_appointments()
        return [self._build_booked_row(appointment) for appointment in appointments]

    def _build_patient_directory_rows(self) -> list[dict[str, str]]:
        return [
            {
                'Patient ID': patient.patient_id,
                'Name': patient.full_name,
                'Phone': patient.phone or '',
                'Address': patient.address or '',
            }
            for patient in self.repository.list_patients()
        ]

    def _build_doctor_directory_rows(self) -> list[dict[str, str]]:
        return [
            {
                'Doctor ID': doctor.doctor_id,
                'Name': doctor.full_name,
                'Specialty': doctor.specialty,
                'Phone': doctor.phone or '',
                'Email': doctor.email or '',
                'Gender': doctor.gender or '',
                'Clinic': doctor.clinic_name,
            }
            for doctor in self.repository.list_doctors()
        ]

    def _build_patient_appointment_rows(self, patient_id: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for appointment in self.appointments.list_patient_appointments(patient_id):
            rows.append(self._build_booked_row(appointment))
        return rows

    def _build_booked_row(self, appointment) -> dict[str, str]:
        return {
            'Appointment ID': appointment.appointment_id,
            'Patient ID': appointment.patient_id or 'unassigned',
            'Date': appointment.appointment_date.isoformat(),
            'Time': appointment.appointment_time.strftime('%H:%M'),
            'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
            'Specialty': appointment.specialty.replace('_', ' '),
            'Status': appointment.status.value,
            'Reason': appointment.booking_reason,
        }

    def _build_open_rows_from_slots(self, appointments: list) -> list[dict[str, str]]:
        return [
            {
                'Appointment ID': appointment.appointment_id,
                'Date': appointment.appointment_date.isoformat(),
                'Time': appointment.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                'Specialty': appointment.specialty.replace('_', ' '),
                'Status': appointment.status.value,
            }
            for appointment in appointments
        ]

    def _parse_requested_time(self, value: str | None) -> time | None:
        if not value:
            return None
        try:
            return time.fromisoformat(value)
        except ValueError:
            return None
