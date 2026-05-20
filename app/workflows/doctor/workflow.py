from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, time, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.repositories.appointment_repository import get_appointment_repository
from app.repositories.sample_data_repository import get_sample_repository
from app.schemas.domain import ActorType, Appointment, MedicalRecordEntry, Patient, RetrievedEvidence
from app.services.chroma_retrieval import ChromaRetrievalService
from app.services.doctor_intent_service import DoctorIntentDecision, doctor_intent_service
from app.services.doctor_response_generator import doctor_response_generator
from app.services.external_summary_generator import build_medline_payload, external_summary_generator
from app.services.langsmith_service import langsmith_service
from app.services.medlineplus_service import MedlinePlusService


class DoctorWorkflowState(TypedDict, total=False):
    session_id: str
    actor: ActorType
    user_query: str
    selected_doctor_id: str | None
    selected_appointment_id: str | None
    active_patient_id: str | None
    patient_name_query: str | None
    patient_phone_query: str | None
    schedule_date_query: str | None
    reschedule_date: str | None
    conversation_context: dict[str, object] | None
    selected_patient: Patient | None
    selected_appointment: Appointment | None
    selected_doctor_name: str | None
    appointment_rows: list[dict[str, str]]
    patient_records: list[MedicalRecordEntry]
    patient_history_summary: dict[str, object] | None
    history_rag_results: list[RetrievedEvidence]
    memory_rag_results: list[RetrievedEvidence]
    external_summary: str | None
    medline_payload: str | None
    final_response: str | None
    workflow_type: str | None
    intent_decision: DoctorIntentDecision | None
    planner_source: str | None
    pending_confirmation: dict[str, object] | None
    langsmith_enabled: bool
    langsmith_run_id: str | None
    langsmith_run_url: str | None


class DoctorAssistantWorkflow:
    def __init__(self) -> None:
        self.repository = get_sample_repository()
        self.appointments = get_appointment_repository()
        self.retrieval = ChromaRetrievalService()
        self.medline = MedlinePlusService()
        self.graph = self._build_graph()

    def run(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        state.setdefault('actor', ActorType.DOCTOR)
        state['langsmith_enabled'] = langsmith_service.enabled
        try:
            run_id: str | None = None
            with langsmith_service.trace_context(
                'doctor_workflow',
                run_type='chain',
                inputs={'query': state.get('user_query'), 'patient_id': state.get('active_patient_id')},
                metadata={'session_id': state.get('session_id')},
                tags=['healthcare-assistant', 'doctor-workflow'],
            ) as run:
                if run is not None:
                    run_id = str(run.id)
                result = self.graph.invoke(state)
            if run_id:
                result['langsmith_run_id'] = run_id
                langsmith_service.flush()
                result['langsmith_run_url'] = langsmith_service.get_verified_run_url(run_id)
                self.repository.log_interaction(
                    actor='doctor',
                    session_id=result.get('session_id', 'sess_doctor'),
                    user_message=result.get('user_query', ''),
                    assistant_message=result.get('final_response'),
                    workflow_type=result.get('workflow_type'),
                    context=result.get('conversation_context'),
                )
                if result.get('langsmith_run_id') and result.get('langsmith_run_url'):
                    self.repository.log_langsmith_run(
                        session_id=result.get('session_id', 'sess_doctor'),
                        actor='doctor',
                        workflow_type=result.get('workflow_type'),
                        run_id=result['langsmith_run_id'],
                        trace_url=result.get('langsmith_run_url'),
                    )
            return result
        except Exception as error:
            self.repository.log_system_error(
                session_id=state.get('session_id', 'sess_doctor'),
                actor='doctor',
                stage='doctor_workflow',
                error=error,
            )
            raise

    def stream(self, state: DoctorWorkflowState) -> Iterator[DoctorWorkflowState]:
        yield from self.graph.stream(state, stream_mode='values')

    def _build_graph(self):
        builder = StateGraph(DoctorWorkflowState)
        builder.add_node('initialize', self._initialize)
        builder.add_node('classify_intent', self._classify_intent)
        builder.add_node('show_schedule', self._show_schedule)
        builder.add_node('view_patient_history', self._view_patient_history)
        builder.add_node('research_symptoms', self._research_symptoms)
        builder.add_node('amend_appointment', self._amend_appointment)
        builder.add_node('cancel_appointment', self._cancel_appointment)

        builder.add_edge(START, 'initialize')
        builder.add_edge('initialize', 'classify_intent')
        builder.add_conditional_edges(
            'classify_intent',
            self._route_after_intent,
            {
                'show_schedule': 'show_schedule',
                'view_patient_history': 'view_patient_history',
                'research_symptoms': 'research_symptoms',
                'amend_appointment': 'amend_appointment',
                'cancel_appointment': 'cancel_appointment',
                'done': END,
            },
        )
        for node in ('show_schedule', 'view_patient_history', 'research_symptoms', 'amend_appointment', 'cancel_appointment'):
            builder.add_edge(node, END)
        return builder.compile()

    def _initialize(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        state.setdefault('conversation_context', {})
        state.setdefault('appointment_rows', [])
        state.setdefault('patient_records', [])
        state.setdefault('patient_history_summary', None)
        state.setdefault('history_rag_results', [])
        state.setdefault('memory_rag_results', [])
        state.setdefault('external_summary', None)
        state.setdefault('workflow_type', None)
        state.setdefault('planner_source', None)
        state.setdefault('pending_confirmation', None)
        selected_doctor_id = state.get('selected_doctor_id')
        state['selected_doctor_name'] = self.appointments.get_doctor_display_name(selected_doctor_id) if selected_doctor_id else None
        state['selected_appointment'] = self.appointments.get_appointment_by_id(state.get('selected_appointment_id') or '')
        state['selected_patient'] = self._resolve_selected_patient(state)
        return state

    def _classify_intent(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        query = state.get('user_query', '')
        context = state.get('conversation_context') or {}
        decision = doctor_intent_service.classify(query, json.dumps(context, sort_keys=True))
        if state.get('schedule_date_query') and not decision.lookup_date:
            decision.lookup_date = state['schedule_date_query']
        if state.get('reschedule_date') and not decision.target_date:
            decision.target_date = state['reschedule_date']
        if state.get('active_patient_id') and not decision.patient_id:
            decision.patient_id = state['active_patient_id']
        if state.get('selected_appointment_id') and not decision.appointment_id:
            decision.appointment_id = state['selected_appointment_id']
        if state.get('selected_doctor_id') and not decision.doctor_id:
            decision.doctor_id = state['selected_doctor_id']
        if decision.appointment_id and state.get('selected_appointment') is None:
            state['selected_appointment'] = self.appointments.get_appointment_by_id(decision.appointment_id)
        state['intent_decision'] = decision
        state['workflow_type'] = decision.intent
        state['planner_source'] = 'llm_or_fallback'
        self.repository.log_planner_trace(
            actor='doctor',
            session_id=state.get('session_id', 'sess_doctor'),
            user_message=query,
            context=context,
            planner_output=decision.model_dump(mode='json'),
            final_workflow_type=decision.intent,
        )
        return state

    def _show_schedule(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        decision = state.get('intent_decision')
        selected_doctor_id = state.get('selected_doctor_id')
        if not selected_doctor_id:
            state['final_response'] = 'Please select a doctor profile to continue.'
            return state
        access_error = self._doctor_scope_error(state)
        if access_error:
            state['final_response'] = access_error
            return state
        requested_date = date.fromisoformat(decision.lookup_date) if decision and decision.lookup_date else None
        selected_doctor_name = state.get('selected_doctor_name') or self.appointments.get_doctor_display_name(selected_doctor_id)
        if requested_date is not None:
            appointments = self.appointments.list_doctor_appointments_for_date(selected_doctor_id, requested_date)
            state['appointment_rows'] = self._format_appointments(appointments)
            if appointments:
                message = f"Appointments for {selected_doctor_name} on {requested_date.isoformat()}."
            else:
                message = f"There are no appointments scheduled for {selected_doctor_name} on {requested_date.isoformat()}."
        else:
            today = date.today()
            today_appointments = self.appointments.list_doctor_appointments_for_date(selected_doctor_id, today)
            if today_appointments:
                appointments = today_appointments
                state['appointment_rows'] = self._format_appointments(appointments)
                message = f"Today's appointments for {selected_doctor_name}."
            else:
                week_start = today - timedelta(days=today.weekday())
                week_end = week_start + timedelta(days=5)
                appointments = self.appointments.list_doctor_appointments_for_range(selected_doctor_id, week_start, week_end)
                state['appointment_rows'] = self._format_appointments(appointments)
                if appointments:
                    message = f'No booked appointments for today. Here are the scheduled appointments for {selected_doctor_name} this week.'
                else:
                    message = f'There are no appointments scheduled for {selected_doctor_name} today or this week.'
        state['final_response'] = doctor_response_generator.synthesize(
            workflow_type='show_schedule',
            payload_json=json.dumps(
                {
                    'message': message,
                    'appointment_count': len(state['appointment_rows']),
                    'schedule_scope': decision.schedule_scope if decision else 'general',
                    'doctor_name': selected_doctor_name,
                    'has_appointments': bool(state['appointment_rows']),
                }
            ),
        )
        state['pending_confirmation'] = None
        return state

    def _view_patient_history(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        patient = self._resolve_selected_patient(state)
        if patient is None:
            state['final_response'] = 'Please select an appointment or provide a patient id, name, or phone number.'
            return state
        state['selected_patient'] = patient
        state['patient_records'] = self.repository.get_records_for_patient(patient.patient_id)
        state['patient_history_summary'] = self.repository.build_patient_history_summary(patient.patient_id)
        state['appointment_rows'] = self._format_appointments(self.appointments.list_patient_appointments(patient.patient_id))
        payload = dict(state['patient_history_summary'] or {})
        payload['patient_name'] = patient.full_name
        payload['patient_id'] = patient.patient_id
        payload['patient_scope'] = decision.patient_scope if (decision := state.get('intent_decision')) else 'none'
        state['final_response'] = doctor_response_generator.synthesize(
            workflow_type='view_patient_history',
            payload_json=json.dumps(payload),
        )
        state['pending_confirmation'] = None
        return state

    def _research_symptoms(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        patient = self._resolve_selected_patient(state)
        decision = state.get('intent_decision')
        query = (decision.symptoms_query if decision and decision.symptoms_query else state.get('user_query')) or ''
        if patient is not None:
            state['selected_patient'] = patient
            state['patient_history_summary'] = self.repository.build_patient_history_summary(patient.patient_id)
            try:
                state['history_rag_results'] = self.retrieval.query_patient_records(patient.patient_id, query, n_results=3)
                state['memory_rag_results'] = self.retrieval.query_patient_memory(patient.patient_id, query, n_results=2)
            except Exception:
                state['history_rag_results'] = []
                state['memory_rag_results'] = []
        try:
            medline_results = self.medline.search_topics(query)
            payload = build_medline_payload(medline_results)
            state['medline_payload'] = payload
            state['external_summary'] = external_summary_generator.summarize('doctor', 'doctor_research', query, payload)
        except Exception:
            medline_results = None
            state['external_summary'] = None
        summary = state.get('patient_history_summary') or {}
        payload = {
            'patient_name': patient.full_name if patient is not None else None,
            'patient_id': patient.patient_id if patient is not None else None,
            'latest_visit_summary': summary.get('latest_visit_summary'),
            'external_summary': state.get('external_summary'),
            'history_match': state['history_rag_results'][0].text[:220].strip() if state.get('history_rag_results') else None,
            'next_review_points': [],
        }
        if state.get('external_summary'):
            payload['next_review_points'].append('Review the external evidence alongside the current patient history.')
        if patient is not None:
            payload['next_review_points'].append('Confirm whether the current question changes immediate follow-up or treatment planning.')
        if decision is not None:
            payload['research_scope'] = decision.research_scope
            payload['patient_scope'] = decision.patient_scope
        state['final_response'] = doctor_response_generator.synthesize(
            workflow_type='research_symptoms',
            payload_json=json.dumps(payload),
        )
        state['pending_confirmation'] = None
        return state

    def _amend_appointment(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        decision = state.get('intent_decision')
        appointment = self._resolve_selected_appointment(state)
        target_date = decision.target_date if decision else None
        target_time = self._parse_requested_time(decision.target_time if decision else None)
        access_error = self._doctor_scope_error(state, appointment)
        if access_error:
            state['final_response'] = access_error
            return state
        if decision and (decision.applies_to_all or decision.source_date):
            if not target_date:
                state['final_response'] = 'Please provide the new appointment date in YYYY-MM-DD format.'
                return state
            if decision.target_end_date:
                state['final_response'] = (
                    f"I translated that request into a range from {target_date} to {decision.target_end_date}. "
                    'Please choose one exact target date for the reschedule.'
                )
                return state
            if target_time is not None:
                state['final_response'] = (
                    'Bulk reschedules can only move appointments by date. '
                    'Please choose one appointment if you want to move it to a specific time.'
                )
                return state
            source_date = date.fromisoformat(decision.source_date) if decision.source_date else None
            if source_date is not None:
                updated = self.appointments.bulk_reschedule_doctor_appointments(
                    doctor_query=state.get('selected_doctor_id') or '',
                    source_date=source_date,
                    new_date=date.fromisoformat(target_date),
                    booked_by_actor=ActorType.DOCTOR,
                )
            else:
                updated = []
                for item in self.appointments.list_doctor_appointments(state.get('selected_doctor_id') or ''):
                    moved = self.appointments.reschedule_appointment(
                        item.appointment_id,
                        patient_id=item.patient_id,
                        new_date=date.fromisoformat(target_date),
                        booked_by_actor=ActorType.DOCTOR,
                    )
                    if moved is not None:
                        updated.append(moved)
            state['appointment_rows'] = self._format_appointments(updated)
            selected_doctor_name = state.get('selected_doctor_name') or self.appointments.get_doctor_display_name(state.get('selected_doctor_id'))
            if not updated:
                if source_date is not None:
                    state['final_response'] = f'There are no booked appointments for Dr. {selected_doctor_name} on {source_date.isoformat()} to reschedule.'
                else:
                    state['final_response'] = f'There are no current booked appointments for Dr. {selected_doctor_name} to reschedule.'
                return state
            state['final_response'] = doctor_response_generator.synthesize(
                workflow_type='amend_appointment',
                payload_json=json.dumps(
                    {
                        'patient_scope': 'none',
                        'message': (
                            f"I rescheduled {len(updated)} appointment(s) for Dr. {selected_doctor_name} to {target_date}."
                        ),
                    }
                ),
            )
            state['pending_confirmation'] = None
            return state
        if appointment is None:
            state['final_response'] = 'Please select an appointment before rescheduling it.'
            return state
        if not target_date:
            state['final_response'] = 'Please provide the new appointment date in YYYY-MM-DD format.'
            return state
        if decision and (decision.target_end_date or decision.appointment_id or target_time is not None):
            _, updated = self.appointments.rebook_appointment_to_matching_available(
                appointment.appointment_id,
                patient_id=appointment.patient_id,
                booking_reason=appointment.booking_reason,
                booked_by_actor=ActorType.DOCTOR,
                target_date=date.fromisoformat(target_date),
                target_end_date=date.fromisoformat(decision.target_end_date) if decision.target_end_date else None,
                doctor_query=state.get('selected_doctor_id'),
                target_time=target_time,
            )
        else:
            updated = self.appointments.reschedule_appointment(
                appointment.appointment_id,
                patient_id=appointment.patient_id,
                new_date=date.fromisoformat(target_date),
                booked_by_actor=ActorType.DOCTOR,
            )
        if updated is None:
            state['final_response'] = 'I could not reschedule that appointment right now.'
            return state
        state['selected_appointment'] = updated
        state['appointment_rows'] = self._format_appointments(self.appointments.list_patient_appointments(updated.patient_id))
        state['final_response'] = doctor_response_generator.synthesize(
            workflow_type='amend_appointment',
            payload_json=json.dumps(
                {
                    'patient_scope': decision.patient_scope if decision else 'selected_appointment',
                    'message': (
                        f"I rescheduled the appointment to {updated.appointment_date.isoformat()} at "
                        f"{updated.appointment_time.strftime('%H:%M')} for patient {updated.patient_id}. "
                        f"Appointment reference: {updated.appointment_id}."
                    ),
                }
            ),
        )
        state['pending_confirmation'] = None
        return state

    def _cancel_appointment(self, state: DoctorWorkflowState) -> DoctorWorkflowState:
        decision = state.get('intent_decision')
        appointment = self._resolve_selected_appointment(state)
        access_error = self._doctor_scope_error(state, appointment)
        if access_error:
            state['final_response'] = access_error
            return state
        if decision and (decision.applies_to_all or decision.source_date):
            selected_doctor_id = state.get('selected_doctor_id') or ''
            selected_doctor_name = state.get('selected_doctor_name') or self.appointments.get_doctor_display_name(selected_doctor_id)
            if decision.source_date:
                appointments = self.appointments.list_doctor_appointments_for_date(selected_doctor_id, date.fromisoformat(decision.source_date))
            else:
                appointments = self.appointments.list_doctor_appointments(selected_doctor_id)
            if not appointments:
                if decision.source_date:
                    state['final_response'] = f'There are no booked appointments for Dr. {selected_doctor_name} on {decision.source_date}.'
                else:
                    state['final_response'] = f'There are no current booked appointments for Dr. {selected_doctor_name}.'
                return state
            distinct_dates = sorted({item.appointment_date.isoformat() for item in appointments})
            if decision.applies_to_all and not decision.source_date and len(distinct_dates) > 1:
                state['appointment_rows'] = self._format_appointments(appointments)
                state['pending_confirmation'] = {
                    'action_type': 'bulk_cancel_doctor_appointments',
                    'doctor_id': selected_doctor_id,
                    'available_dates': distinct_dates,
                }
                state['final_response'] = doctor_response_generator.synthesize(
                    workflow_type='cancel_appointment',
                    payload_json=json.dumps(
                        {
                            'patient_scope': 'none',
                            'message': (
                                f"I found {len(appointments)} booked appointment(s) for Dr. {selected_doctor_name} across multiple dates. "
                                f"Please confirm if you want to cancel all current appointments, or specify one date only. "
                                f"Current booked dates: {', '.join(distinct_dates)}."
                            ),
                        }
                    ),
                )
                return state
            cancelled_rows: list[dict[str, str]] = []
            for item in appointments:
                reopened = self.appointments.cancel_appointment(
                    item.appointment_id,
                    patient_id=item.patient_id,
                    booked_by_actor=ActorType.DOCTOR,
                )
                if reopened is None:
                    continue
                cancelled_rows.append(
                    {
                        'Appointment ID': item.appointment_id,
                        'Date': item.appointment_date.isoformat(),
                        'Time': item.appointment_time.strftime('%H:%M'),
                        'Doctor': self.appointments.get_doctor_display_name(item.doctor_id),
                        'Patient ID': item.patient_id or 'unassigned',
                        'Specialty': item.specialty.replace('_', ' '),
                        'Status': 'cancelled',
                        'Reason': item.booking_reason,
                    }
                )
            state['appointment_rows'] = cancelled_rows
            scope_text = f" on {decision.source_date}" if decision.source_date else ''
            state['final_response'] = doctor_response_generator.synthesize(
                workflow_type='cancel_appointment',
                payload_json=json.dumps(
                    {
                        'patient_scope': 'none',
                        'message': f"I cancelled {len(cancelled_rows)} appointment(s) for Dr. {selected_doctor_name}{scope_text}.",
                    }
                ),
            )
            state['pending_confirmation'] = None
            return state
        if appointment is None:
            state['final_response'] = 'Please select an appointment before cancelling it.'
            return state
        reopened = self.appointments.cancel_appointment(
            appointment.appointment_id,
            patient_id=appointment.patient_id,
            booked_by_actor=ActorType.DOCTOR,
        )
        if reopened is None:
            state['final_response'] = 'I could not cancel that appointment right now.'
            return state
        state['appointment_rows'] = self._format_appointments(self.appointments.list_patient_appointments(appointment.patient_id))
        state['final_response'] = doctor_response_generator.synthesize(
            workflow_type='cancel_appointment',
            payload_json=json.dumps(
                {
                    'patient_scope': decision.patient_scope if decision else 'selected_appointment',
                    'message': (
                        f"I cancelled the appointment on {appointment.appointment_date.isoformat()} at "
                        f"{appointment.appointment_time.strftime('%H:%M')} for patient {appointment.patient_id}. "
                        f"Appointment reference: {appointment.appointment_id}."
                    ),
                }
            ),
        )
        state['pending_confirmation'] = None
        return state

    def _route_after_intent(self, state: DoctorWorkflowState) -> str:
        decision = state.get('intent_decision')
        if decision is None:
            state['final_response'] = 'How may I help with schedule review, patient records, or symptom research today?'
            return 'done'
        valid = {'show_schedule', 'view_patient_history', 'research_symptoms', 'amend_appointment', 'cancel_appointment'}
        return decision.intent if decision.intent in valid else 'done'

    def _resolve_selected_patient(self, state: DoctorWorkflowState) -> Patient | None:
        appointment = state.get('selected_appointment')
        if appointment is not None and appointment.patient_id:
            return self.repository.get_patient_by_id(appointment.patient_id)
        decision = state.get('intent_decision')
        if decision and decision.patient_id:
            patient = self.repository.get_patient_by_id(decision.patient_id)
            if patient is not None:
                return patient
        if decision and decision.patient_phone:
            patient = self.repository.get_patient_by_phone(decision.patient_phone)
            if patient is not None:
                return patient
        if decision and decision.patient_name:
            matches = self.repository.search_patients_by_name(decision.patient_name)
            if matches:
                return matches[0]
        if state.get('active_patient_id'):
            patient = self.repository.get_patient_by_id(state['active_patient_id'])
            if patient is not None:
                return patient
        if state.get('patient_phone_query'):
            patient = self.repository.get_patient_by_phone(state['patient_phone_query'])
            if patient is not None:
                return patient
        if state.get('patient_name_query'):
            matches = self.repository.search_patients_by_name(state['patient_name_query'])
            if matches:
                return matches[0]
        return None

    def _resolve_selected_appointment(self, state: DoctorWorkflowState) -> Appointment | None:
        appointment = state.get('selected_appointment')
        if appointment is not None:
            return appointment
        decision = state.get('intent_decision')
        if decision and decision.appointment_id:
            appointment = self.appointments.get_appointment_by_id(decision.appointment_id)
            if appointment is not None:
                return appointment
        appointment_id = state.get('selected_appointment_id')
        if appointment_id:
            return self.appointments.get_appointment_by_id(appointment_id)
        patient = self._resolve_selected_patient(state)
        if patient is not None:
            current = self.appointments.list_current_patient_appointments(patient.patient_id)
            selected_doctor_id = state.get('selected_doctor_id')
            if selected_doctor_id:
                current = [appointment for appointment in current if appointment.doctor_id == selected_doctor_id]
            return current[0] if current else None
        return None

    def _doctor_scope_error(
        self,
        state: DoctorWorkflowState,
        appointment: Appointment | None = None,
    ) -> str | None:
        selected_doctor_id = state.get('selected_doctor_id')
        if not selected_doctor_id:
            return 'Please select a doctor profile to continue.'
        decision = state.get('intent_decision')
        requested_doctor_id = None
        if decision is not None:
            if decision.doctor_id:
                requested_doctor_id = self.appointments.resolve_doctor_id(decision.doctor_id)
            elif decision.doctor_name:
                requested_doctor_id = self.appointments.resolve_doctor_id(decision.doctor_name)
        if appointment is not None and appointment.doctor_id != selected_doctor_id:
            return 'This doctor session can only manage appointments for the selected doctor profile.'
        if requested_doctor_id and requested_doctor_id != selected_doctor_id:
            return 'This doctor session is limited to the selected doctor profile. Please use the current doctor schedule and appointments only.'
        return None

    def _format_appointments(self, appointments: list[Appointment]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for appointment in appointments:
            rows.append(
                {
                    'Appointment ID': appointment.appointment_id,
                    'Date': appointment.appointment_date.isoformat(),
                    'Time': appointment.appointment_time.strftime('%H:%M'),
                    'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                    'Patient ID': appointment.patient_id or 'unassigned',
                    'Specialty': appointment.specialty.replace('_', ' '),
                    'Status': appointment.status.value,
                    'Reason': appointment.booking_reason,
                }
            )
        return rows

    def _parse_requested_time(self, value: str | None) -> time | None:
        if not value:
            return None
        try:
            return time.fromisoformat(value)
        except ValueError:
            return None
