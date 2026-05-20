from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import date, time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.repositories.appointment_repository import get_appointment_repository
from app.repositories.sample_data_repository import get_sample_repository
from app.schemas.domain import ActorType, Patient, RetrievedEvidence
from app.services.date_resolution import resolve_date_or_range
from app.services.chroma_retrieval import ChromaRetrievalService
from app.services.external_summary_generator import build_medline_payload, external_summary_generator
from app.services.langsmith_service import langsmith_service
from app.services.medlineplus_service import MedlinePlusService
from app.services.patient_intent_service import PatientIntentDecision, patient_intent_service
from app.services.patient_response_generator import patient_response_generator


class PatientWorkflowState(TypedDict, total=False):
    session_id: str
    actor: ActorType
    user_query: str
    active_patient_id: str | None
    patient_phone: str | None
    registration_first_name: str | None
    registration_last_name: str | None
    registration_address: str | None
    conversation_context: dict[str, object] | None
    patient_profile: Patient | None
    patient_history: list[dict[str, str]]
    history_rag_results: list[RetrievedEvidence]
    memory_rag_results: list[RetrievedEvidence]
    appointment_rows: list[dict[str, str]]
    open_appointment_rows: list[dict[str, str]]
    visit_history_rows: list[dict[str, str]]
    patient_resolution_status: str | None
    workflow_type: str | None
    registration_note: str | None
    intent_decision: PatientIntentDecision | None
    requested_appointment_date: str | None
    requested_appointment_end_date: str | None
    requested_appointment_time: str | None
    requested_appointment_id: str | None
    requested_doctor_query: str | None
    medline_payload: str | None
    external_summary: str | None
    final_response: str | None
    planner_source: str | None
    booking_context: dict[str, object] | None
    langsmith_enabled: bool
    langsmith_run_id: str | None
    langsmith_run_url: str | None


class PatientAssistantWorkflow:
    def __init__(self) -> None:
        self.repository = get_sample_repository()
        self.appointments = get_appointment_repository()
        self.medline = MedlinePlusService()
        self.retrieval = ChromaRetrievalService()
        self.graph = self._build_graph()

    def run(self, state: PatientWorkflowState) -> PatientWorkflowState:
        state.setdefault('actor', ActorType.PATIENT)
        state['langsmith_enabled'] = langsmith_service.enabled
        try:
            run_id: str | None = None
            with langsmith_service.trace_context(
                'patient_workflow',
                run_type='chain',
                inputs={'query': state.get('user_query'), 'phone': state.get('patient_phone')},
                metadata={'session_id': state.get('session_id')},
                tags=['healthcare-assistant', 'patient-workflow'],
            ) as run:
                if run is not None:
                    run_id = str(run.id)
                result = self.graph.invoke(state)
            if run_id:
                result['langsmith_run_id'] = run_id
                langsmith_service.flush()
                result['langsmith_run_url'] = langsmith_service.get_verified_run_url(run_id)
                self.repository.log_patient_interaction(
                    session_id=result.get('session_id', 'sess_patient'),
                    user_message=result.get('user_query', ''),
                    assistant_message=result.get('final_response'),
                    workflow_type=result.get('intent_decision').intent if result.get('intent_decision') else None,
                    context=result.get('conversation_context'),
                )
                if result.get('langsmith_run_id') and result.get('langsmith_run_url'):
                    self.repository.log_langsmith_run(
                        session_id=result.get('session_id', 'sess_patient'),
                        actor='patient',
                        workflow_type=result.get('workflow_type'),
                        run_id=result['langsmith_run_id'],
                        trace_url=result.get('langsmith_run_url'),
                    )
            return result
        except Exception as error:
            self.repository.log_system_error(
                session_id=state.get('session_id', 'sess_patient'),
                actor='patient',
                stage='patient_workflow',
                error=error,
            )
            raise

    def stream(self, state: PatientWorkflowState) -> Iterator[PatientWorkflowState]:
        yield from self.graph.stream(state, stream_mode='values')

    def _build_graph(self):
        builder = StateGraph(PatientWorkflowState)
        builder.add_node('initialize', self._initialize)
        builder.add_node('resolve_patient', self._resolve_patient)
        builder.add_node('classify_intent', self._classify_intent)
        builder.add_node('handle_medical_history', self._handle_medical_history)
        builder.add_node('handle_appointments', self._handle_appointments)
        builder.add_node('handle_open_appointments', self._handle_open_appointments)
        builder.add_node('handle_symptoms', self._handle_symptoms)
        builder.add_node('handle_booking', self._handle_booking)
        builder.add_node('handle_reschedule', self._handle_reschedule)
        builder.add_node('handle_cancel', self._handle_cancel)

        builder.add_edge(START, 'initialize')
        builder.add_edge('initialize', 'resolve_patient')
        builder.add_conditional_edges(
            'resolve_patient',
            self._route_after_resolution,
            {
                'continue': 'classify_intent',
                'done': END,
            },
        )
        builder.add_conditional_edges(
            'classify_intent',
            self._route_after_intent,
            {
                'medical_history': 'handle_medical_history',
                'appointments': 'handle_appointments',
                'open_appointments': 'handle_open_appointments',
                'symptoms': 'handle_symptoms',
                'booking': 'handle_booking',
                'reschedule': 'handle_reschedule',
                'cancel': 'handle_cancel',
                'done': END,
            },
        )
        builder.add_edge('handle_medical_history', END)
        builder.add_edge('handle_appointments', END)
        builder.add_edge('handle_open_appointments', END)
        builder.add_edge('handle_symptoms', END)
        builder.add_edge('handle_booking', END)
        builder.add_edge('handle_reschedule', END)
        builder.add_edge('handle_cancel', END)
        return builder.compile()

    def _initialize(self, state: PatientWorkflowState) -> PatientWorkflowState:
        state.setdefault('conversation_context', {})
        state.setdefault('active_patient_id', None)
        state.setdefault('patient_history', [])
        state.setdefault('appointment_rows', [])
        state.setdefault('open_appointment_rows', [])
        state.setdefault('visit_history_rows', [])
        state.setdefault('history_rag_results', [])
        state.setdefault('memory_rag_results', [])
        state.setdefault('patient_resolution_status', None)
        state.setdefault('workflow_type', None)
        state.setdefault('registration_note', None)
        state.setdefault('intent_decision', None)
        state.setdefault('requested_appointment_date', None)
        state.setdefault('requested_appointment_end_date', None)
        state.setdefault('requested_appointment_time', None)
        state.setdefault('requested_appointment_id', None)
        state.setdefault('requested_doctor_query', None)
        state.setdefault('planner_source', None)
        state.setdefault('booking_context', None)
        return state

    def _resolve_patient(self, state: PatientWorkflowState) -> PatientWorkflowState:
        query = (state.get('user_query') or '').strip()
        phone = state.get('patient_phone') or self._extract_phone(query)
        if phone:
            state['patient_phone'] = phone

        if not phone:
            state['patient_resolution_status'] = 'awaiting_phone'
            state['final_response'] = "Please share your phone number so I can look up your record."
            return state

        patient = self.repository.get_patient_by_phone(phone)
        first_name = (state.get('registration_first_name') or '').strip()
        last_name = (state.get('registration_last_name') or '').strip()
        address = (state.get('registration_address') or '').strip()
        if patient is None and not (first_name and last_name):
            state['patient_resolution_status'] = 'awaiting_registration'
            state['final_response'] = (
                f"I could not find a record for phone number {phone}. "
                "Please complete the registration form with first name, last name, and optional address."
            )
            return state

        if patient is None:
            patient = self.repository.register_patient(phone, first_name, last_name, address or None)
            state['registration_note'] = f"I created your patient profile successfully, {first_name}."
            state['patient_resolution_status'] = 'resolved'
            state['patient_profile'] = patient
            state['active_patient_id'] = patient.patient_id
            return state

        state['patient_profile'] = patient
        state['active_patient_id'] = patient.patient_id
        state['patient_resolution_status'] = 'resolved'
        return state

    def _classify_intent(self, state: PatientWorkflowState) -> PatientWorkflowState:
        if state.get('patient_profile') is None:
            return state
        query = state.get('user_query', '')
        if not query:
            state['intent_decision'] = PatientIntentDecision(intent='general_help', rationale='Empty query.')
            return state
        context = state.get('conversation_context') or {}
        decision = patient_intent_service.classify(query, json.dumps(context, sort_keys=True))
        fallback_target_date, fallback_target_end_date = resolve_date_or_range(query)
        fallback_appointment_id = self._extract_appointment_id(query)
        decision.target_date = decision.target_date or fallback_target_date
        decision.target_end_date = decision.target_end_date or fallback_target_end_date
        decision.appointment_id = decision.appointment_id or fallback_appointment_id
        if decision.target_date:
            state['requested_appointment_date'] = decision.target_date
        if decision.target_end_date:
            state['requested_appointment_end_date'] = decision.target_end_date
        if decision.target_time:
            state['requested_appointment_time'] = decision.target_time
        if decision.appointment_id:
            state['requested_appointment_id'] = decision.appointment_id
        if decision.doctor_query:
            state['requested_doctor_query'] = decision.doctor_query
        state['intent_decision'] = decision
        state['workflow_type'] = decision.intent
        state['planner_source'] = 'llm_or_fallback'
        if decision.intent in {'identify_patient', 'general_help'}:
            patient = state['patient_profile']
            first_name = patient.full_name.strip().split()[0] if patient.full_name.strip() else 'there'
            if state.get('registration_note'):
                state['final_response'] = (
                    f"{state['registration_note']} How may I assist you today, {first_name}?"
                )
            else:
                state['final_response'] = (
                    f"Hello {first_name}, I found your record using phone number {patient.phone}. "
                    f"How may I assist you today, {first_name}?"
                )
        self.repository.log_patient_planner_trace(
            session_id=state.get('session_id', 'sess_patient'),
            user_message=query,
            context=context,
            planner_output=decision.model_dump(mode='json'),
            final_workflow_type=decision.intent,
        )
        return state

    def _handle_appointments(self, state: PatientWorkflowState) -> PatientWorkflowState:
        patient = state['patient_profile']
        decision = state.get('intent_decision')
        wants_past = bool(decision and decision.appointment_scope == 'past')
        if wants_past:
            appointments = self.appointments.list_patient_appointments(patient.patient_id)
        else:
            appointments = self.appointments.list_current_patient_appointments(patient.patient_id)
        cancelled_appointments = self.appointments.list_cancelled_patient_appointments(patient.patient_id)
        state['appointment_rows'] = [
            {
                'Appointment ID': appointment.appointment_id,
                'Date': appointment.appointment_date.isoformat(),
                'Time': appointment.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                'Specialty': appointment.specialty.replace('_', ' '),
                'Status': appointment.status.value,
                'Reason': appointment.booking_reason,
            }
            for appointment in appointments
        ]
        if state['appointment_rows']:
            label = 'appointment record(s)' if wants_past else 'scheduled appointment(s)'
            state['final_response'] = f"I found {len(state['appointment_rows'])} {label} for {patient.full_name}."
        elif cancelled_appointments and not wants_past:
            latest_cancelled = cancelled_appointments[0]
            state['final_response'] = (
                "You do not have any open or scheduled appointments right now. "
                f"Your most recent appointment on {latest_cancelled.appointment_date.isoformat()} was cancelled."
            )
        else:
            state['final_response'] = (
                f"I could not find any appointment history for {patient.full_name}."
                if wants_past
                else "You do not have any open or scheduled appointments right now."
            )
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _handle_medical_history(self, state: PatientWorkflowState) -> PatientWorkflowState:
        patient = state['patient_profile']
        history_summary = self.repository.build_patient_history_summary(patient.patient_id)
        visit_rows = self.repository.build_patient_visit_history(patient.patient_id)
        state['visit_history_rows'] = visit_rows
        if visit_rows:
            record_count = history_summary.get('record_count') or len(visit_rows)
            latest_visit = history_summary.get('latest_visit_date')
            latest_summary = history_summary.get('latest_visit_summary')
            response_parts = [
                f"I found {record_count} medical record entr{'y' if record_count == 1 else 'ies'} for {patient.full_name}.",
            ]
            if latest_visit:
                response_parts.append(f"Your most recent documented visit was on {latest_visit}.")
            if latest_summary:
                response_parts.append(f"The latest visit focused on {latest_summary}.")
            state['final_response'] = ' '.join(response_parts)
        else:
            state['final_response'] = f"I could not find any medical history records for {patient.full_name}."
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _handle_open_appointments(self, state: PatientWorkflowState) -> PatientWorkflowState:
        start_date = date.fromisoformat(state['requested_appointment_date']) if state.get('requested_appointment_date') else None
        end_date = date.fromisoformat(state['requested_appointment_end_date']) if state.get('requested_appointment_end_date') else None
        requested_time = self._parse_requested_time(state.get('requested_appointment_time'))
        doctor_query = state.get('requested_doctor_query')
        if start_date and end_date:
            self.appointments.ensure_open_slots_for_range(start_date, end_date)
            open_slots = self.appointments.find_matching_open_slots(
                target_date=start_date,
                end_date=end_date,
                doctor_query=doctor_query,
                target_time=requested_time,
            )
        elif start_date:
            self.appointments.ensure_open_slots_for_range(start_date, start_date)
            open_slots = self.appointments.find_matching_open_slots(
                target_date=start_date,
                doctor_query=doctor_query,
                target_time=requested_time,
            )
        elif doctor_query:
            open_slots = self.appointments.find_matching_open_slots(doctor_query=doctor_query, target_time=requested_time)
        else:
            open_slots = self.appointments.find_matching_open_slots(target_time=requested_time)
        state['open_appointment_rows'] = [
            {
                'Appointment ID': appointment.appointment_id,
                'Date': appointment.appointment_date.isoformat(),
                'Time': appointment.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                'Specialty': appointment.specialty.replace('_', ' '),
            }
            for appointment in open_slots
        ]
        if state['open_appointment_rows']:
            if start_date and end_date:
                state['final_response'] = (
                    f"I found {len(state['open_appointment_rows'])} available appointment slot(s) "
                    f"between {start_date.isoformat()} and {end_date.isoformat()}."
                )
            elif start_date:
                state['final_response'] = f"I found {len(state['open_appointment_rows'])} available appointment slot(s) for {start_date.isoformat()}."
            elif doctor_query:
                state['final_response'] = f"I found {len(state['open_appointment_rows'])} available appointment slot(s) for {doctor_query}."
            else:
                state['final_response'] = f"I found {len(state['open_appointment_rows'])} open appointment slot(s)."
        else:
            if start_date and end_date:
                state['final_response'] = f"I could not find any available appointment slots between {start_date.isoformat()} and {end_date.isoformat()}."
            elif start_date:
                state['final_response'] = f"I could not find any available appointment slots for {start_date.isoformat()}."
            elif doctor_query:
                state['final_response'] = f"I could not find any available appointment slots for {doctor_query} right now."
            else:
                state['final_response'] = "There are no open appointment slots right now."
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _handle_symptoms(self, state: PatientWorkflowState) -> PatientWorkflowState:
        patient = state['patient_profile']
        query = state.get('user_query', '')
        state['patient_history'] = self.repository.build_patient_visit_history(patient.patient_id)
        try:
            state['history_rag_results'] = self.retrieval.query_patient_records(patient.patient_id, query, n_results=3)
            state['memory_rag_results'] = self.retrieval.query_patient_memory(patient.patient_id, query, n_results=2)
        except Exception:
            state['history_rag_results'] = []
            state['memory_rag_results'] = []
        diagnosis_hint = state['patient_history'][0]['Diagnosis'] if state['patient_history'] else None
        prior_plan = state['patient_history'][0]['Treatment Summary'] if state['patient_history'] else None
        route_type = 'historical_match' if diagnosis_hint else 'mixed_or_uncertain'
        external_summary = None
        if state.get('intent_decision') and state['intent_decision'].needs_medline:
            try:
                results = self.medline.search_topics(query)
                payload = build_medline_payload(results)
                external_summary = external_summary_generator.summarize('patient', 'patient_research', query, payload)
                state['medline_payload'] = payload
            except Exception:
                external_summary = None
        state['external_summary'] = external_summary
        state['final_response'] = patient_response_generator.synthesize(
            user_query=query,
            route_type=route_type,
            specialty='general_medicine',
            diagnosis_hint=diagnosis_hint,
            prior_plan=prior_plan,
            recent_visit_summary=prior_plan,
            external_summary=external_summary,
            booking_requested=False,
        )
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _handle_booking(self, state: PatientWorkflowState) -> PatientWorkflowState:
        patient = state['patient_profile']
        query = state.get('user_query', '')
        requested_date = state.get('requested_appointment_date')
        requested_end_date = state.get('requested_appointment_end_date')
        requested_time = state.get('requested_appointment_time')
        doctor_query = state.get('requested_doctor_query')
        state['booking_context'] = None
        decision = state.get('intent_decision')
        booking_followup_action = decision.booking_followup_action if decision else 'none'
        has_symptoms = bool(decision and decision.symptoms_present)
        diagnosis_hint = None
        prior_plan = None
        external_summary = None
        route_type = 'mixed_or_uncertain'
        if has_symptoms:
            state['patient_history'] = self.repository.build_patient_visit_history(patient.patient_id)
            diagnosis_hint = state['patient_history'][0]['Diagnosis'] if state['patient_history'] else None
            prior_plan = state['patient_history'][0]['Treatment Summary'] if state['patient_history'] else None
            route_type = 'historical_match' if diagnosis_hint else 'mixed_or_uncertain'
            if decision and decision.needs_medline:
                try:
                    results = self.medline.search_topics(query)
                    payload = build_medline_payload(results)
                    external_summary = external_summary_generator.summarize('patient', 'patient_research', query, payload)
                    state['medline_payload'] = payload
                except Exception:
                    external_summary = None
            state['external_summary'] = external_summary
        else:
            state['external_summary'] = None
        preferred_doctor = self._preferred_doctor_for_patient(patient.patient_id)
        if booking_followup_action == 'change_doctor_preference':
            if doctor_query:
                state['requested_doctor_query'] = doctor_query
            else:
                state['requested_doctor_query'] = None
                state['open_appointment_rows'] = self._build_general_open_slot_rows(limit=8)
                state['final_response'] = (
                    f"Certainly, {patient.full_name.split()[0]}. "
                    "Please tell me which doctor or department you would prefer instead. "
                    f"Current options include: {self._doctor_option_summary()}."
                )
                return state
        if requested_date and not doctor_query:
            if preferred_doctor is not None:
                state['open_appointment_rows'] = self._build_open_slot_rows_for_doctor(preferred_doctor.full_name, limit=5)
                state['booking_context'] = {
                    'clarification_type': 'confirm_preferred_doctor_for_requested_date',
                    'suggested_doctor_query': preferred_doctor.full_name,
                    'suggested_specialty': preferred_doctor.specialty,
                    'requested_date': requested_date,
                    'requested_end_date': requested_end_date,
                    'requested_time': requested_time,
                }
                state['final_response'] = (
                    f"I can help with that {'time on ' if requested_time else 'date around '}{requested_date}."
                    f" Would you like to book with Dr. {preferred_doctor.full_name}, "
                    f"who handled your last recorded visit, for the next available slot"
                    f"{' at ' + requested_time if requested_time else ''} around {requested_date}? "
                    "If not, please share another doctor or department."
                )
                return state
            state['open_appointment_rows'] = self._build_general_open_slot_rows(limit=8)
            state['final_response'] = (
                f"I can help find an appointment around {requested_date}. "
                f"{'I will look for ' + requested_time + ' if possible. ' if requested_time else ''}"
                "Please tell me which doctor or department you would like to meet. "
                f"Current options include: {self._doctor_option_summary()}."
            )
            return state
        if not requested_date and not doctor_query:
            if preferred_doctor is not None:
                state['open_appointment_rows'] = self._build_open_slot_rows_for_doctor(preferred_doctor.full_name, limit=5)
                state['booking_context'] = {
                    'clarification_type': 'confirm_preferred_doctor_next_available',
                    'suggested_doctor_query': preferred_doctor.full_name,
                    'suggested_specialty': preferred_doctor.specialty,
                }
                state['final_response'] = (
                    f"Your last recorded visit was with Dr. {preferred_doctor.full_name}. "
                    f"Would you like me to book the next available slot with Dr. {preferred_doctor.full_name}? "
                    "If yes, reply yes. You can also share a preferred date or another doctor or department."
                )
                return state
            state['open_appointment_rows'] = self._build_general_open_slot_rows(limit=8)
            state['final_response'] = (
                f"I can help book that for you, {patient.full_name.split()[0]}. "
                "Please tell me which doctor or department you would like to meet. "
                f"Current options include: {self._doctor_option_summary()}."
            )
            return state
        if requested_date and requested_end_date:
            return self._handle_open_appointments(state)
        if requested_date:
            booked = self.appointments.schedule_matching_available_for_patient(
                patient_id=patient.patient_id,
                booking_reason=state.get('user_query', 'Patient requested appointment'),
                booked_by_actor=ActorType.PATIENT,
                target_date=date.fromisoformat(requested_date),
                target_end_date=date.fromisoformat(requested_end_date) if requested_end_date else None,
                doctor_query=doctor_query,
                target_time=self._parse_requested_time(requested_time),
            )
        else:
            if doctor_query:
                booked = self.appointments.schedule_matching_available_for_patient(
                    patient_id=patient.patient_id,
                    booking_reason=state.get('user_query', 'Patient requested appointment'),
                    booked_by_actor=ActorType.PATIENT,
                    doctor_query=doctor_query,
                    target_time=self._parse_requested_time(requested_time),
                )
            else:
                booked = self.appointments.schedule_next_available_for_patient(
                    patient_id=patient.patient_id,
                    booking_reason=state.get('user_query', 'Patient requested appointment'),
                    booked_by_actor=ActorType.PATIENT,
                )
        if booked is None:
            if doctor_query and self.appointments.resolve_doctor_id(doctor_query) is None:
                state['open_appointment_rows'] = self._build_general_open_slot_rows(limit=8)
                state['final_response'] = (
                    f"I could not match the doctor name '{doctor_query}' in the current registry. "
                    f"Please choose one of the available doctors or departments instead: {self._doctor_option_summary()}."
                )
                return state
            if requested_date:
                qualifier = f" with {doctor_query}" if doctor_query else ''
                time_qualifier = f" at {requested_time}" if requested_time else ''
                state['open_appointment_rows'] = self._build_open_slot_rows_for_doctor(doctor_query, limit=5) if doctor_query else self._build_general_open_slot_rows(limit=8)
                state['final_response'] = (
                    f"I could not find any open appointment slots for {requested_date}{time_qualifier}{qualifier}. "
                    "Please choose another available time or date from the options below."
                )
            else:
                state['final_response'] = (
                    "I could not find any open appointment slots right now. "
                    f"Available doctors and departments include: {self._doctor_option_summary()}."
                )
            return state
        state['appointment_rows'] = [
            {
                'Appointment ID': item.appointment_id,
                'Date': item.appointment_date.isoformat(),
                'Time': item.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(item.doctor_id),
                'Specialty': item.specialty.replace('_', ' '),
                'Status': item.status.value,
                'Reason': item.booking_reason,
            }
            for item in self.appointments.list_current_patient_appointments(patient.patient_id)
        ]
        state['final_response'] = patient_response_generator.synthesize(
            user_query=query,
            route_type=route_type,
            specialty='general_medicine',
            diagnosis_hint=diagnosis_hint,
            prior_plan=prior_plan,
            recent_visit_summary=prior_plan,
            external_summary=external_summary,
            booking_requested=True,
            booking_confirmed=True,
            confirmed_appointment={
                'date': booked.appointment_date.isoformat(),
                'time': booked.appointment_time.strftime('%H:%M'),
                'doctor_name': self.appointments.get_doctor_display_name(booked.doctor_id),
                'appointment_id': booked.appointment_id,
            },
        )
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _handle_reschedule(self, state: PatientWorkflowState) -> PatientWorkflowState:
        patient = state['patient_profile']
        requested_date = state.get('requested_appointment_date')
        requested_end_date = state.get('requested_appointment_end_date')
        requested_time = state.get('requested_appointment_time')
        requested_appointment_id = state.get('requested_appointment_id')
        doctor_query = state.get('requested_doctor_query')
        parsed_requested_time = self._parse_requested_time(requested_time)
        current_appointments = self.appointments.list_current_patient_appointments(patient.patient_id)
        if not current_appointments:
            if requested_appointment_id:
                state['final_response'] = (
                    "I could not find that appointment among your current scheduled appointments. "
                    "Please check the appointment reference or ask me to show your current appointments."
                )
            else:
                state['final_response'] = f"I could not find a current appointment to reschedule for {patient.full_name}."
            return state
        if not requested_date:
            state['final_response'] = "I can help reschedule your appointment. Please share the new date in YYYY-MM-DD format."
            return state
        current = self.appointments.get_appointment_by_id(requested_appointment_id) if requested_appointment_id else current_appointments[0]
        if current is None or current.patient_id != patient.patient_id:
            state['final_response'] = (
                "I could not find that appointment among your current scheduled appointments. "
                "Please check the appointment reference or ask me to show your current appointments."
            )
            return state
        effective_doctor_query = doctor_query or self.appointments.get_doctor_display_name(current.doctor_id)
        if requested_end_date or requested_appointment_id or parsed_requested_time is not None:
            _, updated = self.appointments.rebook_appointment_to_matching_available(
                current.appointment_id,
                patient_id=patient.patient_id,
                booking_reason=current.booking_reason,
                booked_by_actor=ActorType.PATIENT,
                target_date=date.fromisoformat(requested_date),
                target_end_date=date.fromisoformat(requested_end_date) if requested_end_date else None,
                doctor_query=effective_doctor_query,
                target_time=parsed_requested_time,
            )
        else:
            updated = self.appointments.reschedule_appointment(
                current.appointment_id,
                patient_id=patient.patient_id,
                new_date=date.fromisoformat(requested_date),
                booked_by_actor=ActorType.PATIENT,
            )
        if updated is None:
            doctor_display_name = self.appointments.get_doctor_display_name(current.doctor_id)
            matching_open_slots = self.appointments.find_matching_open_slots(
                target_date=date.fromisoformat(requested_date),
                end_date=date.fromisoformat(requested_end_date) if requested_end_date else None,
                doctor_query=effective_doctor_query,
                target_time=parsed_requested_time,
            )
            if not matching_open_slots:
                state['open_appointment_rows'] = self._build_open_slot_rows_for_doctor(effective_doctor_query, limit=5)
                time_qualifier = f" at {requested_time}" if requested_time else ''
                state['final_response'] = (
                    f"I could not find an available slot on {requested_date}{time_qualifier} with {doctor_display_name} to reschedule this appointment. "
                    "Please try another date, or choose from the next available slots shown below."
                )
            else:
                state['final_response'] = "I could not reschedule the appointment right now."
            return state
        state['appointment_rows'] = [
            {
                'Appointment ID': item.appointment_id,
                'Date': item.appointment_date.isoformat(),
                'Time': item.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(item.doctor_id),
                'Specialty': item.specialty.replace('_', ' '),
                'Status': item.status.value,
                'Reason': item.booking_reason,
            }
            for item in self.appointments.list_current_patient_appointments(patient.patient_id)
        ]
        state['final_response'] = (
            f"I rescheduled your appointment to {updated.appointment_date.isoformat()} at "
            f"{updated.appointment_time.strftime('%H:%M')} with {self.appointments.get_doctor_display_name(updated.doctor_id)}. "
            f"Appointment reference: {updated.appointment_id}."
        )
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _handle_cancel(self, state: PatientWorkflowState) -> PatientWorkflowState:
        patient = state['patient_profile']
        current_appointments = self.appointments.list_current_patient_appointments(patient.patient_id)
        if not current_appointments:
            state['final_response'] = f"I could not find a current appointment to cancel for {patient.full_name}."
            return state
        current = current_appointments[0]
        reopened = self.appointments.cancel_appointment(
            current.appointment_id,
            patient_id=patient.patient_id,
            booked_by_actor=ActorType.PATIENT,
        )
        if reopened is None:
            state['final_response'] = "I could not cancel the appointment right now."
            return state
        state['final_response'] = (
            f"I cancelled your appointment on {current.appointment_date.isoformat()} at "
            f"{current.appointment_time.strftime('%H:%M')}. Appointment reference: {current.appointment_id}."
        )
        if state.get('registration_note'):
            state['final_response'] = f"{state['registration_note']} {state['final_response']}"
        return state

    def _route_after_resolution(self, state: PatientWorkflowState) -> str:
        if state.get('patient_resolution_status') in {'awaiting_phone', 'awaiting_registration'}:
            return 'done'
        return 'continue'

    def _route_after_intent(self, state: PatientWorkflowState) -> str:
        decision = state.get('intent_decision')
        if decision is None:
            state['final_response'] = "How may I assist you today?"
            return 'done'
        return {
            'show_medical_history': 'medical_history',
            'show_past_appointments': 'appointments',
            'show_open_appointments': 'open_appointments',
            'symptom_research': 'symptoms',
            'book_appointment': 'booking',
            'amend_appointment': 'reschedule',
            'cancel_appointment': 'cancel',
        }.get(decision.intent, 'done')

    def _extract_phone(self, text: str) -> str | None:
        digits = ''.join(ch for ch in text if ch.isdigit())
        return digits if len(digits) >= 10 else None

    def _parse_requested_time(self, value: str | None) -> time | None:
        if not value:
            return None
        try:
            return time.fromisoformat(value)
        except ValueError:
            return None

    def _extract_appointment_id(self, text: str) -> str | None:
        match = re.search(r'\b((?:appt(?:_open(?:_auto)?)?_[a-z0-9_]+)|(?:slot_doc_[a-z0-9_]+))\b', text.lower())
        return match.group(1) if match else None

    def _preferred_doctor_for_patient(self, patient_id: str):
        recent_record = self.repository.get_recent_record_for_patient(patient_id)
        if recent_record and recent_record.doctor_id:
            doctor = self.repository.get_doctor_by_id(recent_record.doctor_id)
            if doctor is not None:
                return doctor
        current_appointments = self.appointments.list_patient_appointments(patient_id)
        for appointment in current_appointments:
            doctor = self.appointments.get_doctor_by_id(appointment.doctor_id)
            if doctor is not None:
                return doctor
        return None

    def _build_open_slot_rows_for_doctor(self, doctor_query: str, *, limit: int = 5) -> list[dict[str, str]]:
        slots = self.appointments.find_matching_open_slots(doctor_query=doctor_query)[:limit]
        return [
            {
                'Appointment ID': appointment.appointment_id,
                'Date': appointment.appointment_date.isoformat(),
                'Time': appointment.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                'Specialty': appointment.specialty.replace('_', ' '),
            }
            for appointment in slots
        ]

    def _build_general_open_slot_rows(self, *, limit: int = 8) -> list[dict[str, str]]:
        slots = self.appointments.list_open_appointments()[:limit]
        return [
            {
                'Appointment ID': appointment.appointment_id,
                'Date': appointment.appointment_date.isoformat(),
                'Time': appointment.appointment_time.strftime('%H:%M'),
                'Doctor': self.appointments.get_doctor_display_name(appointment.doctor_id),
                'Specialty': appointment.specialty.replace('_', ' '),
            }
            for appointment in slots
        ]

    def _doctor_option_summary(self) -> str:
        doctors = self.repository.list_doctors()
        if not doctors:
            return 'no doctors are currently listed'
        unique_options: list[str] = []
        for doctor in doctors[:4]:
            unique_options.append(f"Dr. {doctor.full_name} ({doctor.specialty})")
        return '; '.join(unique_options)
