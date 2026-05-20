from __future__ import annotations

import json
from functools import lru_cache

from openai import OpenAI

from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class AttendantResponseGenerator:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('ATTENDANT_RESPONSE_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def synthesize(self, *, workflow_type: str, payload_json: str) -> str:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': load_prompt('attendant/response_synthesizer.txt')},
                        {'role': 'user', 'content': json.dumps({'workflow_type': workflow_type, 'payload': json.loads(payload_json)})},
                    ],
                )
                content = response.choices[0].message.content or '{}'
                data = json.loads(content)
                text = str(data.get('response_text', '')).strip()
                if text:
                    return text
            except Exception:
                pass
        payload = json.loads(payload_json)
        if workflow_type == 'execute_attendant_tasks':
            errors = payload.get('validation_errors') or payload.get('errors') or []
            outcomes = payload.get('task_outcomes') or []
            if errors:
                return ' '.join(str(item) for item in errors)
            if outcomes:
                parts: list[str] = []
                for outcome in outcomes:
                    action = outcome.get('action')
                    status = outcome.get('status')
                    if status == 'failed':
                        if outcome.get('error'):
                            parts.append(str(outcome['error']))
                            continue
                        if action == 'book_appointments':
                            detail = 'I could not complete the requested booking.'
                            if outcome.get('suggestion'):
                                detail = f"{detail} {outcome['suggestion']}"
                            parts.append(detail)
                            continue
                    if status == 'options':
                        count = outcome.get('count', 0)
                        target_date = outcome.get('target_date') or 'the requested date'
                        target_end_date = outcome.get('target_end_date')
                        target_time = outcome.get('target_time')
                        doctor_query = outcome.get('doctor_query')
                        doctor_text = f" for {doctor_query}" if doctor_query else ''
                        time_text = f" at {target_time}" if target_time else ''
                        if action == 'book_appointments':
                            if target_end_date:
                                parts.append(
                                    f"I found {count} available booking option(s){doctor_text}{time_text} between {target_date} and {target_end_date}."
                                )
                            else:
                                parts.append(f"I found {count} available booking option(s){doctor_text}{time_text} for {target_date}.")
                        elif action == 'reschedule_appointments':
                            if target_end_date:
                                parts.append(
                                    f"I found {count} available rebooking option(s){doctor_text}{time_text} between {target_date} and {target_end_date}."
                                )
                            else:
                                parts.append(f"I found {count} available rebooking option(s){doctor_text}{time_text} for {target_date}.")
                        continue
                    if action == 'show_open_appointments':
                        count = outcome.get('count', 0)
                        parts.append(f"I found {count} open appointment slot(s).")
                    elif action == 'show_booked_appointments':
                        count = outcome.get('count', 0)
                        parts.append(f"I found {count} booked or scheduled appointment(s).")
                    elif action == 'show_active_patients':
                        count = outcome.get('count', 0)
                        parts.append(f"I found {count} active patient record(s).")
                    elif action == 'show_doctors':
                        count = outcome.get('count', 0)
                        parts.append(f"I found {count} active doctor record(s).")
                    elif action == 'view_patient_history':
                        patient_name = outcome.get('patient_name') or 'the selected patient'
                        parts.append(f"Here is the history summary for {patient_name}.")
                    elif action == 'update_patient_history':
                        patient_name = outcome.get('patient_name') or 'the selected patient'
                        summary_text = outcome.get('summary_text')
                        latest_visit_summary = outcome.get('latest_visit_summary')
                        if summary_text:
                            parts.append(f"I updated the medical history for {patient_name}. {summary_text}")
                        elif latest_visit_summary:
                            parts.append(f"I updated the medical history for {patient_name}. Latest update: {latest_visit_summary}")
                        else:
                            parts.append(f"I updated the medical history for {patient_name}.")
                    elif action == 'edit_patient_details':
                        patient_name = outcome.get('patient_name') or 'the selected patient'
                        parts.append(f"I updated the patient profile for {patient_name}.")
                    elif action == 'delete_patient':
                        patient_name = outcome.get('patient_name') or 'the selected patient'
                        parts.append(f"I deleted the patient profile for {patient_name}.")
                    elif action == 'register_doctor':
                        doctor_name = outcome.get('doctor_name') or 'the doctor'
                        specialty = outcome.get('doctor_specialty') or 'the selected specialty'
                        parts.append(f"I registered Dr. {doctor_name} under {specialty}.")
                    elif action == 'edit_doctor_details':
                        doctor_name = outcome.get('doctor_name') or 'the doctor'
                        parts.append(f"I updated Dr. {doctor_name}'s profile.")
                    elif action == 'cancel_appointments':
                        count = outcome.get('count', 0)
                        if count > 1:
                            parts.append(f"I cancelled {count} appointment(s).")
                        else:
                            appointment_date = outcome.get('date') or 'the selected date'
                            appointment_time = outcome.get('time') or 'the selected time'
                            parts.append(f"I cancelled the appointment on {appointment_date} at {appointment_time}.")
                    elif action == 'reschedule_appointments':
                        count = outcome.get('count', 0)
                        target_date = outcome.get('target_date') or 'the requested date'
                        target_time = outcome.get('target_time')
                        time_text = f" at {target_time}" if target_time else ''
                        if count > 1:
                            parts.append(f"I rescheduled {count} appointment(s) to {target_date}{time_text}.")
                        else:
                            parts.append(f"I rescheduled the appointment to {target_date}{time_text}.")
                    elif action == 'book_appointments':
                        booked_count = outcome.get('booked_count', 0)
                        unbooked_count = outcome.get('unbooked_count', 0)
                        target_date = outcome.get('target_date') or 'the requested date'
                        target_time = outcome.get('target_time')
                        doctor_query = outcome.get('doctor_query')
                        doctor_text = f" with {doctor_query}" if doctor_query else ''
                        time_text = f" at {target_time}" if target_time else ''
                        if booked_count and not unbooked_count:
                            if booked_count > 1:
                                parts.append(f"I booked {booked_count} appointment(s) for {target_date}{time_text}{doctor_text}.")
                            else:
                                appointment_time = outcome.get('time') or 'the scheduled time'
                                parts.append(f"I booked the appointment for {target_date} at {appointment_time}{doctor_text}.")
                        elif booked_count and unbooked_count:
                            detail = (
                                f"I booked {booked_count} appointment(s) for {target_date}{time_text}{doctor_text}, "
                                f"but {unbooked_count} appointment(s) could not be rebooked because matching open slots were not available."
                            )
                            if outcome.get('suggestion'):
                                detail = f"{detail} {outcome['suggestion']}"
                            parts.append(detail)
                if parts:
                    return ' '.join(str(item) for item in parts if item)
            return 'The requested attendant operations have been processed.'
        patient_name = payload.get('patient_name') or 'the selected patient'
        appointment_scope = payload.get('appointment_scope') or 'none'
        patient_scope = payload.get('patient_scope') or 'none'
        if workflow_type == 'view_patient_history':
            latest = payload.get('latest_visit_summary')
            conditions = ', '.join(payload.get('primary_conditions') or [])
            parts = [f"Here is the current history summary for {patient_name}."]
            if conditions:
                parts.append(f"Primary conditions on record: {conditions}.")
            if latest:
                parts.append(f"Latest visit summary: {latest}.")
            return ' '.join(parts)
        if workflow_type == 'show_open_appointments':
            open_count = payload.get('open_count', 0)
            return (
                f"I found {open_count} open appointment slot(s)."
                if open_count
                else 'There are no open appointment slots right now.'
            )
        if workflow_type == 'show_booked_appointments':
            appointment_count = payload.get('appointment_count', 0)
            return (
                f"I found {appointment_count} booked or scheduled appointment(s)."
                if appointment_count
                else 'There are no booked or scheduled appointments right now.'
            )
        if workflow_type == 'show_active_patients':
            patient_count = payload.get('patient_count', 0)
            return f"I found {patient_count} active patient record(s)."
        if workflow_type == 'show_doctors':
            doctor_count = payload.get('doctor_count', 0)
            return f"I found {doctor_count} active doctor record(s)."
        if workflow_type in {'register_doctor', 'edit_doctor_details'}:
            return str(payload.get('message') or 'The doctor registry has been updated.')
        if workflow_type in {'schedule_patient_appointment', 'reschedule_patient_appointment', 'cancel_patient_appointment'}:
            action = {
                'schedule_patient_appointment': 'scheduled',
                'reschedule_patient_appointment': 'rescheduled',
                'cancel_patient_appointment': 'cancelled',
            }.get(workflow_type, 'updated')
            if patient_scope == 'selected_patient':
                return str(payload.get('message') or f"The selected patient's appointment was {action}.")
            return str(payload.get('message') or f'The patient appointment was {action}.')
        if workflow_type == 'bulk_reschedule_booked_appointments':
            updated_count = payload.get('updated_count', 0)
            doctor_query = payload.get('doctor_query') or 'the selected doctor'
            source_date = payload.get('source_date') or 'the original date'
            target_date = payload.get('target_date') or 'the new date'
            if updated_count:
                return f"I rescheduled {updated_count} appointment(s) for {doctor_query} from {source_date} to {target_date}."
            return f"I could not find any matching appointments for {doctor_query} on {source_date}."
        return str(payload.get('message') or 'The attendant request has been processed.')


attendant_response_generator = AttendantResponseGenerator()
