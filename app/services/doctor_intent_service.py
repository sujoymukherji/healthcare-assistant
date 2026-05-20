from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from app.services.date_resolution import resolve_date_or_range, resolve_single_date
from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class DoctorIntentDecision(BaseModel):
    intent: str
    rationale: str = ''
    lookup_date: str | None = None
    source_date: str | None = None
    target_date: str | None = None
    target_end_date: str | None = None
    target_time: str | None = None
    appointment_id: str | None = None
    doctor_name: str | None = None
    doctor_id: str | None = None
    patient_name: str | None = None
    patient_phone: str | None = None
    patient_id: str | None = None
    symptoms_query: str | None = None
    use_selected_appointment: bool = False
    applies_to_all: bool = False
    schedule_scope: str = 'none'
    patient_scope: str = 'none'
    research_scope: str = 'none'


class DoctorIntentService:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('DOCTOR_INTENT_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def classify(self, user_query: str, conversation_context_json: str = '') -> DoctorIntentDecision:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': load_prompt('doctor/intent_classifier.txt')},
                        {
                            'role': 'user',
                            'content': json.dumps(
                                {
                                    'user_query': user_query,
                                    'conversation_context': json.loads(conversation_context_json) if conversation_context_json else {},
                                }
                            ),
                        },
                    ],
                )
                content = response.choices[0].message.content or '{}'
                return DoctorIntentDecision.model_validate(json.loads(content))
            except Exception:
                pass
        return self._fallback(user_query, conversation_context_json)

    def _fallback(self, user_query: str, conversation_context_json: str = '') -> DoctorIntentDecision:
        lowered = user_query.strip().lower()
        context = json.loads(conversation_context_json) if conversation_context_json else {}
        lookup_date = _extract_date(user_query)
        appointment_id = _extract_appointment_id(user_query)
        source_date, target_date, target_end_date = _extract_reschedule_dates(user_query)
        target_time = _extract_time(user_query)
        applies_to_all = bool(re.search(r'\b(all|current appointments|my appointments)\b', lowered))
        symptom_terms = (
            r'\b(symptom|symptoms|breathlessness|shortness of breath|dry cough|cough|migraine|headache|dizziness|dizzy|'
            r'nausea|vomiting|rash|rashes|teary eyes|burning sensation|nose bleed|nosebleed|fatigue|fever|pain)\b'
        )
        if _is_pending_bulk_cancel_confirmation(context) and re.search(r'\b(yes|confirm|cancel all|all of them|all)\b', lowered):
            return DoctorIntentDecision(
                intent='cancel_appointment',
                rationale='Fallback bulk cancel confirmation intent.',
                applies_to_all=True,
                schedule_scope='general',
            )
        if re.search(r'\b(cancel|delete|remove)\b', lowered) and re.search(r'\bappointment(s)?\b', lowered):
            return DoctorIntentDecision(
                intent='cancel_appointment',
                rationale='Fallback cancel intent.',
                lookup_date=lookup_date,
                source_date=lookup_date,
                appointment_id=appointment_id,
                use_selected_appointment=True,
                applies_to_all=applies_to_all,
                patient_scope='selected_appointment',
                target_time=target_time,
            )
        if _is_pending_bulk_cancel_confirmation(context) and lookup_date:
            return DoctorIntentDecision(
                intent='cancel_appointment',
                rationale='Fallback selected-day bulk cancel confirmation intent.',
                lookup_date=lookup_date,
                source_date=lookup_date,
                applies_to_all=False,
                schedule_scope='date',
            )
        if re.search(r'\b(reschedule|move|change|amend|update)\b', lowered) and re.search(r'\bappointment(s)?\b', lowered):
            return DoctorIntentDecision(
                intent='amend_appointment',
                rationale='Fallback amend intent.',
                source_date=source_date,
                target_date=target_date or lookup_date,
                target_end_date=target_end_date,
                target_time=target_time,
                appointment_id=appointment_id,
                use_selected_appointment=True,
                applies_to_all=applies_to_all,
                patient_scope='selected_appointment',
            )
        if re.search(r'\b(research|treatment|diagnosis|disease|cause|causes|caused by|why|look for|look into)\b', lowered) or re.search(symptom_terms, lowered):
            research_scope = 'treatment' if 'treatment' in lowered else 'diagnosis' if ('diagnosis' in lowered or 'disease' in lowered or 'cause' in lowered or 'causes' in lowered or 'why' in lowered) else 'symptoms'
            patient_id = _extract_patient_id(user_query)
            patient_phone = _extract_phone(user_query)
            return DoctorIntentDecision(
                intent='research_symptoms',
                rationale='Fallback doctor research intent.',
                patient_id=patient_id,
                patient_phone=patient_phone,
                symptoms_query=user_query,
                patient_scope='direct_lookup' if (patient_id or patient_phone) else 'active_patient',
                research_scope=research_scope,
            )
        if re.search(r'\b(history|record|records|medical history|patient history)\b', lowered):
            patient_id = _extract_patient_id(user_query)
            patient_phone = _extract_phone(user_query)
            return DoctorIntentDecision(
                intent='view_patient_history',
                rationale='Fallback patient history intent.',
                patient_id=patient_id,
                patient_phone=patient_phone,
                patient_scope='direct_lookup' if (patient_id or patient_phone) else 'none',
            )
        if re.search(r'\b(appointment|appointments|schedule|today|tomorrow|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', lowered):
            if 'today' in lowered:
                schedule_scope = 'today'
            elif lookup_date:
                schedule_scope = 'date'
            elif 'week' in lowered:
                schedule_scope = 'week'
            else:
                schedule_scope = 'general'
            return DoctorIntentDecision(intent='show_schedule', rationale='Fallback doctor schedule intent.', lookup_date=lookup_date, schedule_scope=schedule_scope)
        return DoctorIntentDecision(intent='general_help', rationale='Fallback general help intent.')


def _extract_date(query: str) -> str | None:
    return resolve_single_date(query)


def _extract_phone(text: str) -> str | None:
    digits = ''.join(ch for ch in text if ch.isdigit())
    return digits if len(digits) >= 10 else None


def _extract_patient_id(text: str) -> str | None:
    match = re.search(r'\b([a-z]{3,}_[a-z0-9_]+)\b', text.lower())
    return match.group(1) if match else None


def _extract_appointment_id(text: str) -> str | None:
    match = re.search(r'\b((?:appt(?:_open(?:_auto)?)?_[a-z0-9_]+)|(?:slot_doc_[a-z0-9_]+))\b', text.lower())
    return match.group(1) if match else None


def _extract_time(text: str) -> str | None:
    twelve_hour = re.search(r'\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([ap])\.?m?\.?\b', text, flags=re.IGNORECASE)
    if twelve_hour:
        hour = int(twelve_hour.group(1))
        minute = int(twelve_hour.group(2) or '00')
        meridiem = twelve_hour.group(3).lower()
        if meridiem == 'p' and hour != 12:
            hour += 12
        if meridiem == 'a' and hour == 12:
            hour = 0
        return f'{hour:02d}:{minute:02d}'
    twenty_four_hour = re.search(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', text)
    if twenty_four_hour:
        return f"{int(twenty_four_hour.group(1)):02d}:{int(twenty_four_hour.group(2)):02d}"
    return None


def _extract_reschedule_dates(query: str) -> tuple[str | None, str | None, str | None]:
    lowered = query.lower()
    if ' to next week' in lowered or lowered.endswith('next week'):
        source = _extract_first_date(query)
        target = _next_business_week_range()
        return source, target[0], target[1]
    if re.search(r'\b(next week|this week|next month)\b', lowered):
        source = _extract_first_date(query)
        target_date, target_end_date = resolve_date_or_range(query)
        if target_date or target_end_date:
            return source, target_date, target_end_date
    if ' to ' in lowered:
        left, right = re.split(r'\bto\b', query, maxsplit=1, flags=re.IGNORECASE)
        source = _extract_first_date(left)
        target = _extract_date(right)
        if source or target:
            return source, target, None
    matches = re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b', query)
    if len(matches) >= 2:
        return matches[0], matches[-1], None
    source = _extract_first_date(query)
    target = _extract_date(query)
    if source and target and source != target:
        return source, target, None
    return None, target, None


def _extract_first_date(query: str) -> str | None:
    return resolve_single_date(query, prefer_first=True)


def _next_business_week_range() -> tuple[str, str]:
    from datetime import date, timedelta
    today = date.today()
    next_monday = today + timedelta(days=(7 - today.weekday()))
    next_saturday = next_monday + timedelta(days=5)
    return next_monday.isoformat(), next_saturday.isoformat()


def _is_pending_bulk_cancel_confirmation(context: dict[str, object]) -> bool:
    pending = context.get('pending_confirmation') or {}
    if isinstance(pending, dict):
        return pending.get('action_type') == 'bulk_cancel_doctor_appointments'
    return False


doctor_intent_service = DoctorIntentService()
