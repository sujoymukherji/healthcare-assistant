from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from app.services.date_resolution import resolve_date_or_range, resolve_single_date
from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class PatientIntentDecision(BaseModel):
    intent: str
    rationale: str = ''
    needs_medline: bool = False
    booking_requested: bool = False
    appointment_id: str | None = None
    target_date: str | None = None
    target_end_date: str | None = None
    target_time: str | None = None
    doctor_query: str | None = None
    appointment_scope: str = 'none'
    symptoms_present: bool = False
    booking_followup_action: str = 'none'
    clarification_focus: str = 'none'


class PatientIntentService:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('PATIENT_INTENT_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def classify(self, user_query: str, conversation_context_json: str = '') -> PatientIntentDecision:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': load_prompt('patient/intent_classifier.txt')},
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
                return PatientIntentDecision.model_validate(json.loads(content))
            except Exception:
                pass
        return self._fallback(user_query, conversation_context_json)

    def _fallback(self, user_query: str, conversation_context_json: str = '') -> PatientIntentDecision:
        lowered = user_query.strip().lower()
        context = json.loads(conversation_context_json) if conversation_context_json else {}
        booking_context = context.get('booking_context') or {}
        target_date, target_end_date = _extract_date_or_range(user_query)
        target_time = _extract_time(user_query)
        doctor_query = _extract_doctor_query(user_query)
        appointment_id = _extract_appointment_id(user_query)
        if lowered in {'yes', 'yes please', 'please do', 'go ahead', 'sure', 'okay', 'ok'} and booking_context:
            if booking_context.get('clarification_type') in {'confirm_preferred_doctor_next_available', 'confirm_preferred_doctor_for_requested_date'}:
                return PatientIntentDecision(
                    intent='book_appointment',
                    rationale='Fallback booking confirmation from prior clarification.',
                    booking_requested=True,
                    doctor_query=booking_context.get('suggested_doctor_query'),
                    target_date=booking_context.get('requested_date'),
                    target_end_date=booking_context.get('requested_end_date'),
                    target_time=booking_context.get('requested_time'),
                    booking_followup_action='confirm_suggested_doctor',
                )
        if booking_context and re.search(r'\b(another|different|someone else|other)\s+doctor\b', lowered):
            return PatientIntentDecision(
                intent='book_appointment',
                rationale='Fallback booking follow-up changing doctor preference.',
                booking_requested=True,
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=target_time,
                booking_followup_action='change_doctor_preference',
                clarification_focus='doctor_or_department',
            )
        if booking_context and re.search(r'\b(another|different|other)\s+(department|specialty)\b', lowered):
            return PatientIntentDecision(
                intent='book_appointment',
                rationale='Fallback booking follow-up changing department preference.',
                booking_requested=True,
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=target_time,
                booking_followup_action='change_doctor_preference',
                clarification_focus='doctor_or_department',
            )
        if re.search(r'\b(cancel|delete|remove)\b', lowered):
            return PatientIntentDecision(intent='cancel_appointment', rationale='Fallback cancel intent.', appointment_id=appointment_id)
        if re.search(r'\b(medical history|clinical history|visit history|past visits|previous diagnoses|treatment history)\b', lowered):
            return PatientIntentDecision(intent='show_medical_history', rationale='Fallback medical history intent.')
        if re.search(r'\b(book|schedule)\b', lowered):
            symptoms_present = bool(re.search(r'\b(symptom|symptoms|pain|fever|cough|headache|nausea|dizzy|restless|rash|nosebleed|fatigue)\b', lowered))
            return PatientIntentDecision(
                intent='book_appointment',
                rationale='Fallback booking intent.',
                booking_requested=True,
                needs_medline=symptoms_present,
                appointment_id=appointment_id,
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=target_time,
                doctor_query=doctor_query,
                symptoms_present=symptoms_present,
                booking_followup_action='provide_constraints' if any([doctor_query, target_date, target_end_date, target_time]) else 'needs_clarification',
                clarification_focus='doctor_or_department' if not any([doctor_query, target_date, target_end_date, target_time]) else 'none',
            )
        if re.search(r'\b(reschedule|amend|change|move|update)\b', lowered):
            return PatientIntentDecision(
                intent='amend_appointment',
                rationale='Fallback reschedule intent.',
                appointment_id=appointment_id,
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=target_time,
                doctor_query=doctor_query,
            )
        if re.search(r'\b(open|available)\b', lowered) and re.search(r'\b(appointment|appointments|slot|slots)\b', lowered):
            return PatientIntentDecision(
                intent='show_open_appointments',
                rationale='Fallback open appointment intent.',
                appointment_scope='open',
                appointment_id=appointment_id,
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=target_time,
                doctor_query=doctor_query,
            )
        if re.search(r'\b(past|previous|prior|history)\b', lowered) and re.search(r'\b(appointment|appointments)\b', lowered):
            return PatientIntentDecision(intent='show_past_appointments', rationale='Fallback past appointment intent.', appointment_scope='past')
        if re.search(r'\b(current|scheduled|upcoming|my|show|check)\b', lowered) and re.search(r'\b(appointment|appointments)\b', lowered):
            if doctor_query or target_date or target_end_date:
                return PatientIntentDecision(
                    intent='show_open_appointments',
                    rationale='Fallback availability lookup intent.',
                    appointment_scope='open',
                    appointment_id=appointment_id,
                    target_date=target_date,
                    target_end_date=target_end_date,
                    target_time=target_time,
                    doctor_query=doctor_query,
                )
            return PatientIntentDecision(intent='show_past_appointments', rationale='Fallback current appointment intent.', appointment_scope='current')
        if re.search(r'\b(symptom|symptoms|pain|fever|cough|headache|nausea|dizzy|restless|rash)\b', lowered):
            return PatientIntentDecision(intent='symptom_research', rationale='Fallback symptom intent.', needs_medline=True, symptoms_present=True)
        if _looks_like_phone_number(user_query):
            return PatientIntentDecision(intent='identify_patient', rationale='Fallback phone identification intent.')
        return PatientIntentDecision(intent='general_help', rationale='Fallback general help intent.')


def _extract_date_or_range(query: str) -> tuple[str | None, str | None]:
    return resolve_date_or_range(query)


def _extract_single_date(query: str) -> str | None:
    return resolve_single_date(query)


def _extract_doctor_query(query: str) -> str | None:
    specialty_match = re.search(
        r'\b(orthopedic|orthopaedic|ortho|general medicine|general med|gen med|general physician|medicine|pulmonology|pulmonologist|pulmonary|pulm|cardiology|cardiologist|cardiac|cardio|dermatology|dermatologist|neurology|neurologist|ent|ophthalmology|ophthalmologist)\b',
        query,
        flags=re.IGNORECASE,
    )
    if specialty_match:
        value = specialty_match.group(1).strip().lower()
        aliases = {
            'orthopaedic': 'orthopedic',
            'ortho': 'orthopedic',
            'general physician': 'general medicine',
            'general med': 'general medicine',
            'gen med': 'general medicine',
            'medicine': 'general medicine',
            'pulmonologist': 'pulmonology',
            'pulmonary': 'pulmonology',
            'pulm': 'pulmonology',
            'cardiologist': 'cardiology',
            'cardiac': 'cardiology',
            'cardio': 'cardiology',
            'dermatologist': 'dermatology',
            'neurologist': 'neurology',
            'ophthalmologist': 'ophthalmology',
        }
        return aliases.get(value, value)
    doctor_match = re.search(r'\bwith\s+(?:dr\.?\s+|doctor\s+)?([A-Z][A-Za-z.\s]+)$', query)
    if doctor_match:
        return doctor_match.group(1).strip().strip()
    direct_booking_match = re.search(
        r'\b(?:book|schedule)(?:\s+(?:an?\s+)?appointment)?\s+(?:with\s+)?(?:dr\.?\s+|doctor\s+)?([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z.]*){0,3})\s+(?:at|on|for)\b',
        query,
    )
    if direct_booking_match:
        return direct_booking_match.group(1).strip()
    return None


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


def _looks_like_phone_number(text: str) -> bool:
    digits = ''.join(ch for ch in text if ch.isdigit())
    return len(digits) >= 10


patient_intent_service = PatientIntentService()
