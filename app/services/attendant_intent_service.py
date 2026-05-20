from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, Field

from app.services.date_resolution import resolve_date_or_range, resolve_single_date
from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class AttendantTask(BaseModel):
    action: str
    target_scope: str = 'single'
    patient_name: str | None = None
    patient_phone: str | None = None
    patient_id: str | None = None
    doctor_query: str | None = None
    target_doctor_query: str | None = None
    target_date: str | None = None
    target_end_date: str | None = None
    target_time: str | None = None
    source_date: str | None = None
    appointment_id: str | None = None
    edit_first_name: str | None = None
    edit_last_name: str | None = None
    edit_phone: str | None = None
    edit_address: str | None = None
    doctor_name: str | None = None
    doctor_phone: str | None = None
    doctor_email: str | None = None
    doctor_specialty: str | None = None
    doctor_gender: str | None = None
    history_update_text: str | None = None
    depends_on_previous: bool = False


class AttendantIntentDecision(BaseModel):
    tasks: list[AttendantTask] = Field(default_factory=list)
    rationale: str = ''


class AttendantIntentService:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('ATTENDANT_INTENT_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def classify(self, user_query: str, conversation_context_json: str = '') -> AttendantIntentDecision:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': load_prompt('attendant/intent_classifier.txt')},
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
                data = AttendantIntentDecision.model_validate(json.loads(content))
                if data.tasks:
                    return data
            except Exception:
                pass
        return self._fallback(user_query)

    def _fallback(self, user_query: str) -> AttendantIntentDecision:
        segments = [segment.strip() for segment in re.split(r'[.\n]+', user_query) if segment.strip()]
        tasks: list[AttendantTask] = []
        for segment in segments:
            task = self._fallback_single(segment, user_query=user_query)
            if task is not None:
                tasks.append(task)
        if tasks:
            return AttendantIntentDecision(tasks=tasks, rationale='Fallback parsed attendant tasks.')
        return AttendantIntentDecision(tasks=[AttendantTask(action='general_help', target_scope='none')], rationale='Fallback general help.')

    def _fallback_single(self, segment: str, *, user_query: str) -> AttendantTask | None:
        lowered = segment.lower()
        if re.search(r'\b(show|list|view|check)\b', lowered) and re.search(r'\bdoctors\b', lowered):
            return AttendantTask(action='show_doctors', target_scope='global')
        if re.search(r'\b(register|add|create)\b', lowered) and re.search(r'\bdoctor\b', lowered):
            return AttendantTask(
                action='register_doctor',
                target_scope='single',
                doctor_name=_extract_doctor_name(segment),
                doctor_phone=_extract_phone_update(segment),
                doctor_email=_extract_email(segment),
                doctor_specialty=_extract_specialty(segment),
                doctor_gender=_extract_gender(segment),
            )
        if re.search(r'\b(update|edit)\b', lowered) and re.search(r'\bdoctor\b', lowered):
            return AttendantTask(
                action='edit_doctor_details',
                target_scope='single',
                doctor_query=_extract_doctor_reference(segment),
                doctor_name=_extract_doctor_name_update(segment),
                doctor_phone=_extract_phone_update(segment),
                doctor_email=_extract_email(segment),
                doctor_specialty=_extract_specialty(segment),
                doctor_gender=_extract_gender(segment),
            )
        if re.search(r'\b(show|list|view|check)\b', lowered) and re.search(r'\b(active patients|registered patients|patients)\b', lowered):
            return AttendantTask(action='show_active_patients', target_scope='global')
        if re.search(r'\b(show|list|view|check)\b', lowered) and re.search(r'\b(open|available)\b', lowered) and re.search(r'\b(appointment|appointments|slot|slots)\b', lowered):
            target_date, target_end_date = _extract_date_or_range(segment)
            return AttendantTask(
                action='show_open_appointments',
                target_scope='global',
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=_extract_time(segment),
                doctor_query=_extract_doctor_or_specialty_query(segment),
            )
        if re.search(r'\b(show|list|view|check)\b', lowered) and re.search(r'\b(booked|scheduled|appointment|appointments)\b', lowered):
            target_date, target_end_date = _extract_date_or_range(segment)
            return AttendantTask(
                action='show_booked_appointments',
                target_scope='global',
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=_extract_time(segment),
                doctor_query=_extract_doctor_or_specialty_query(segment),
            )
        if re.search(r'\b(delete|remove)\b', lowered) and re.search(r'\bpatient\b', lowered):
            return AttendantTask(action='delete_patient', target_scope='single', **_extract_patient_reference(segment))
        if re.search(r'\b(update|edit)\b', lowered) and re.search(r'\b(patient|phone|address|name)\b', lowered):
            return AttendantTask(
                action='edit_patient_details',
                target_scope='single',
                edit_first_name=_extract_name_update(segment, first=True),
                edit_last_name=_extract_name_update(segment, first=False),
                edit_phone=_extract_phone_update(segment),
                edit_address=_extract_address_update(segment),
                **_extract_patient_reference(segment),
            )
        if re.search(r'\b(update|add|upload|save|record|write)\b', lowered) and re.search(r'\b(history|summary|report|follow[\s-]?up|diagnosis|note)\b', lowered):
            return AttendantTask(
                action='update_patient_history',
                target_scope='single',
                doctor_query=_extract_target_doctor_reference(segment) or _extract_doctor_or_specialty_query(segment),
                history_update_text=segment.strip(),
                **_extract_patient_reference(segment),
            )
        if re.search(r'\b(history|record|records|diagnosis|follow-up)\b', lowered):
            return AttendantTask(action='view_patient_history', target_scope='single', **_extract_patient_reference(segment))
        if re.search(r'\b(cancel|delete|remove)\b', lowered) and (
            re.search(r'\bappointment', lowered) or re.search(r'\bpat_[a-z0-9_]+\b', lowered)
        ):
            return AttendantTask(
                action='cancel_appointments',
                target_scope='batch' if re.search(r'\b(all|appointments)\b', lowered) and re.search(r'\bdoctor\b', lowered) else 'single',
                source_date=_extract_single_date(segment),
                doctor_query=_extract_doctor_or_specialty_query(segment),
                appointment_id=_extract_appointment_id(segment),
                **_extract_patient_reference(segment),
            )
        if re.search(r'\b(rebook|book|schedule)\b', lowered) and (
            re.search(r'\bappointment', lowered) or re.search(r'\bpat_[a-z0-9_]+\b', lowered)
        ):
            target_date, target_end_date = _extract_date_or_range(segment)
            return AttendantTask(
                action='book_appointments',
                target_scope='batch' if re.search(r'\bappointments\b', lowered) else 'single',
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=_extract_time(segment),
                doctor_query=_extract_target_doctor_reference(segment) or _extract_doctor_or_specialty_query(segment),
                appointment_id=_extract_appointment_id(segment),
                depends_on_previous=bool(re.search(r'\b(cancelled|canceled|them|those|re-book)\b', lowered)),
                **_extract_patient_reference(segment),
            )
        if re.search(r'\b(reschedule|move|change|amend)\b', lowered) and (
            re.search(r'\bappointment', lowered) or re.search(r'\bpat_[a-z0-9_]+\b', lowered)
        ):
            source_date, target_date, target_end_date = _extract_reschedule_dates(segment)
            return AttendantTask(
                action='reschedule_appointments',
                target_scope='batch' if re.search(r'\b(all|appointments)\b', lowered) else 'single',
                target_date=target_date,
                target_end_date=target_end_date,
                target_time=_extract_time(segment),
                source_date=source_date,
                doctor_query=_extract_target_doctor_reference(segment) or _extract_doctor_or_specialty_query(segment),
                appointment_id=_extract_appointment_id(segment),
                **_extract_patient_reference(segment),
            )
        return None


def _extract_date_or_range(query: str) -> tuple[str | None, str | None]:
    return resolve_date_or_range(query)


def _extract_single_date(query: str, *, first: bool | None = None) -> str | None:
    matches = re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b', query)
    if matches:
        if first is True:
            return matches[0]
        if first is False:
            return matches[-1]
        return matches[-1]
    return resolve_single_date(query, prefer_first=bool(first))


def _extract_reschedule_dates(query: str) -> tuple[str | None, str | None, str | None]:
    lowered = query.lower()
    if ' to next week' in lowered or lowered.endswith('next week'):
        source = _extract_single_date(query, first=True)
        target_date, target_end_date = _extract_date_or_range('next week')
        return source, target_date, target_end_date
    if ' to this week' in lowered or lowered.endswith('this week'):
        source = _extract_single_date(query, first=True)
        target_date, target_end_date = _extract_date_or_range('this week')
        return source, target_date, target_end_date
    if ' to next month' in lowered or lowered.endswith('next month'):
        source = _extract_single_date(query, first=True)
        target_date, target_end_date = _extract_date_or_range('next month')
        return source, target_date, target_end_date
    matches = re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b', query)
    if len(matches) >= 2:
        return matches[0], matches[-1], None
    natural_matches = re.findall(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(20\d{2}))?\b', query)
    if len(natural_matches) >= 2:
        return _extract_single_date(query, first=True), _extract_single_date(query, first=False), None
    target_date, target_end_date = _extract_date_or_range(query)
    return None, target_date, target_end_date


def _extract_appointment_id(query: str) -> str | None:
    match = re.search(r'\b((?:appt_[a-z0-9_]+)|(?:slot_doc_[a-z0-9_]+))\b', query, flags=re.IGNORECASE)
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


def _extract_patient_reference(query: str) -> dict[str, str]:
    patient_id_match = re.search(r'\b(pat_[a-z0-9_]+)\b', query, flags=re.IGNORECASE)
    if patient_id_match:
        return {'patient_id': patient_id_match.group(1)}
    phone_match = re.search(r'\b(?:\+?\d[\d\s-]{8,}\d)\b', query)
    if phone_match and re.search(r'\b(for|patient|record)\b', query, flags=re.IGNORECASE):
        digits = ''.join(ch for ch in phone_match.group(0) if ch.isdigit())
        if len(digits) >= 10:
            return {'patient_phone': digits}
    name_patterns = [
        r'\bupdate\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+?)(?=\s+(?:phone|address|name|record|history)\b|$)',
        r'\bdelete\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+?)(?=\s*$)',
        r'\bshow(?:\s+the)?\s+(?:history|records|record)\s+(?:for\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bfor\s+patient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return {'patient_name': match.group(1).strip()}
    return {}


def _extract_doctor_reference(query: str) -> str | None:
    patterns = [
        r'\bfor\s+doctor\s+["-]?\s*([A-Za-z][A-Za-z.\s]+?)(?:\s+from|\s+to|$)',
        r'\bfor\s+([A-Za-z][A-Za-z.\s]+?)(?:\s+from|\s+to|$)',
        r'\bdoctor\s+["-]?\s*([A-Za-z][A-Za-z.\s]+?)(?:\s+from|\s+to|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip('"').strip()
            if _looks_like_date_phrase(candidate):
                continue
            return candidate
    return None


def _extract_doctor_or_specialty_query(query: str) -> str | None:
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
    return _extract_doctor_reference(query)


def _extract_target_doctor_reference(query: str) -> str | None:
    patterns = [
        r'\bwith\s+doctor\s+["-]?\s*([A-Za-z][A-Za-z.\s]+?)(?:$|\s+for|\s+on|\s+at)',
        r'\bwith\s+["-]?\s*([A-Za-z][A-Za-z.\s]+?)(?:$|\s+for|\s+on|\s+at)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip('"').strip()
            if _looks_like_date_phrase(candidate):
                continue
            return candidate
    return None


def _looks_like_date_phrase(value: str) -> bool:
    lowered = value.lower()
    if any(token in lowered for token in ['today', 'tomorrow', 'week', 'month', 'slot', 'available']):
        return True
    return resolve_single_date(value) is not None


def _extract_phone_update(query: str) -> str | None:
    match = re.search(r'\bphone(?:\s+number)?\s+(?:to|as)\s+([+\d][\d\s-]{7,}\d)\b', query, flags=re.IGNORECASE)
    if not match:
        phone_match = re.search(r'\b(?:\+?\d[\d\s-]{8,}\d)\b', query)
        if phone_match and re.search(r'\bdoctor\b', query, flags=re.IGNORECASE):
            digits = ''.join(ch for ch in phone_match.group(0) if ch.isdigit())
            return digits if len(digits) >= 9 else None
        return None
    digits = ''.join(ch for ch in match.group(1) if ch.isdigit())
    return digits if len(digits) >= 9 else None


def _extract_email(query: str) -> str | None:
    match = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', query)
    return match.group(1) if match else None


def _extract_address_update(query: str) -> str | None:
    match = re.search(r'\baddress\s+(?:to|as)\s+(.+)$', query, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('.')
    return None


def _extract_name_update(query: str, *, first: bool) -> str | None:
    match = re.search(r'\b(?:first\s+name|last\s+name|name)\s+(?:to|as)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)', query, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    parts = value.split()
    if first:
        return parts[0] if parts else None
    if len(parts) > 1:
        return ' '.join(parts[1:])
    return None


def _extract_doctor_name(query: str) -> str | None:
    match = re.search(r'\bdoctor\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+)+)', query)
    if match:
        return match.group(1).strip()
    name_match = re.search(r'\bregister\s+(?:doctor\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+)+)', query)
    return name_match.group(1).strip() if name_match else None


def _extract_doctor_name_update(query: str) -> str | None:
    match = re.search(r'\bname\s+(?:to|as)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+)+)', query)
    return match.group(1).strip() if match else None


def _extract_specialty(query: str) -> str | None:
    match = re.search(r'\b(?:specialty|specialisation|specialization)\s+(?:to|as)?\s*([A-Za-z][A-Za-z\s]+)', query, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('.')
    tail_match = re.search(r'\bdoctor\b.*?\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*$', query)
    return tail_match.group(1).strip() if tail_match and re.search(r'\bmedicine|cardio|pulmo|ortho|surgeon', tail_match.group(1), flags=re.IGNORECASE) else None


def _extract_gender(query: str) -> str | None:
    match = re.search(r'\b(male|female|other)\b', query, flags=re.IGNORECASE)
    return match.group(1).title() if match else None


attendant_intent_service = AttendantIntentService()
