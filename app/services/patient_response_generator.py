from __future__ import annotations

import json
import re

from openai import OpenAI

from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class PatientResponseGenerator:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('PATIENT_RESPONSE_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    def synthesize(
        self,
        *,
        user_query: str,
        route_type: str,
        specialty: str,
        diagnosis_hint: str | None,
        prior_plan: str | None,
        recent_visit_summary: str | None,
        external_summary: str | None,
        booking_requested: bool,
        booking_confirmed: bool = False,
        confirmed_appointment: dict[str, str] | None = None,
    ) -> str:
        payload = {
            'user_query': user_query,
            'route_type': route_type,
            'specialty': specialty.replace('_', ' '),
            'diagnosis_hint': self._clean_diagnosis(diagnosis_hint),
            'prior_plan': self._clean_plan(prior_plan),
            'recent_visit_summary': self._clean_sentence(recent_visit_summary),
            'external_summary': self._clean_sentence(external_summary),
            'booking_requested': booking_requested,
            'booking_confirmed': booking_confirmed,
            'confirmed_appointment': confirmed_appointment or {},
        }
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {
                            'role': 'system',
                            'content': load_prompt('patient/response_synthesizer.txt'),
                        },
                        {'role': 'user', 'content': json.dumps(payload)},
                    ],
                )
                content = response.choices[0].message.content or '{}'
                data = json.loads(content)
                text = self._clean_sentence(str(data.get('response_text', '')).strip(), preserve_multi=True)
                if text:
                    return text
            except Exception:
                pass
        return self._heuristic_response(**payload)

    def _heuristic_response(
        self,
        *,
        user_query: str,
        route_type: str,
        specialty: str,
        diagnosis_hint: str | None,
        prior_plan: str | None,
        recent_visit_summary: str | None,
        external_summary: str | None,
        booking_requested: bool,
        booking_confirmed: bool = False,
        confirmed_appointment: dict[str, str] | None = None,
    ) -> str:
        parts: list[str] = []
        if route_type == 'historical_match':
            parts.append(f"Your symptoms may overlap with a previous issue, so a follow-up with {specialty} is a reasonable next step.")
            if diagnosis_hint:
                parts.append(f"Your record most closely aligns with {diagnosis_hint}.")
            if recent_visit_summary:
                parts.append(f"Your recent care was focused on {recent_visit_summary}.")
            elif prior_plan:
                parts.append(f"Your earlier care plan focused on {prior_plan}.")
        else:
            parts.append(f"These symptoms should be reviewed with {specialty} so a clinician can evaluate the cause properly.")
            if recent_visit_summary:
                parts.append(f"Your recent care was focused on {recent_visit_summary}.")
            elif prior_plan:
                parts.append(f"Your earlier care plan focused on {prior_plan}.")
        if external_summary:
            parts.append(external_summary)
        if booking_confirmed and confirmed_appointment:
            doctor_name = confirmed_appointment.get('doctor_name') or 'your clinician'
            appointment_date = confirmed_appointment.get('date') or 'the scheduled date'
            appointment_time = confirmed_appointment.get('time') or 'the scheduled time'
            appointment_id = confirmed_appointment.get('appointment_id') or ''
            parts = [part for part in parts if 'let me know if you would like to book' not in part.lower()]
            parts.append(
                f"Your appointment is confirmed for {appointment_date} at {appointment_time} with {doctor_name}."
            )
            if appointment_id:
                parts.append(f"Appointment reference: {appointment_id}.")
        elif booking_requested:
            parts = [part for part in parts if 'let me know if you would like to book' not in part.lower()]
        return ' '.join(part.strip() for part in parts if part).strip()

    def _clean_diagnosis(self, text: str | None) -> str | None:
        if not text:
            return None
        cleaned = re.sub(r'^\s*Diagnosis:\s*', '', text, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned or None

    def _clean_plan(self, text: str | None) -> str | None:
        if not text:
            return None
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = re.split(r'(?<=[.!?])\s+', cleaned)[0].strip()
        cleaned = cleaned.rstrip('.')
        if len(cleaned) > 90:
            cleaned = cleaned[:87].rstrip() + '...'
        return cleaned or None

    def _clean_sentence(self, text: str | None, preserve_multi: bool = False) -> str | None:
        if not text:
            return None
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if preserve_multi:
            return cleaned or None
        first = re.split(r'(?<=[.!?])\s+', cleaned)[0].strip()
        return first or None


patient_response_generator = PatientResponseGenerator()
