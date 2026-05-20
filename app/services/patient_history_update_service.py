from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, Field

from app.schemas.domain import Patient
from app.services.patient_summary_parser import patient_summary_parser
from app.services.prompt_loader import load_prompt
from app.services.record_summary_generator import record_summary_generator
from app.utils.config import get_env


class PatientHistoryUpdate(BaseModel):
    title: str = 'Follow-up Visit Update'
    visit_date: str | None = None
    subjective: str | None = None
    objective: str | None = None
    assessment: str | None = None
    plan: str | None = None
    primary_conditions: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    cleared_conditions: list[str] = Field(default_factory=list)
    latest_visit_summary: str | None = None
    summary_text: str | None = None


class PatientHistoryUpdateService:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('ATTENDANT_HISTORY_UPDATE_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=256)
    def extract_update(
        self,
        *,
        patient_json: str,
        current_summary_json: str,
        user_query: str,
        report_text: str,
        report_name: str,
    ) -> PatientHistoryUpdate:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': load_prompt('attendant/history_update_parser.txt')},
                        {
                            'role': 'user',
                            'content': json.dumps(
                                {
                                    'patient': json.loads(patient_json),
                                    'current_summary': json.loads(current_summary_json),
                                    'user_query': user_query,
                                    'uploaded_report_name': report_name or None,
                                    'uploaded_report_text': report_text or None,
                                }
                            ),
                        },
                    ],
                )
                content = response.choices[0].message.content or '{}'
                data = PatientHistoryUpdate.model_validate(json.loads(content))
                if data.visit_date and not self._supports_explicit_date(user_query, report_text):
                    data = data.model_copy(update={'visit_date': None})
                return data
            except Exception:
                pass
        return self._fallback(
            patient=Patient.model_validate(json.loads(patient_json)),
            current_summary=json.loads(current_summary_json),
            user_query=user_query,
            report_text=report_text,
            report_name=report_name,
        )

    def _fallback(
        self,
        *,
        patient: Patient,
        current_summary: dict[str, object],
        user_query: str,
        report_text: str,
        report_name: str,
    ) -> PatientHistoryUpdate:
        combined = ' '.join(part.strip() for part in [user_query, report_text] if part and part.strip())
        insights = patient_summary_parser.parse_summary(combined)
        cleared_conditions = self._extract_cleared_conditions(combined, patient, current_summary)
        assessment = self._extract_assessment(combined)
        plan = self._extract_plan(combined)
        visit_summary = record_summary_generator.summarize_visit(assessment, plan) or None
        title = 'Uploaded Doctor Report' if report_name else 'Follow-up Visit Update'
        summary_text = self._build_summary_text(patient, insights, cleared_conditions, visit_summary, combined)
        return PatientHistoryUpdate(
            title=title,
            subjective=user_query.strip() or None,
            assessment=assessment,
            plan=plan,
            primary_conditions=insights.primary_conditions,
            chronic_conditions=insights.chronic_conditions,
            allergies=insights.allergies,
            cleared_conditions=cleared_conditions,
            latest_visit_summary=visit_summary,
            summary_text=summary_text,
        )

    def _extract_cleared_conditions(
        self,
        text: str,
        patient: Patient,
        current_summary: dict[str, object],
    ) -> list[str]:
        lowered = text.lower()
        if not re.search(r'\b(clear|cleared|resolved|no longer|free of|recovered)\b', lowered):
            return []
        known_conditions = {
            *patient.primary_conditions,
            *patient.chronic_conditions,
            *(current_summary.get('primary_conditions') or []),
            *(current_summary.get('chronic_conditions') or []),
        }
        matches: list[str] = []
        for condition in known_conditions:
            if condition and condition.lower() in lowered:
                matches.append(condition)
        return sorted(set(matches))

    def _extract_assessment(self, text: str) -> str | None:
        match = re.search(r'(?:assessment|diagnosis|impression)\s*[:\-]\s*([^.]+)', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if re.search(r'\b(cleared|resolved|no active infection|stable)\b', text, flags=re.IGNORECASE):
            cleaned = ' '.join(text.split())
            return cleaned[:180].rstrip('.')
        return None

    def _extract_plan(self, text: str) -> str | None:
        match = re.search(r'(?:plan|next step|follow[- ]?up)\s*[:\-]\s*([^.]+)', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _build_summary_text(
        self,
        patient: Patient,
        insights,
        cleared_conditions: list[str],
        visit_summary: str | None,
        combined: str,
    ) -> str | None:
        resulting_conditions = [
            condition
            for condition in patient.primary_conditions
            if condition not in cleared_conditions
        ]
        for condition in insights.primary_conditions:
            if condition not in resulting_conditions:
                resulting_conditions.append(condition)
        if visit_summary and resulting_conditions:
            return f"{patient.full_name}'s current history reflects {', '.join(resulting_conditions)}. {visit_summary}"
        if visit_summary:
            return f"{patient.full_name}'s current history has been updated. {visit_summary}"
        cleaned = ' '.join(combined.split())
        return cleaned[:220].rstrip() + ('...' if len(cleaned) > 220 else '') if cleaned else None

    def _supports_explicit_date(self, user_query: str, report_text: str) -> bool:
        combined = f'{user_query}\n{report_text}'
        if re.search(r'\b(20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b', combined):
            return True
        if re.search(r'\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', combined, flags=re.IGNORECASE):
            return True
        if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', combined, flags=re.IGNORECASE):
            return True
        return False


patient_history_update_service = PatientHistoryUpdateService()
