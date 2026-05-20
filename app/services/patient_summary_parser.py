from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, Field

from app.utils.config import get_env


class PatientSummaryInsights(BaseModel):
    primary_conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)


class PatientSummaryParser:
    """Parses workbook summary text into structured patient profile attributes."""

    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('SUMMARY_PARSER_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=256)
    def parse_summary(self, summary: str) -> PatientSummaryInsights:
        cleaned = ' '.join((summary or '').split())
        if not cleaned:
            return PatientSummaryInsights()
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {
                            'role': 'system',
                            'content': (
                                'Extract only explicitly supported medical profile facts from the summary. '
                                'Return strict JSON with keys primary_conditions, allergies, chronic_conditions. '
                                'Use arrays of short strings. If the summary says the patient is healthy or gives '
                                'no evidence for a field, return an empty array for that field. Do not infer or invent.'
                            ),
                        },
                        {'role': 'user', 'content': cleaned},
                    ],
                )
                content = response.choices[0].message.content or '{}'
                data = json.loads(content)
                return PatientSummaryInsights.model_validate(data)
            except Exception:
                pass
        return self._heuristic_parse(cleaned)

    def _heuristic_parse(self, summary: str) -> PatientSummaryInsights:
        lowered = summary.lower()
        if any(token in lowered for token in ('fit and healthy', 'doing well', 'no current complaints')):
            return PatientSummaryInsights()

        primary_conditions: list[str] = []
        chronic_conditions: list[str] = []
        allergies: list[str] = []

        allergy_match = re.search(r'(?:allerg(?:y|ies)|allergic to)\s*[:\-]?\s*([^.]+)', summary, flags=re.IGNORECASE)
        if allergy_match:
            raw_items = re.split(r',|/| and ', allergy_match.group(1))
            allergies = [item.strip(' .') for item in raw_items if item.strip(' .')]
            if any(item.lower() in {'none', 'no known allergies', 'nkda'} for item in allergies):
                allergies = []

        known_conditions = [
            'hypertension',
            'upper respiratory infection',
            'type 2 diabetes mellitus',
            'diabetes',
            'pcos',
            'migraine',
            'migraine headaches',
            'kidney disease',
            'asthma',
        ]
        for condition in known_conditions:
            if condition not in lowered:
                continue
            label = condition.title() if condition != 'pcos' else 'PCOS'
            if label == 'Type 2 Diabetes Mellitus':
                primary_conditions.append(label)
                chronic_conditions.append(label)
            elif condition in {'hypertension', 'kidney disease', 'asthma', 'pcos', 'migraine', 'migraine headaches', 'diabetes'}:
                primary_conditions.append(label)
                chronic_conditions.append(label)
            else:
                primary_conditions.append(label)

        diagnosis_match = re.search(r'diagnosis\s*:\s*(.+?)(?:\.\s*plan:|\.\s*return|$)', summary, flags=re.IGNORECASE)
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
            diagnosis = re.sub(r'\s*\([A-Z]\d+(?:\.\d+)?\)\s*$', '', diagnosis).strip()
            if diagnosis and diagnosis not in primary_conditions:
                primary_conditions.append(diagnosis)
                if any(token in diagnosis.lower() for token in ('diabetes', 'hypertension', 'migraine', 'pcos', 'kidney')):
                    chronic_conditions.append(diagnosis)

        return PatientSummaryInsights(
            primary_conditions=sorted(set(primary_conditions)),
            allergies=sorted(set(allergies)),
            chronic_conditions=sorted(set(chronic_conditions)),
        )


patient_summary_parser = PatientSummaryParser()
