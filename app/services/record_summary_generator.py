from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI

from app.utils.config import get_env


class RecordSummaryGenerator:
    """Builds short patient-friendly summaries for medical visits."""

    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('RECORD_SUMMARY_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def summarize_visit(self, assessment: str | None, plan: str | None) -> str:
        assessment_text = ' '.join((assessment or '').split())
        plan_text = ' '.join((plan or '').split())
        if not assessment_text and not plan_text:
            return ''
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
                                'Summarize the medical visit in one short sentence for a patient-facing UI. '
                                'Return strict JSON with one key: visit_summary. Keep it under 24 words, '
                                'mention the diagnosis or visit reason and the next care step, and avoid jargon when possible.'
                            ),
                        },
                        {
                            'role': 'user',
                            'content': json.dumps({'assessment': assessment_text, 'plan': plan_text}),
                        },
                    ],
                )
                content = response.choices[0].message.content or '{}'
                data = json.loads(content)
                summary = ' '.join(str(data.get('visit_summary', '')).split())
                if summary:
                    return summary
            except Exception:
                pass
        return self._heuristic_summary(assessment_text, plan_text)

    def _heuristic_summary(self, assessment: str, plan: str) -> str:
        diagnosis = self._clean_assessment(assessment)
        next_step = self._clean_plan(plan)
        if diagnosis and next_step:
            return f'{diagnosis}. {next_step}'
        return diagnosis or next_step or ''

    def _clean_assessment(self, value: str) -> str:
        cleaned = value.replace('Diagnosis:', '').replace('DIAGNOSIS:', '').strip()
        cleaned = re.sub(r'\[[^\]]+\]', '', cleaned).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if not cleaned:
            return ''
        if len(cleaned) > 100:
            cleaned = cleaned.split('.')[0].strip()
        return cleaned.rstrip('.')

    def _clean_plan(self, value: str) -> str:
        cleaned = re.sub(r'\s+', ' ', value).strip().rstrip('.')
        if not cleaned:
            return ''
        first_sentence = re.split(r'(?<=[.!?])\s+', cleaned)[0].strip().rstrip('.')
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117].rstrip() + '...'
        return first_sentence


record_summary_generator = RecordSummaryGenerator()
