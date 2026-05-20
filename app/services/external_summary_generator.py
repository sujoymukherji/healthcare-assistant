from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI

from app.schemas.contracts import SearchMedlinePlusTopicsOutput
from app.schemas.domain import ActorType
from app.ui.components import clean_text
from app.utils.config import get_env


class ExternalSummaryGenerator:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('MEDLINE_SUMMARY_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def summarize(self, actor: str, workflow_type: str, user_query: str, payload_json: str) -> str | None:
        style_instructions = self._style_for_actor(actor, workflow_type)
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
                                'You are summarizing MedlinePlus evidence for a healthcare assistant. '
                                'Return strict JSON with one key: external_summary. '
                                'Only include information directly relevant to the current question. '
                                'Do not dump raw MedlinePlus snippets. '
                                'Do not include definitions, background physiology, category tables, or textbook explanations. '
                                'If the evidence does not add value, return an empty string.'
                            ),
                        },
                        {
                            'role': 'user',
                            'content': json.dumps(
                                {
                                    'actor': actor,
                                    'workflow_type': workflow_type,
                                    'style_instructions': style_instructions,
                                    'user_query': user_query,
                                    'medlineplus_results': json.loads(payload_json),
                                }
                            ),
                        },
                    ],
                )
                content = response.choices[0].message.content or '{}'
                data = json.loads(content)
                summary = str(data.get('external_summary', '')).strip()
                return clean_text(summary) or None
            except Exception:
                pass
        return self._heuristic_summary(actor, workflow_type, user_query, json.loads(payload_json))

    def _style_for_actor(self, actor: str, workflow_type: str) -> str:
        if actor == ActorType.DOCTOR.value or workflow_type == 'doctor_research':
            return (
                'Doctor style: provide a concise, structured markdown summary with 2-4 short sections or bullets. '
                'Focus on clinically relevant symptom/treatment context, red flags, and follow-up considerations. '
                'Keep it compact and evidence-oriented.'
            )
        if actor == ActorType.ATTENDANT.value or workflow_type == 'attendant_research':
            return (
                'Attendant style: provide a short operational summary in plain language. '
                'Keep it to 2-3 short sentences focused on relevance and next-step direction.'
            )
        return (
            'Patient style: provide a short plain-language summary with at most two concise sentences. '
            'Prefer care-path relevance, actionability, and urgency guidance when appropriate.'
        )

    def _heuristic_summary(self, actor: str, workflow_type: str, user_query: str, results: list[dict[str, str]]) -> str | None:
        query_tokens = {token for token in re.findall(r'[a-zA-Z]+', user_query.lower()) if len(token) > 3}
        kept: list[str] = []
        for item in results[:3]:
            title = clean_text(item.get('title', ''))
            snippet = clean_text(item.get('snippet', ''))
            if not snippet and title:
                snippet = title
            sentences = re.split(r'(?<=[.!?])\s+', snippet)
            relevant_sentences: list[str] = []
            for sentence in sentences:
                lowered = sentence.lower()
                if any(
                    phrase in lowered
                    for phrase in (
                        'what is',
                        'how is',
                        'blood pressure category',
                        'types of high blood pressure',
                        'systolic',
                        'diastolic',
                    )
                ):
                    continue
                if query_tokens and any(token in lowered for token in query_tokens):
                    relevant_sentences.append(sentence.strip())
                elif any(flag in lowered for flag in ('seek medical care', 'call 911', 'emergency', 'urgent', 'doctor', 'provider')):
                    relevant_sentences.append(sentence.strip())
            summary_bits = relevant_sentences[:1] or ([f'{title}: {sentences[0].strip()}'] if sentences and title else [])
            for bit in summary_bits:
                cleaned = clean_text(bit)
                if cleaned:
                    kept.append(cleaned)
            if len(kept) >= 3:
                break
        if not kept:
            return None
        if actor == ActorType.DOCTOR.value or workflow_type == 'doctor_research':
            title = clean_text(results[0].get('title', '')) if results else ''
            sections: list[str] = []
            if title:
                sections.append(f"**Relevant Topic**\n- {title}")
            sections.append("**Clinical Relevance**\n- " + "\n- ".join(kept[:2]))
            if len(kept) > 2:
                sections.append(f"**Follow-up Consideration**\n- {kept[2]}")
            return "\n\n".join(section for section in sections if section)
        if actor == ActorType.ATTENDANT.value or workflow_type == 'attendant_research':
            return ' '.join(kept[:2])
        return ' '.join(kept[:2])


def build_medline_payload(results: SearchMedlinePlusTopicsOutput | None) -> str:
    if not results:
        return '[]'
    payload = [
        {
            'title': item.title,
            'snippet': item.snippet or '',
            'source_group': item.source_group,
            'url': item.url,
        }
        for item in results.results[:5]
    ]
    return json.dumps(payload, ensure_ascii=True)


external_summary_generator = ExternalSummaryGenerator()
