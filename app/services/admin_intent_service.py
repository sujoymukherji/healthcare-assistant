from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class AdminIntentDecision(BaseModel):
    intent: str
    rationale: str = ''
    actor_filter: str | None = None


class AdminIntentService:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('ADMIN_INTENT_MODEL', 'gpt-4o-mini')
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    @lru_cache(maxsize=512)
    def classify(self, user_query: str, conversation_context_json: str = '') -> AdminIntentDecision:
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={'type': 'json_object'},
                    messages=[
                        {'role': 'system', 'content': load_prompt('admin/intent_classifier.txt')},
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
                return AdminIntentDecision.model_validate(json.loads(content))
            except Exception:
                pass
        return self._fallback(user_query)

    def _fallback(self, user_query: str) -> AdminIntentDecision:
        lowered = user_query.strip().lower()
        actor_filter = _extract_actor_filter(lowered)
        if re.search(r'\b(langsmith|trace url|run url|run id)\b', lowered):
            return AdminIntentDecision(intent='view_langsmith_runs', rationale='Fallback LangSmith intent.', actor_filter=actor_filter)
        if re.search(r'\b(error|errors|failure|failures|exception|exceptions|issue|issues)\b', lowered):
            return AdminIntentDecision(intent='view_system_errors', rationale='Fallback error intent.', actor_filter=actor_filter)
        if re.search(r'\b(planner|trace|traces|routing|decision|workflow)\b', lowered):
            return AdminIntentDecision(intent='view_planner_traces', rationale='Fallback planner trace intent.', actor_filter=actor_filter)
        if re.search(r'\b(log|logs|interaction|interactions|chat)\b', lowered):
            return AdminIntentDecision(intent='view_interaction_logs', rationale='Fallback interaction log intent.', actor_filter=actor_filter)
        return AdminIntentDecision(intent='general_help', rationale='Fallback general help intent.', actor_filter=actor_filter)


def _extract_actor_filter(lowered: str) -> str | None:
    for actor in ('patient', 'doctor', 'attendant', 'it_admin'):
        if actor.replace('_', ' ') in lowered or actor in lowered:
            return actor
    return None


admin_intent_service = AdminIntentService()
