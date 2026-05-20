from __future__ import annotations

import json
from functools import lru_cache

from openai import OpenAI

from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class AdminResponseGenerator:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('ADMIN_RESPONSE_MODEL', 'gpt-4o-mini')
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
                        {'role': 'system', 'content': load_prompt('admin/response_synthesizer.txt')},
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
        label = payload.get('label') or 'records'
        count = payload.get('count', 0)
        actor = payload.get('actor_filter')
        if actor:
            return f'I found {count} {label} for actor {actor}.'
        return f'I found {count} {label}.'


admin_response_generator = AdminResponseGenerator()
