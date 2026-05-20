from __future__ import annotations

import json
from functools import lru_cache

from openai import OpenAI

from app.services.prompt_loader import load_prompt
from app.utils.config import get_env


class DoctorResponseGenerator:
    def __init__(self) -> None:
        self.api_key = get_env('OPENAI_API_KEY')
        self.model = get_env('DOCTOR_RESPONSE_MODEL', 'gpt-4o-mini')
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
                        {'role': 'system', 'content': load_prompt('doctor/response_synthesizer.txt')},
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
        schedule_scope = payload.get('schedule_scope') or 'none'
        patient_scope = payload.get('patient_scope') or 'none'
        research_scope = payload.get('research_scope') or 'none'
        if workflow_type == 'show_schedule':
            if payload.get('message'):
                base = str(payload['message'])
                count = payload.get('appointment_count')
                has_appointments = bool(payload.get('has_appointments'))
                if not has_appointments:
                    return base
                if count is not None and schedule_scope in {'today', 'week', 'date', 'general'}:
                    return f"{base} {count} appointment(s) found."
                return base
            return 'Doctor schedule prepared.'
        if workflow_type == 'view_patient_history':
            patient_name = payload.get('patient_name') or 'Unknown patient'
            lines = [f'**Patient**\n- {patient_name}']
            if payload.get('patient_id'):
                lines[0] += f" ({payload['patient_id']})"
            if payload.get('latest_visit_summary'):
                lines.append(f"**Latest History**\n- {payload['latest_visit_summary']}")
            if payload.get('primary_conditions'):
                lines.append('**Known Conditions**\n- ' + '\n- '.join(payload['primary_conditions']))
            if payload.get('chronic_conditions'):
                lines.append('**Chronic Conditions**\n- ' + '\n- '.join(payload['chronic_conditions']))
            if payload.get('allergies'):
                lines.append('**Allergies**\n- ' + '\n- '.join(payload['allergies']))
            return '\n\n'.join(lines)
        if workflow_type == 'research_symptoms':
            sections: list[str] = []
            if payload.get('patient_name'):
                patient_line = f"{payload['patient_name']}"
                if payload.get('patient_id'):
                    patient_line += f" ({payload['patient_id']})"
                sections.append(f"**Patient Context**\n- {patient_line}")
            if payload.get('latest_visit_summary'):
                sections.append(f"**Relevant History**\n- {payload['latest_visit_summary']}")
            if payload.get('history_match'):
                sections.append(f"**History Match**\n- {payload['history_match']}")
            if payload.get('external_summary'):
                heading = 'External Evidence'
                if research_scope != 'none':
                    heading = f'External Evidence ({research_scope.title()})'
                sections.append(f"**{heading}**\n{payload['external_summary']}")
            if payload.get('next_review_points'):
                sections.append('**Next Review Points**\n- ' + '\n- '.join(payload['next_review_points']))
            message = '\n\n'.join(section for section in sections if section) or str(payload.get('message') or 'Doctor research summary prepared.')
            if research_scope != 'none':
                return f"**Research Focus**\n- {research_scope}\n\n{message}"
            return message
        if workflow_type in {'amend_appointment', 'cancel_appointment'}:
            return str(payload.get('message') or 'The appointment request was processed.')
        return str(payload.get('message') or 'Doctor request processed.')


doctor_response_generator = DoctorResponseGenerator()
