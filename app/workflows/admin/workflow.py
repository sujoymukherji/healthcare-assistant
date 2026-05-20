from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.repositories.local_database import get_local_database_repository
from app.repositories.sample_data_repository import get_sample_repository
from app.schemas.domain import ActorType
from app.services.admin_intent_service import AdminIntentDecision, admin_intent_service
from app.services.admin_response_generator import admin_response_generator
from app.services.langsmith_service import langsmith_service


class AdminWorkflowState(TypedDict, total=False):
    session_id: str
    actor: ActorType
    user_query: str
    selected_view: str | None
    actor_filter: str | None
    conversation_context: dict[str, object] | None
    interaction_rows: list[dict[str, object]]
    planner_trace_rows: list[dict[str, object]]
    system_error_rows: list[dict[str, object]]
    langsmith_run_rows: list[dict[str, object]]
    final_response: str | None
    workflow_type: str | None
    intent_decision: AdminIntentDecision | None
    planner_source: str | None
    langsmith_enabled: bool
    langsmith_run_id: str | None
    langsmith_run_url: str | None


class AdminAssistantWorkflow:
    def __init__(self) -> None:
        self.database = get_local_database_repository()
        self.repository = get_sample_repository()
        self.graph = self._build_graph()

    def run(self, state: AdminWorkflowState) -> AdminWorkflowState:
        state.setdefault('actor', ActorType.IT_ADMIN)
        state['langsmith_enabled'] = langsmith_service.enabled
        try:
            run_id: str | None = None
            with langsmith_service.trace_context(
                'admin_workflow',
                run_type='chain',
                inputs={'query': state.get('user_query'), 'selected_view': state.get('selected_view')},
                metadata={'session_id': state.get('session_id')},
                tags=['healthcare-assistant', 'admin-workflow'],
            ) as run:
                if run is not None:
                    run_id = str(run.id)
                result = self.graph.invoke(state)
            if run_id:
                result['langsmith_run_id'] = run_id
                langsmith_service.flush()
                result['langsmith_run_url'] = langsmith_service.get_verified_run_url(run_id)
            return result
        except Exception as error:
            raise

    def stream(self, state: AdminWorkflowState) -> Iterator[AdminWorkflowState]:
        yield from self.graph.stream(state, stream_mode='values')

    def _build_graph(self):
        builder = StateGraph(AdminWorkflowState)
        builder.add_node('initialize', self._initialize)
        builder.add_node('classify_intent', self._classify_intent)
        builder.add_node('view_interaction_logs', self._view_interaction_logs)
        builder.add_node('view_planner_traces', self._view_planner_traces)
        builder.add_node('view_system_errors', self._view_system_errors)
        builder.add_node('view_langsmith_runs', self._view_langsmith_runs)

        builder.add_edge(START, 'initialize')
        builder.add_edge('initialize', 'classify_intent')
        builder.add_conditional_edges(
            'classify_intent',
            self._route_after_intent,
            {
                'view_interaction_logs': 'view_interaction_logs',
                'view_planner_traces': 'view_planner_traces',
                'view_system_errors': 'view_system_errors',
                'view_langsmith_runs': 'view_langsmith_runs',
                'done': END,
            },
        )
        for node in ('view_interaction_logs', 'view_planner_traces', 'view_system_errors', 'view_langsmith_runs'):
            builder.add_edge(node, END)
        return builder.compile()

    def _initialize(self, state: AdminWorkflowState) -> AdminWorkflowState:
        state.setdefault('conversation_context', {})
        state.setdefault('interaction_rows', [])
        state.setdefault('planner_trace_rows', [])
        state.setdefault('system_error_rows', [])
        state.setdefault('langsmith_run_rows', [])
        state.setdefault('workflow_type', None)
        state.setdefault('planner_source', None)
        return state

    def _classify_intent(self, state: AdminWorkflowState) -> AdminWorkflowState:
        query = state.get('user_query', '')
        context = state.get('conversation_context') or {}
        decision = admin_intent_service.classify(query, json.dumps(context, sort_keys=True))
        if state.get('actor_filter'):
            decision.actor_filter = state['actor_filter']
        state['intent_decision'] = decision
        state['workflow_type'] = decision.intent
        state['planner_source'] = 'llm_or_fallback'
        return state

    def _view_interaction_logs(self, state: AdminWorkflowState) -> AdminWorkflowState:
        actor = state.get('intent_decision').actor_filter if state.get('intent_decision') else None
        rows = self.database.list_interaction_logs(actor=actor, for_date=date.today())
        state['interaction_rows'] = rows
        state['final_response'] = admin_response_generator.synthesize(
            workflow_type='view_interaction_logs',
            payload_json=json.dumps({'count': len(rows), 'label': 'interaction log entries', 'actor_filter': actor}),
        )
        return state

    def _view_planner_traces(self, state: AdminWorkflowState) -> AdminWorkflowState:
        actor = state.get('intent_decision').actor_filter if state.get('intent_decision') else None
        rows = self._annotate_planner_trace_rows(self.database.list_planner_traces(actor=actor, for_date=date.today()))
        state['planner_trace_rows'] = rows
        state['final_response'] = admin_response_generator.synthesize(
            workflow_type='view_planner_traces',
            payload_json=json.dumps({'count': len(rows), 'label': 'planner trace entries', 'actor_filter': actor}),
        )
        return state

    def _view_system_errors(self, state: AdminWorkflowState) -> AdminWorkflowState:
        actor = state.get('intent_decision').actor_filter if state.get('intent_decision') else None
        rows = self.database.list_system_errors(actor=actor, for_date=date.today())
        state['system_error_rows'] = rows
        state['final_response'] = admin_response_generator.synthesize(
            workflow_type='view_system_errors',
            payload_json=json.dumps({'count': len(rows), 'label': 'system error entries', 'actor_filter': actor}),
        )
        return state

    def _view_langsmith_runs(self, state: AdminWorkflowState) -> AdminWorkflowState:
        actor = state.get('intent_decision').actor_filter if state.get('intent_decision') else None
        rows = self.database.list_langsmith_runs(actor=actor, for_date=date.today())
        state['langsmith_run_rows'] = rows
        state['final_response'] = admin_response_generator.synthesize(
            workflow_type='view_langsmith_runs',
            payload_json=json.dumps({'count': len(rows), 'label': 'LangSmith run entries', 'actor_filter': actor}),
        )
        return state

    def _route_after_intent(self, state: AdminWorkflowState) -> str:
        decision = state.get('intent_decision')
        if decision is None:
            state['final_response'] = 'Select an admin view or ask for logs, traces, errors, or LangSmith runs.'
            return 'done'
        valid = {'view_interaction_logs', 'view_planner_traces', 'view_system_errors', 'view_langsmith_runs'}
        return decision.intent if decision.intent in valid else 'done'

    def _annotate_planner_trace_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        annotated_rows: list[dict[str, object]] = []
        for row in rows:
            planner_output = row.get('Planner Output') or {}
            extracted = self._extract_trace_scheduling_fields(planner_output)
            annotated = dict(row)
            annotated.update(extracted)
            annotated_rows.append(annotated)
        return annotated_rows

    def _extract_trace_scheduling_fields(self, planner_output: object) -> dict[str, str]:
        if not isinstance(planner_output, dict):
            return {
                'Resolved Date': '',
                'Resolved End Date': '',
                'Resolved Time': '',
                'Doctor / Specialty': '',
                'Appointment Ref': '',
            }
        if isinstance(planner_output.get('tasks'), list):
            return self._extract_attendant_task_fields(planner_output['tasks'])
        doctor_or_patient_date = planner_output.get('target_date') or planner_output.get('lookup_date') or planner_output.get('source_date') or ''
        return {
            'Resolved Date': str(doctor_or_patient_date or ''),
            'Resolved End Date': str(planner_output.get('target_end_date') or ''),
            'Resolved Time': str(planner_output.get('target_time') or ''),
            'Doctor / Specialty': str(
                planner_output.get('doctor_query')
                or planner_output.get('target_doctor_query')
                or planner_output.get('doctor_name')
                or planner_output.get('doctor_id')
                or ''
            ),
            'Appointment Ref': str(planner_output.get('appointment_id') or ''),
        }

    def _extract_attendant_task_fields(self, tasks: list[object]) -> dict[str, str]:
        dates: list[str] = []
        end_dates: list[str] = []
        times: list[str] = []
        doctors: list[str] = []
        appointment_refs: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            for value, bucket in (
                (task.get('target_date') or task.get('source_date'), dates),
                (task.get('target_end_date'), end_dates),
                (task.get('target_time'), times),
                (task.get('doctor_query') or task.get('target_doctor_query'), doctors),
                (task.get('appointment_id'), appointment_refs),
            ):
                if value and value not in bucket:
                    bucket.append(str(value))
        return {
            'Resolved Date': ', '.join(dates),
            'Resolved End Date': ', '.join(end_dates),
            'Resolved Time': ', '.join(times),
            'Doctor / Specialty': ', '.join(doctors),
            'Appointment Ref': ', '.join(appointment_refs),
        }
