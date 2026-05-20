from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from typing import Any

import langsmith as ls
from langsmith import Client
from langsmith.run_helpers import trace
from langsmith.utils import LangSmithConflictError, LangSmithNotFoundError

from app.utils.config import get_env


class LangSmithService:
    """Optional LangSmith tracing helper."""

    def __init__(self) -> None:
        self.api_key = get_env("LANGSMITH_API_KEY")
        self.project = get_env("LANGSMITH_PROJECT", "healthcare-assistant")
        self.endpoint = get_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self._client: Client | None = None
        self._project_checked = False

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self) -> Client | None:
        if not self.enabled:
            return None
        if self._client is None:
            self._client = Client(api_key=self.api_key, api_url=self.endpoint)
        return self._client

    def ensure_project(self) -> bool:
        if not self.enabled or self.client is None:
            return False
        if self._project_checked:
            return True
        try:
            self.client.read_project(project_name=self.project)
        except LangSmithNotFoundError:
            try:
                self.client.create_project(project_name=self.project)
            except LangSmithConflictError:
                pass
        self._project_checked = True
        return True

    def trace_context(
        self,
        name: str,
        run_type: str = "chain",
        *,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        if not self.enabled or self.client is None:
            return contextlib.nullcontext(None)
        self.ensure_project()
        stack = contextlib.ExitStack()
        stack.enter_context(
            ls.tracing_context(
                enabled=True,
                project_name=self.project,
                client=self.client,
            )
        )
        run = stack.enter_context(
            trace(
                name=name,
                run_type=run_type,
                inputs=inputs,
                metadata=metadata,
                tags=tags,
                project_name=self.project,
                client=self.client,
            )
        )
        return _TraceContext(stack, run)

    def get_run_url(self, run) -> str | None:
        if not run or self.client is None:
            return None
        try:
            self.ensure_project()
            return self.client.get_run_url(run=run, project_name=self.project)
        except Exception:
            return None

    def get_verified_run_url(self, run_id: str | None, *, retries: int = 3, delay_seconds: float = 0.5) -> str | None:
        if not run_id or self.client is None:
            return None
        self.ensure_project()
        for attempt in range(retries):
            try:
                run = self.client.read_run(run_id)
                return self.client.get_run_url(run=run, project_name=self.project)
            except Exception:
                if attempt < retries - 1:
                    time.sleep(delay_seconds)
        return None

    def flush(self) -> None:
        if self.client is None:
            return
        try:
            result = self.client.flush()
            if inspect.isawaitable(result):
                try:
                    asyncio.run(result)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(result)
                    finally:
                        loop.close()
        except Exception:
            return


class _TraceContext:
    def __init__(self, stack: contextlib.ExitStack, run) -> None:
        self._stack = stack
        self._run = run

    def __enter__(self):
        return self._run

    def __exit__(self, exc_type, exc, tb):
        return self._stack.__exit__(exc_type, exc, tb)


langsmith_service = LangSmithService()
