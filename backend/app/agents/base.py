"""Shared agent plumbing: execution context, typed task errors and tool-call recording."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..services import store
from ..services.llm import Providers
from ..services.runtime_settings import RuntimeSettings
from ..services.store import TaskView


class TaskError(RuntimeError):
    """A task failed. `kind` drives recovery routing; `retryable` is the agent's own view."""

    def __init__(self, message: str, kind: str = "unknown", retryable: bool = True):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class Cancelled(RuntimeError):
    pass


@dataclass
class AgentContext:
    run_id: str
    simulated: bool
    providers: Providers
    cancel: asyncio.Event
    settings: RuntimeSettings
    run_params: dict
    task: TaskView
    outputs: dict[str, dict] = field(default_factory=dict)  # completed task key -> output
    tasks: list[TaskView] = field(default_factory=list)

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def upstream(self, action: str) -> list[tuple[TaskView, dict]]:
        """Outputs of completed direct dependencies with the given action."""
        by_key = {t.key: t for t in self.tasks}
        return [(by_key[k], self.outputs[k]) for k in self.task.depends_on
                if k in by_key and by_key[k].action == action and k in self.outputs]

    def upstream_any(self, action: str) -> list[tuple[TaskView, dict]]:
        """Outputs of completed tasks anywhere in the run with the given action."""
        return [(t, self.outputs[t.key]) for t in self.tasks if t.action == action and t.key in self.outputs]

    def activity(self, text: str):
        return _Activity(self, text)

    async def tool(self, name: str, args: dict, fn: Callable[[], Awaitable[Any]],
                   summarize: Callable[[Any], str] = lambda r: "ok", simulated: bool | None = None) -> Any:
        """Run one tool call with persisted start/finish records. Never reports success unless fn returned."""
        self.check_cancel()
        sim = self.simulated if simulated is None else simulated
        tid = store.begin_tool(self.run_id, self.task.id, name, args, sim)
        t0 = time.perf_counter()
        try:
            res = await fn()
        except BaseException as e:
            store.end_tool(self.run_id, tid, self.task.id, name, ok=False, error=f"{type(e).__name__}: {e}"[:500],
                           duration_ms=int((time.perf_counter() - t0) * 1000))
            raise
        store.end_tool(self.run_id, tid, self.task.id, name, ok=True, summary=summarize(res),
                       duration_ms=int((time.perf_counter() - t0) * 1000))
        return res


class _Activity:
    def __init__(self, ctx: AgentContext, text: str):
        self.ctx, self.text, self.aid = ctx, text, ""

    async def __aenter__(self):
        self.aid = store.begin_activity(self.ctx.run_id, self.ctx.task.id, self.ctx.task.agent, self.text)
        return self

    async def __aexit__(self, *exc):
        store.end_activity(self.ctx.run_id, self.aid, self.ctx.task.id, self.ctx.task.agent)
        return False
