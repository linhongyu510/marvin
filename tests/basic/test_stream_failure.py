"""Regression tests for failure propagation through the public streaming APIs.

`run_tasks_stream`, `Task.run_stream`, and `run_stream` accept
`raise_on_failure`. A failure the orchestrator raises must reach the caller
after the events are drained, matching `run_tasks_async`; `raise_on_failure=False`
stays silent. Failures happen during execution via a public tool, never by
pre-marking the task.
"""

import asyncio

import pytest

import marvin
from marvin.engine.events import OrchestratorEndEvent, OrchestratorErrorEvent


def _failing_task() -> marvin.Task:
    """Build a task whose public tool marks it failed during execution."""
    holder: dict = {}

    async def failer() -> str:
        await holder["task"].mark_failed("boom")
        return "marked"

    task = marvin.Task("do it", allow_fail=True, tools=[failer])
    holder["task"] = task
    return task


def _pending_background_tasks() -> list:
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current and not t.done()]


class TestRunTasksStreamFailure:
    async def test_raise_on_failure_propagates_after_events(self, test_model):
        task = _failing_task()
        events = []
        with pytest.raises(ValueError, match="boom"):
            async for event in marvin.run_tasks_stream([task], raise_on_failure=True):
                events.append(event)

        # Error event and terminal event are still yielded before the raise.
        assert events[-2].type == "orchestrator-error"
        assert events[-1].type == "orchestrator-end"
        assert task.is_failed()

    async def test_raise_on_failure_false_stays_silent(self, test_model):
        task = _failing_task()
        events = []
        async for event in marvin.run_tasks_stream([task], raise_on_failure=False):
            events.append(event)

        assert task.is_failed()
        assert not any(isinstance(e, OrchestratorErrorEvent) for e in events)
        assert events[-1].type == "orchestrator-end"


class TestTaskRunStreamFailure:
    async def test_task_run_stream_raise_true_propagates(self, test_model):
        task = _failing_task()
        events = []
        with pytest.raises(ValueError, match="boom"):
            async for event in task.run_stream(raise_on_failure=True):
                events.append(event)

        assert events[-2].type == "orchestrator-error"
        assert events[-1].type == "orchestrator-end"
        assert task.is_failed()

    async def test_task_run_stream_raise_false_stays_silent(self, test_model):
        task = _failing_task()
        events = []
        async for event in task.run_stream(raise_on_failure=False):
            events.append(event)

        assert task.is_failed()
        assert not any(isinstance(e, OrchestratorErrorEvent) for e in events)
        assert events[-1].type == "orchestrator-end"


class TestStreamSuccessAndCleanup:
    async def test_success_stream_completes_without_raise(self, test_model):
        events = []
        async for event in marvin.run_stream("hello"):
            events.append(event)

        assert events[0].type == "orchestrator-start"
        assert isinstance(events[-1], OrchestratorEndEvent)

    async def test_early_aclose_cancels_without_propagating_error(self, test_model):
        async def slow() -> str:
            await asyncio.sleep(10)
            return "ok"

        generator = marvin.run_stream("go", tools=[slow])
        async for event in generator:
            if event.type == "tool-call":
                break
        # Early close cancels the background orchestrator; it must not raise a
        # background exception and must leave no pending task.
        await generator.aclose()
        await asyncio.sleep(0.3)
        assert _pending_background_tasks() == []

    async def test_concurrent_failure_and_success_groups(self, test_model):
        async def failing(True_: bool):
            task = _failing_task()
            events = []
            try:
                async for event in task.run_stream(raise_on_failure=True_):
                    events.append(event.type)
            except ValueError as exc:
                return ("error", str(exc), events)
            return ("ok", None, events)

        async def success(i: int):
            events = []
            async for event in marvin.run_stream(f"hello {i}"):
                events.append(event.type)
            return ("ok", None, events)

        results = await asyncio.gather(
            failing(True),
            failing(False),
            success(1),
            success(2),
        )
        await asyncio.sleep(0.3)

        statuses = [r[0] for r in results]
        assert statuses.count("error") == 1
        assert statuses.count("ok") == 3
        # The silent failure still reaches its terminal event; successes end clean.
        for _, _, events in results:
            assert events[-1] == "orchestrator-end"
        assert _pending_background_tasks() == []
