from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable
from uuid import uuid4

from engine.storage.system_store import SystemStore


class TaskManager:
    def __init__(self, store: SystemStore) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ashare-task")
        self._futures: dict[str, Future[Any]] = {}

    def submit(
        self,
        task_type: str,
        mode: str | None,
        fn: Callable[[Callable[[float, str], None]], dict[str, Any]],
    ) -> dict[str, Any]:
        task_id = uuid4().hex[:12]
        self.store.create_task(task_id, task_type, mode)

        def progress(progress_value: float, message: str) -> None:
            self.store.update_task(
                task_id,
                status="running",
                progress=progress_value,
                message=message,
            )

        def runner() -> dict[str, Any]:
            self.store.update_task(
                task_id,
                status="running",
                progress=0.01,
                message="starting",
            )
            try:
                result = fn(progress)
            except Exception as exc:  # pragma: no cover - exercised through API tests
                self.store.update_task(
                    task_id,
                    status="failed",
                    progress=1.0,
                    message="failed",
                    error=str(exc),
                )
                raise
            self.store.update_task(
                task_id,
                status="completed",
                progress=1.0,
                message="completed",
                result=result,
            )
            return result

        self._futures[task_id] = self.executor.submit(runner)
        task = self.store.get_task(task_id)
        return task or {"task_id": task_id}

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self.store.get_task(task_id)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_tasks(limit)

