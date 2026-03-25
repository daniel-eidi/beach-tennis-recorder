"""
Asynchronous clip processing queue.

AGENT-03 · TASK-03-06

Processes clip tasks in a background thread so the camera / main thread
is never blocked.  Uses a bounded queue with overflow handling.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Default maximum number of pending tasks before overflow kicks in.
DEFAULT_MAX_QUEUE_SIZE = 50


def _structured_log(
    task: str,
    status: str,
    ms: Optional[float] = None,
    **extra: Any,
) -> None:
    entry: dict[str, Any] = {"agent": "03", "task": task, "status": status}
    if ms is not None:
        entry["ms"] = round(ms, 2)
    entry.update(extra)
    logger.info(json.dumps(entry))


@dataclass
class ClipTask:
    """A single clip processing task to be executed in the background."""

    buffer_path: str
    start_time: float
    end_time: float
    match_id: int | str
    rally_number: int
    callback: Optional[Callable[..., Any]] = None
    error_callback: Optional[Callable[[Exception], Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ClipQueue:
    """Thread-safe background queue for clip processing.

    Usage::

        def on_done(result):
            print("Clip ready:", result)

        q = ClipQueue(processor_fn=my_process_fn)
        q.start()
        q.enqueue(ClipTask(..., callback=on_done))
        # ... later ...
        q.stop()

    The *processor_fn* receives a :class:`ClipTask` and returns an
    arbitrary result that is forwarded to ``task.callback``.
    """

    def __init__(
        self,
        processor_fn: Callable[[ClipTask], Any],
        max_size: int = DEFAULT_MAX_QUEUE_SIZE,
        num_workers: int = 1,
    ) -> None:
        """
        Args:
            processor_fn: Callable that processes a ClipTask and returns a result.
            max_size: Maximum pending tasks. 0 = unlimited.
            num_workers: Number of worker threads.
        """
        self._processor_fn = processor_fn
        self._queue: queue.Queue[Optional[ClipTask]] = queue.Queue(maxsize=max_size)
        self._num_workers = num_workers
        self._workers: list[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
        self._tasks_processed = 0
        self._tasks_dropped = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background worker threads."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for i in range(self._num_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"clip-worker-{i}",
                    daemon=True,
                )
                t.start()
                self._workers.append(t)
        _structured_log("03-06", "ok", event="queue_started", workers=self._num_workers)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal workers to stop and wait for them to finish.

        Args:
            timeout: Max seconds to wait per worker thread.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Send sentinel values to unblock workers
        for _ in self._workers:
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:
                pass

        for w in self._workers:
            w.join(timeout=timeout)

        self._workers.clear()
        _structured_log(
            "03-06", "ok",
            event="queue_stopped",
            processed=self._tasks_processed,
            dropped=self._tasks_dropped,
        )

    def enqueue(self, task: ClipTask) -> bool:
        """Add a clip task to the processing queue.

        Returns:
            True if the task was enqueued, False if the queue is full
            (task is dropped).
        """
        try:
            self._queue.put_nowait(task)
            _structured_log(
                "03-06", "ok",
                event="task_enqueued",
                match_id=str(task.match_id),
                rally=task.rally_number,
                pending=self._queue.qsize(),
            )
            return True
        except queue.Full:
            self._tasks_dropped += 1
            _structured_log(
                "03-06", "warn",
                event="task_dropped_queue_full",
                match_id=str(task.match_id),
                rally=task.rally_number,
                dropped_total=self._tasks_dropped,
            )
            if task.error_callback:
                task.error_callback(
                    RuntimeError("Clip queue is full; task dropped")
                )
            return False

    @property
    def pending(self) -> int:
        """Number of tasks waiting to be processed."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tasks_processed(self) -> int:
        return self._tasks_processed

    @property
    def tasks_dropped(self) -> int:
        return self._tasks_dropped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Main loop executed by each worker thread."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                # Sentinel — exit
                break

            t0 = time.perf_counter()
            try:
                result = self._processor_fn(task)
                elapsed = (time.perf_counter() - t0) * 1000
                self._tasks_processed += 1
                _structured_log(
                    "03-06", "ok",
                    event="task_completed",
                    ms=elapsed,
                    match_id=str(task.match_id),
                    rally=task.rally_number,
                )
                if task.callback:
                    task.callback(result)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                _structured_log(
                    "03-06", "error",
                    event="task_failed",
                    ms=elapsed,
                    error=str(exc),
                    match_id=str(task.match_id),
                    rally=task.rally_number,
                )
                if task.error_callback:
                    task.error_callback(exc)
            finally:
                self._queue.task_done()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple demo with a no-op processor
    def demo_processor(task: ClipTask) -> str:
        time.sleep(0.1)  # simulate work
        return f"Processed rally {task.rally_number}"

    q = ClipQueue(processor_fn=demo_processor, max_size=10)
    q.start()

    for i in range(5):
        q.enqueue(ClipTask(
            buffer_path="/tmp/buffer.mp4",
            start_time=0.0,
            end_time=10.0,
            match_id=1,
            rally_number=i + 1,
            callback=lambda r: print(r),
        ))

    time.sleep(2)
    q.stop()
    print(f"Processed: {q.tasks_processed}, Dropped: {q.tasks_dropped}")
