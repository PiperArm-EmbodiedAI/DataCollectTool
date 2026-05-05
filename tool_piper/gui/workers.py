from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class TaskCancelledError(Exception):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TaskCancelledError("Task cancelled by user")


class WorkerSignals(QObject):
    message = Signal(str)
    progress = Signal(int, int, str, str)
    result = Signal(object)
    error = Signal(str)
    cancelled = Signal(str)
    finished = Signal()


class TaskWorker(QObject):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.cancel_token = CancellationToken()
        self.signals = WorkerSignals()

    def cancel(self) -> None:
        self.cancel_token.cancel()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except TaskCancelledError as exc:
            self.signals.cancelled.emit(str(exc))
        except Exception as exc:
            from tool_piper.gui.formatting import format_exception

            self.signals.error.emit(format_exception(exc))
        finally:
            self.signals.finished.emit()


def run_in_thread(worker: TaskWorker) -> QThread:
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.signals.finished.connect(thread.quit)
    worker.signals.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
