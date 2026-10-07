"""Runs engine work off the UI thread."""
from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QThread, Signal

from ..core.engine import Cancelled
from ..core.safety import BlockedPrompt


class Task(QThread):
    progress = Signal(str, int, int)
    image = Signal(object, str, str)  # PIL image, saved path, parameters
    pose = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[["Task"], object]):
        super().__init__()
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            self.done.emit(self._fn(self))
        except Cancelled:
            self.failed.emit("Остановлено.")
        except (BlockedPrompt, ValueError, FileNotFoundError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                self.failed.emit("Не хватило видеопамяти. Уменьшите размер/апскейл или включите режим «Экономия VRAM».")
            else:
                self.failed.emit(str(e))
                if not isinstance(e, (BlockedPrompt, ValueError, FileNotFoundError)):
                    traceback.print_exc()
        except Exception as e:  # noqa: BLE001 — report anything to the user instead of crashing
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
