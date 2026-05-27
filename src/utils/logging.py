from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional


LogSink = Callable[[str], None]


@dataclass(frozen=True)
class Logger:
    sink: Optional[LogSink] = None

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        if self.sink is not None:
            self.sink(line)
        else:
            print(line, flush=True)
