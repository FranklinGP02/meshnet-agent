"""Buffer circular de logs del agente para telemetría remota (read-only).

Un `logging.Handler` que retiene en memoria las últimas N líneas emitidas por los
loggers `meshnet.*`. El loop de heartbeat las drena y las manda piggyback al
coordinador, que las almacena append-only y las muestra en el dashboard. Así un
colaborador no técnico no tiene que copiar su terminal para depurar su nodo.

Invariante de timestamps: el agente NUNCA manda wall-clock. Cada registro guarda
el `time.monotonic()` de cuando se emitió; al drenar se calcula `age_s` (segundos
desde la emisión) y el coordinador reconstruye created_at = now - age_s.

Thread-safety: `logging.Handler.emit` puede invocarse desde cualquier hilo
(el envío de heartbeat corre en el hilo principal), por eso el deque se protege
con un lock propio además del de Handler.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from meshnet_shared.constants import (
    MAX_LOG_LINES_PER_HEARTBEAT,
    MAX_LOG_MESSAGE_CHARS,
)
from meshnet_shared.schemas import LogLine


class RingLogHandler(logging.Handler):
    """Retiene las últimas `capacity` líneas; `drain()` las extrae como LogLine."""

    def __init__(
        self,
        capacity: int = MAX_LOG_LINES_PER_HEARTBEAT,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._monotonic = monotonic
        # (emitted_monotonic, level, logger, message). maxlen descarta lo más viejo.
        self._buf: deque[tuple[float, str, str, str]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        # Nunca dejes que el logging tumbe al agente: ante cualquier fallo de
        # formateo, descarta la línea silenciosamente (es telemetría best-effort).
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{self.format(record)}" if self.formatter else message
            # Truncado defensivo: el coordinador también capa, pero ahorra payload.
            message = message[:MAX_LOG_MESSAGE_CHARS]
            item = (self._monotonic(), record.levelname[:16], record.name[:64], message)
            with self._lock:
                self._buf.append(item)
        except Exception:
            # El logging jamás debe propagar: best-effort, descarta la línea.
            self.handleError(record)

    def drain(self) -> tuple[LogLine, ...]:
        """Vacía el buffer y devuelve las líneas como LogLine con age_s calculado."""
        now = self._monotonic()
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
        lines: list[LogLine] = []
        for emitted_at, level, logger_name, message in items:
            if not message:  # LogLine exige min_length=1; salta líneas vacías
                continue
            lines.append(
                LogLine(
                    level=level,
                    logger=logger_name,
                    message=message,
                    age_s=max(0.0, now - emitted_at),
                )
            )
        return tuple(lines)


def install_ring_handler(
    capacity: int = MAX_LOG_LINES_PER_HEARTBEAT,
    *,
    logger_name: str = "meshnet",
    level: int = logging.INFO,
) -> RingLogHandler:
    """Engancha un RingLogHandler al logger `meshnet` y devuelve el handler.

    Idempotente: si ya hay uno instalado, lo reutiliza en vez de duplicar."""
    target = logging.getLogger(logger_name)
    for existing in target.handlers:
        if isinstance(existing, RingLogHandler):
            return existing
    handler = RingLogHandler(capacity)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    target.addHandler(handler)
    return handler
