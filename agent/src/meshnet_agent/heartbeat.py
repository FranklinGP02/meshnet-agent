"""Loop de heartbeat del agente.

Resiliencia (regla del plan): el agente NUNCA muere por un error de red.
Retry con backoff exponencial + jitter ante fallos; para limpio ante 401
(key revocada). Las duraciones se miden con time.monotonic(); el agente jamás
manda timestamps de pared (los pone el coordinador).
"""

import logging
import random
import time
from collections.abc import Callable

import httpx

from meshnet_agent import tailscale
from meshnet_agent.config import AgentConfig
from meshnet_agent.hardware import read_metrics
from meshnet_agent.logbuffer import RingLogHandler
from meshnet_agent.roles import AgentState
from meshnet_agent.worker import measure_peer_latencies
from meshnet_shared.constants import HEARTBEAT_INTERVAL_S, NodeRole
from meshnet_shared.schemas import (
    HeartbeatRequest,
    HeartbeatResponse,
    LogLine,
    PeerInfo,
    PeerLatency,
)

log = logging.getLogger("meshnet.heartbeat")

_MAX_BACKOFF_S = HEARTBEAT_INTERVAL_S * 3


def _pct(value: float | None) -> float:
    """Clamp a [0,100]: psutil puede reportar >100% en algunos multinúcleo,
    lo que dispararía un ValidationError del schema."""
    return min(100.0, max(0.0, value or 0.0))


def build_heartbeat(
    state: AgentState | None = None,
    peer_latencies: tuple[PeerLatency, ...] = (),
    recent_logs: tuple[LogLine, ...] = (),
    tailscale_ip: str | None = None,
) -> HeartbeatRequest:
    m = read_metrics()
    return HeartbeatRequest(
        cpu_pct=_pct(m["cpu_pct"]),
        ram_pct=_pct(m["ram_pct"]),
        gpu_pct=_pct(m["gpu_pct"]) if m["gpu_pct"] is not None else None,
        ram_free_gb=max(0.0, m["ram_free_gb"] or 0.0),
        vram_free_gb=m["vram_free_gb"],
        role=state.role if state else NodeRole.IDLE,
        cluster_id=state.cluster_id if state else None,
        peer_latencies=peer_latencies,
        llama_proc_alive=state.llama_proc_alive if state else False,
        recent_logs=recent_logs,
        tailscale_ip=tailscale_ip,
    )


def send_heartbeat(
    client: httpx.Client, api_key: str, payload: HeartbeatRequest
) -> HeartbeatResponse:
    response = client.post(
        "/api/v1/nodes/heartbeat",
        # content= no añade Content-Type por sí solo; sin esto el coordinador
        # devuelve 422 ("Input should be a valid dictionary") en vez de aceptar
        # el heartbeat — verificado en producción (mismo bug que en register).
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        content=payload.model_dump_json(),
    )
    response.raise_for_status()
    return HeartbeatResponse.model_validate_json(response.content)


class AuthRevokedError(RuntimeError):
    """La API key dejó de ser válida (401): parar el loop limpiamente."""


def run_loop(
    config: AgentConfig,
    *,
    state: AgentState | None = None,
    log_handler: RingLogHandler | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    max_iterations: int | None = None,
) -> None:
    """Bucle principal. max_iterations acota la ejecución en tests. Si se pasa
    `state`, el agente reporta su rol/proceso real y aplica los comandos de cluster
    que llegan en la respuesta del heartbeat. Si se pasa `log_handler`, drena sus
    líneas y las manda piggyback (telemetría read-only)."""
    backoff = 1.0
    iterations = 0
    peer_directory: tuple[PeerInfo, ...] = ()
    with httpx.Client(base_url=config.coordinator_url, timeout=10.0) as client:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            start = monotonic()
            try:
                if state and peer_directory:
                    latencies = measure_peer_latencies(peer_directory)
                else:
                    latencies = ()
                recent_logs = log_handler.drain() if log_handler else ()
                # Detectada en cada ciclo (no solo al registrar): la IP de
                # Tailscale puede cambiar (reinstalación, cambio de cuenta/red) y
                # el coordinador debe enterarse, o sigue repartiendo a los peers
                # una IP muerta para siempre (bloquea cualquier clúster).
                current_ip = tailscale.detect_self_ip()
                response = send_heartbeat(
                    client,
                    config.api_key,
                    build_heartbeat(state, latencies, recent_logs, current_ip),
                )
                if state is not None:
                    state.apply_commands(response.commands)
                    peer_directory = response.peer_directory  # medir en la próxima vuelta
                backoff = 1.0  # éxito → resetea el backoff
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == httpx.codes.UNAUTHORIZED:
                    log.error("API key rechazada (401): deteniendo el agente")
                    raise AuthRevokedError from exc
                log.warning("heartbeat %s; reintento en %.1fs", exc, backoff)
                sleep(backoff + random.uniform(0, backoff * 0.3))
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue
            except httpx.HTTPError as exc:
                log.warning("heartbeat falló (%s); reintento en %.1fs", exc, backoff)
                sleep(backoff + random.uniform(0, backoff * 0.3))
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue
            except Exception:
                # Fallo de hardware (psutil) o serialización: el agente NO debe
                # morir; logueamos con traceback y reintentamos con backoff.
                log.error("error inesperado en heartbeat", exc_info=True)
                sleep(backoff + random.uniform(0, backoff * 0.3))
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
                continue
            # Cadencia estable: descuenta el tiempo ya gastado en el envío.
            elapsed = monotonic() - start
            sleep(max(0.0, HEARTBEAT_INTERVAL_S - elapsed))
