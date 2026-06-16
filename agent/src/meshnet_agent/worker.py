"""Medición de latencias a otros nodos del tailnet (ADR-009).

El coordinador entrega el `peer_directory` en la respuesta del heartbeat; el
agente mide el RTT a cada peer y lo reporta en el siguiente heartbeat para que
el planner forme clústeres geo-conscientes. Nunca propaga errores (un peer
inalcanzable se omite, no tumba el loop)."""

import logging
import socket
import time
from collections.abc import Callable

import httpx

from meshnet_agent.config import AgentConfig
from meshnet_agent.roles import AgentState
from meshnet_shared.constants import RPC_PORT, ConnectionMode, NodeRole
from meshnet_shared.schemas import JobPayload, JobResultRequest, PeerInfo, PeerLatency

log = logging.getLogger("meshnet.worker")

_PROBE_TIMEOUT_S = 2.0


def _measure_one(ip: str, port: int) -> float | None:
    """RTT aproximado vía tiempo de establecimiento TCP (ms), o None si falla."""
    start = time.monotonic()
    try:
        with socket.create_connection((ip, port), timeout=_PROBE_TIMEOUT_S):
            return (time.monotonic() - start) * 1000.0
    except OSError:
        return None


def measure_peer_latencies(
    peer_directory: tuple[PeerInfo, ...],
    port: int = RPC_PORT,
    peer_modes: dict[str, ConnectionMode] | None = None,
) -> tuple[PeerLatency, ...]:
    """Mide RTT a cada peer y etiqueta el modo (direct/relay) de Tailscale. Ante
    modo desconocido cae a RELAY (conservador: el planner penaliza relay, ADR-009)."""
    modes = peer_modes if peer_modes is not None else {}
    out: list[PeerLatency] = []
    for peer in peer_directory:
        rtt = _measure_one(peer.tailscale_ip, port)
        if rtt is None:
            log.debug("peer %d (%s) inalcanzable; se omite", peer.node_id, peer.tailscale_ip)
            continue  # peer inalcanzable: se omite (no se inventa latencia)
        mode = modes.get(peer.tailscale_ip, ConnectionMode.RELAY)
        out.append(PeerLatency(node_id=peer.node_id, rtt_ms=rtt, connection=mode))
    if peer_directory and not out:
        log.warning("ningún peer alcanzable de %d; heartbeat sin latencias", len(peer_directory))
    return tuple(out)


def _poll_once(client: httpx.Client, config: AgentConfig, state: AgentState) -> bool:
    """Un ciclo de long-poll: pide un job, lo ejecuta y reporta. Devuelve True si
    procesó un job. Nunca propaga: el agente jamás muere por un job que falla."""
    headers = {"Authorization": f"Bearer {config.api_key}"}
    # content= no añade Content-Type por sí solo; sin esto el POST del resultado
    # devolvería 422 igual que register/heartbeat (mismo bug, mismo origen).
    post_headers = {**headers, "Content-Type": "application/json"}
    try:
        resp = client.get("/api/v1/jobs/next", params={"wait": 20}, headers=headers)
        if resp.status_code == httpx.codes.NO_CONTENT:
            return False
        resp.raise_for_status()
        job = JobPayload.model_validate_json(resp.content)
    except httpx.HTTPError as exc:
        log.warning("long-poll de jobs falló: %s", exc)
        return False

    start = time.monotonic()
    try:
        output = state.run_inference(job.prompt)
    except Exception:
        log.exception("inferencia del job %d falló; el watchdog lo re-encolará", job.job_id)
        return False
    latency_s = time.monotonic() - start

    result = JobResultRequest(
        text=output.text,
        eval_count=output.eval_count,
        eval_duration_s=output.eval_duration_s,
        latency_s=latency_s,
    )
    # Reintentar el reporte: si nunca llega, el coordinador deja el job RUNNING y el
    # watchdog lo re-encola (el guard de idempotencia evita doble pago). Logueamos
    # ERROR si se agotan los reintentos para que el fallo sea visible.
    for attempt in range(3):
        try:
            r = client.post(
                f"/api/v1/jobs/{job.job_id}/result",
                content=result.model_dump_json(),
                headers=post_headers,
            )
            r.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.warning("reporte del job %d falló (intento %d): %s", job.job_id, attempt + 1, exc)
    log.error("no se pudo reportar el job %d tras 3 intentos; quedará al watchdog", job.job_id)
    return True


def run_inference_loop(
    config: AgentConfig,
    state: AgentState,
    *,
    is_running: Callable[[], bool],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Bucle del head: mientras sea head con proceso vivo, hace long-poll de jobs.
    `is_running` permite pararlo (y acotar la ejecución en tests)."""
    with httpx.Client(base_url=config.coordinator_url, timeout=30.0) as client:
        while is_running():
            try:
                if state.role is NodeRole.HEAD and state.llama_proc_alive:
                    if not _poll_once(client, config, state):
                        sleep(1.0)  # sin job: respira antes de re-pollear
                else:
                    sleep(1.0)  # aún no es head: espera
            except Exception:
                # El loop de jobs NUNCA debe morir por un error inesperado.
                log.exception("error inesperado en el loop de inferencia; se reintenta")
                sleep(1.0)
