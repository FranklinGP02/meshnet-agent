"""Detección de Tailscale (ADR-008/009): IP propia del tailnet y modo de conexión
direct vs relay (DERP) por peer.

El parseo de `tailscale status --json` es PURO y testeable; la ejecución del binario
está aislada. Ante cualquier fallo (binario ausente, no logueado, JSON inesperado),
las funciones públicas degradan a None/{} sin propagar — el agente nunca muere.
Las claves del JSON pueden variar entre versiones de Tailscale: parseo defensivo.
"""

import json
import logging
import subprocess

from meshnet_shared.constants import ConnectionMode

log = logging.getLogger("meshnet.tailscale")

_TIMEOUT_S = 10.0


def parse_self_ip(status_json: str) -> str | None:
    """IP IPv4 del tailnet (100.x) desde la salida de `tailscale status --json`."""
    try:
        data = json.loads(status_json)
        ips = data.get("Self", {}).get("TailscaleIPs", []) or []
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    for ip in ips:
        if ":" not in ip:  # IPv4
            return str(ip)
    return None


def parse_peer_modes(status_json: str) -> dict[str, ConnectionMode]:
    """{tailscale_ip: DIRECT|RELAY} por peer. Relay no vacío o sin CurAddr → RELAY."""
    try:
        data = json.loads(status_json)
        peers = data.get("Peer", {}) or {}
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}
    out: dict[str, ConnectionMode] = {}
    for peer in peers.values():
        ips = peer.get("TailscaleIPs", []) or []
        ipv4 = next((ip for ip in ips if ":" not in ip), None)
        if ipv4 is None:
            continue
        relayed = bool(peer.get("Relay")) or not peer.get("CurAddr")
        out[ipv4] = ConnectionMode.RELAY if relayed else ConnectionMode.DIRECT
    return out


def _run_status() -> str | None:
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        log.warning("no se pudo consultar tailscale status: %s %s", exc, stderr)
        return None


def detect_self_ip() -> str | None:
    status = _run_status()
    return parse_self_ip(status) if status else None


def detect_peer_modes() -> dict[str, ConnectionMode]:
    status = _run_status()
    return parse_peer_modes(status) if status else {}
