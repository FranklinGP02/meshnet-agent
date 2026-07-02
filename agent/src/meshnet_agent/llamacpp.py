"""Gestión de procesos llama.cpp (ADR-007): rpc-server (worker) y llama-server (head).

Patrón Protocol + impl real + fake (como benchmark.py): `ProcessManager` aísla el
subprocess para que roles.py y los tests no dependan de binarios reales. En H5 los
binarios y el GGUF se descargan/cachean (downloads.py); el FakeProcessManager se usa
en tests. Los flags RPC se construyen con build_*_argv (puras, ADR-016).
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from meshnet_shared.constants import LLAMACPP_VERSION, MODELS

log = logging.getLogger("meshnet.llamacpp")


@dataclass(frozen=True)
class ProcSpec:
    """Identidad de un proceso para idempotencia: si el comando entrante coincide
    con el spec del proceso vivo, no se relanza (ADR: comandos declarativos)."""

    kind: str  # "rpc_server" | "head"
    port: int = 0
    model: str = ""
    rpc_peers: tuple[str, ...] = ()
    layer_split: tuple[int, ...] = ()


@dataclass(frozen=True)
class InferenceOutput:
    """Resultado de una inferencia local del head."""

    text: str
    eval_count: int
    eval_duration_s: float


class ProcessManager(Protocol):
    def start_rpc_server(self, host: str, port: int) -> None: ...
    def start_head(
        self, model: str, rpc_peers: tuple[str, ...], layer_split: tuple[int, ...]
    ) -> None: ...
    def stop(self) -> None: ...
    def is_alive(self) -> bool: ...
    def current_spec(self) -> ProcSpec | None: ...
    def run_inference(self, prompt: str) -> InferenceOutput: ...


# ── Construcción de argv (PURO, ADR-016) ──────────────────────────────────
# Aísla los flags RPC de llama.cpp en funciones puras con golden tests: cuando
# se valide contra el binario pinned, el ajuste es en UN sitio. VALIDAR contra
# b9631: orden de --tensor-split vs --rpc, si -ngl es necesario, y el flag de host.


def build_rpc_server_argv(binary: str, host: str, port: int) -> list[str]:
    """rpc-server enlazado SOLO a la IP del tailnet (nunca 0.0.0.0, ADR-008)."""
    return [binary, "-H", host, "-p", str(port)]


def build_head_argv(
    binary: str,
    model_path: str,
    *,
    rpc_peers: tuple[str, ...],
    layer_split: tuple[int, ...],
    n_layers: int,
    port: int,
) -> list[str]:
    """argv de llama-server. Sin peers → single-node (sin --rpc/--tensor-split).
    Con peers → --rpc en orden de cadena y --tensor-split con los pesos por device
    (head primero), -ngl=n_layers para forzar el offload distribuido."""
    argv = [binary, "-m", model_path, "--port", str(port), "-ngl", str(n_layers)]
    if rpc_peers:
        expected = len(rpc_peers) + 1  # head + workers
        if len(layer_split) != expected:
            raise ValueError(
                f"layer_split tiene {len(layer_split)} entradas; se esperaban {expected} "
                f"(head + {len(rpc_peers)} devices RPC)"
            )
        argv += ["--rpc", ",".join(rpc_peers)]
        argv += ["--tensor-split", ",".join(str(n) for n in layer_split)]
    return argv


class SubprocessLlamaManager:
    """Implementación real: lanza y supervisa procesos llama.cpp con subprocess.

    NUNCA propaga un fallo de arranque hacia el loop del agente: lo loguea y deja
    is_alive()=False para que el coordinador degrade el clúster por la vía normal.
    """

    def __init__(self, head_port: int = 8080, binary_dir: str | None = None) -> None:
        self._head_port = head_port
        # None → resolver perezosamente vía downloads.ensure_binaries en el 1er uso.
        self._binary_dir = binary_dir
        self._proc: subprocess.Popen[bytes] | None = None
        self._spec: ProcSpec | None = None

    def _bin(self, name: str) -> str:
        from meshnet_agent.downloads import ensure_binaries
        from meshnet_agent.platform_assets import current_llamacpp_asset

        if self._binary_dir is None:
            self._binary_dir = str(ensure_binaries(current_llamacpp_asset()))
        return str(Path(self._binary_dir) / name)

    def _spawn(self, args: list[str]) -> None:
        log.info("lanzando llama.cpp (%s): %s", LLAMACPP_VERSION, " ".join(args))
        self._proc = subprocess.Popen(args)

    def start_rpc_server(self, host: str, port: int) -> None:
        spec = ProcSpec(kind="rpc_server", port=port)
        if self._spec == spec and self.is_alive():
            return  # idempotente: ya corriendo con esta config
        self.stop()
        try:
            self._spawn(build_rpc_server_argv(self._bin("rpc-server"), host, port))
            self._spec = spec
        except (OSError, ValueError, RuntimeError):
            # RuntimeError incluye DownloadError (descarga lazy de binarios). NUNCA
            # propagar al loop del agente: degradar a no-vivo y que el coordinador lo note.
            log.exception("no se pudo lanzar rpc-server")
            self._proc, self._spec = None, None

    def start_head(
        self, model: str, rpc_peers: tuple[str, ...], layer_split: tuple[int, ...]
    ) -> None:
        spec = ProcSpec(kind="head", model=model, rpc_peers=rpc_peers, layer_split=layer_split)
        # is_alive() devuelve True SOLO cuando /health=200 (modelo cargado). Durante la
        # carga (503) el proceso sigue corriendo pero aún no sirve peticiones — no
        # reiniciarlo o entraría en un loop matando el modelo cada 20s.
        proc_running = self._proc is not None and self._proc.poll() is None
        if self._spec == spec and proc_running:
            return
        self.stop()
        try:
            from meshnet_agent.downloads import ensure_model

            model_path = str(ensure_model(model))
            argv = build_head_argv(
                self._bin("llama-server"),
                model_path,
                rpc_peers=rpc_peers,
                layer_split=layer_split,
                n_layers=MODELS[model].n_layers if model in MODELS else sum(layer_split),
                port=self._head_port,
            )
            self._spawn(argv)
            self._spec = spec
        except (OSError, ValueError, RuntimeError):  # incl. DownloadError de la descarga lazy
            log.exception("no se pudo lanzar llama-server")
            self._proc, self._spec = None, None

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc, self._spec = None, None

    def is_alive(self) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            return False
        if self._spec is not None and self._spec.kind == "head":
            # El head está "vivo" cuando llama-server responde /health (modelo cargado).
            try:
                resp = httpx.get(f"http://127.0.0.1:{self._head_port}/health", timeout=2.0)
                return resp.status_code == 200
            except Exception:  # cualquier fallo del probe = no vivo (nunca propaga)
                return False
        return True

    def current_spec(self) -> ProcSpec | None:
        return self._spec

    def run_inference(self, prompt: str) -> InferenceOutput:
        """Ejecuta la inferencia contra el llama-server local (que computa a través
        de los workers por RPC). Solo válido si este nodo es head y está vivo."""
        if self._spec is None or self._spec.kind != "head":
            raise RuntimeError("run_inference solo válido en el head con modelo cargado")
        resp = httpx.post(
            f"http://127.0.0.1:{self._head_port}/completion",
            json={"prompt": prompt, "n_predict": 256},
            timeout=600.0,
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens_predicted")
        predicted_ms = data.get("timings", {}).get("predicted_ms")
        if tokens is None or predicted_ms is None:
            # Sin estos campos el crédito de trabajo saldría 0 silenciosamente.
            log.warning(
                "respuesta de llama-server sin tokens_predicted/timings; créditos serán 0. keys=%s",
                list(data.keys()),
            )
        return InferenceOutput(
            text=data.get("content", ""),
            eval_count=int(tokens or 0),
            eval_duration_s=float(predicted_ms or 0.0) / 1000.0,
        )


class FakeProcessManager:
    """Fake en memoria para tests: sin subprocess ni red."""

    def __init__(self) -> None:
        self._spec: ProcSpec | None = None
        self._alive = False

    def start_rpc_server(self, host: str, port: int) -> None:
        self._spec = ProcSpec(kind="rpc_server", port=port)
        self._alive = True

    def start_head(
        self, model: str, rpc_peers: tuple[str, ...], layer_split: tuple[int, ...]
    ) -> None:
        self._spec = ProcSpec(
            kind="head", model=model, rpc_peers=rpc_peers, layer_split=layer_split
        )
        self._alive = True

    def stop(self) -> None:
        self._spec = None
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def current_spec(self) -> ProcSpec | None:
        return self._spec

    def run_inference(self, prompt: str) -> InferenceOutput:
        # Respuesta determinista para tests (la inferencia real es H5).
        return InferenceOutput(
            text=f"[fake completion for: {prompt[:32]}]", eval_count=64, eval_duration_s=2.0
        )
