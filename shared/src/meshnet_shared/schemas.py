"""Schemas Pydantic del protocolo agente↔coordinador (plano de control).

Congelan el formato de wire: cambiarlos exige revisar las dos puntas.
Los comandos de clúster viajan en la respuesta del heartbeat (ADR-003).
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from meshnet_shared.constants import (
    MAX_COMPLETION_TOKENS,
    MAX_LOG_LINES_PER_HEARTBEAT,
    MAX_LOG_MESSAGE_CHARS,
    MAX_PROMPT_BYTES,
    MAX_RESULT_BYTES,
    ConnectionMode,
    CreditReason,
    NodeRole,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


# ── Registro ────────────────────────────────────────────────────────────────


class HardwareInfo(_FrozenModel):
    os_name: str
    cpu: str
    ram_gb: float = Field(gt=0)
    gpu: str | None = None
    vram_gb: float | None = Field(default=None, ge=0)


class BenchmarkResult(_FrozenModel):
    tokens_per_second: float = Field(gt=0)
    prompt_hash: str = Field(min_length=1, max_length=64)
    model_name: str


class RegisterRequest(_FrozenModel):
    registration_token: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=64)
    hardware: HardwareInfo
    tailscale_ip: str
    benchmark: BenchmarkResult


class RegisterResponse(_FrozenModel):
    """Única respuesta que transporta la api_key en claro: se entrega una sola vez."""

    node_id: int
    api_key: str
    power_factor: float


class RebenchmarkResponse(_FrozenModel):
    node_id: int
    power_factor: float


# ── Heartbeat y comandos de clúster ─────────────────────────────────────────


class PeerLatency(_FrozenModel):
    node_id: int
    rtt_ms: float = Field(ge=0)
    connection: ConnectionMode


class LogLine(_FrozenModel):
    """Una línea de log del agente para telemetría remota (read-only).

    El agente NO manda wall-clock (invariante: los timestamps los pone el
    coordinador); en su lugar manda `age_s`, los segundos transcurridos desde
    que se emitió la línea hasta el envío del heartbeat (medidos con monotonic).
    El coordinador calcula created_at = now - age_s."""

    level: str = Field(min_length=1, max_length=16)  # INFO/WARNING/ERROR/…
    logger: str = Field(min_length=1, max_length=64)  # p.ej. "meshnet.heartbeat"
    message: str = Field(min_length=1, max_length=MAX_LOG_MESSAGE_CHARS)
    age_s: float = Field(ge=0)


class HeartbeatRequest(_FrozenModel):
    cpu_pct: float = Field(ge=0, le=100)
    ram_pct: float = Field(ge=0, le=100)
    gpu_pct: float | None = Field(default=None, ge=0, le=100)
    ram_free_gb: float = Field(ge=0)
    vram_free_gb: float | None = Field(default=None, ge=0)
    role: NodeRole
    cluster_id: int | None = None
    peer_latencies: tuple[PeerLatency, ...] = ()
    llama_proc_alive: bool = False
    # IP de Tailscale detectada en este heartbeat. La del registro es una foto
    # fija que puede quedar obsoleta (cambio de cuenta/red de Tailscale, etc.);
    # sin esto el coordinador sigue dando a los peers una IP muerta para
    # siempre, bloqueando la formación de cualquier clúster (visto en producción).
    tailscale_ip: str | None = None
    # Telemetría de logs: cap a nivel schema para que el coordinador no procese
    # un payload gigante de un agente comprometido.
    recent_logs: tuple[LogLine, ...] = Field(default=(), max_length=MAX_LOG_LINES_PER_HEARTBEAT)


class StartRpcServerCommand(_FrozenModel):
    type: Literal["start_rpc_server"] = "start_rpc_server"
    cluster_id: int  # el agente lo reporta en heartbeat para que el coordinador lo case
    port: int


class StartHeadCommand(_FrozenModel):
    type: Literal["start_head"] = "start_head"
    cluster_id: int
    model: str
    rpc_peers: tuple[str, ...]  # "tailscale_ip:port" de cada worker, en orden de cadena
    layer_split: tuple[int, ...]  # capas por miembro: [head, worker1, worker2, ...]

    @model_validator(mode="after")
    def _head_plus_peers_match_layer_split(self) -> "StartHeadCommand":
        expected = len(self.rpc_peers) + 1  # el head también hospeda capas
        if len(self.layer_split) != expected:
            raise ValueError(
                f"layer_split tiene {len(self.layer_split)} entradas; "
                f"se esperaban {expected} (head + {len(self.rpc_peers)} workers)"
            )
        return self


class StopCommand(_FrozenModel):
    type: Literal["stop"] = "stop"


class RebenchmarkCommand(_FrozenModel):
    type: Literal["rebenchmark"] = "rebenchmark"


ClusterCommand = Annotated[
    StartRpcServerCommand | StartHeadCommand | StopCommand | RebenchmarkCommand,
    Field(discriminator="type"),
]


class PeerInfo(_FrozenModel):
    """Peer del tailnet que el agente debe medir (RTT) — lo entrega el
    coordinador para que el planner tenga latencias antes de formar clústeres."""

    node_id: int
    tailscale_ip: str


class HeartbeatResponse(_FrozenModel):
    server_time: AwareDatetime  # siempre UTC del coordinador; rechaza naive
    commands: tuple[ClusterCommand, ...] = ()
    peer_directory: tuple[PeerInfo, ...] = ()  # peers a los que medir latencia


# ── Jobs de inferencia ──────────────────────────────────────────────────────


class JobPayload(_FrozenModel):
    """Job que el head recoge por long-poll (GET /jobs/next)."""

    job_id: int
    model: str
    prompt: str


class JobResultRequest(_FrozenModel):
    text: str = Field(max_length=MAX_RESULT_BYTES)
    eval_count: int = Field(ge=0, le=MAX_COMPLETION_TOKENS)  # cap anti-inflado (PLAN.md)
    eval_duration_s: float = Field(ge=0)
    latency_s: float = Field(ge=0)


class CreditSplitEntry(_FrozenModel):
    node_id: int
    amount: float
    reason: CreditReason
    is_head: bool = False


class JobResultResponse(_FrozenModel):
    credits_awarded_split: tuple[CreditSplitEntry, ...]


# ── Admin ───────────────────────────────────────────────────────────────────


class CreateClusterRequest(_FrozenModel):
    model: str
    node_ids: tuple[int, ...] | None = None  # None → selección automática del planner


class CreateInferenceRequest(_FrozenModel):
    model: str
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_BYTES)
