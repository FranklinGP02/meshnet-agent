"""Constantes de dominio compartidas (hechos físicos, no tarifas económicas).

Las tarifas de créditos (BASE_RATE, WORK_RATE, BONUS) son configuración del
coordinador y viven en su Settings (ADR-005), no aquí.
"""

from enum import StrEnum
from typing import NamedTuple


class NodeStatus(StrEnum):
    REGISTERED = "registered"
    ONLINE = "online"
    OFFLINE = "offline"


class NodeRole(StrEnum):
    IDLE = "idle"
    WORKER = "worker"
    HEAD = "head"


class ClusterStatus(StrEnum):
    FORMING = "forming"
    LOADING = "loading"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DISSOLVED = "dissolved"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class CreditReason(StrEnum):
    AVAILABILITY = "availability"
    WORK = "work"
    BONUS = "bonus"


class ConnectionMode(StrEnum):
    DIRECT = "direct"
    RELAY = "relay"


class GgufModel(NamedTuple):
    """Metadatos de un modelo GGUF soportado por la red."""

    name: str
    size_gb: float
    n_layers: int
    complexity: float  # peso MODEL_COMPLEXITY para el crédito de trabajo (ADR-005)
    repo: str  # repo de Hugging Face
    filename: str  # nombre del .gguf dentro del repo
    sha256: str | None = None  # verificar al descargar; None = sin verificación estricta

    @property
    def download_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


# Versión pinned de los binarios de llama.cpp (ADR-007). Tag real de
# github.com/ggml-org/llama.cpp/releases; el instalador descarga este tag.
LLAMACPP_VERSION = "b9631"


class LlamaCppAsset(NamedTuple):
    """Asset de binarios de llama.cpp por plataforma. sha256 None = sin verificar
    (se rellena tras la primera descarga; ver scripts/install_agent.ps1)."""

    asset: str  # nombre del .zip/.tar.gz en la release
    sha256: str | None = None

    def url(self, version: str) -> str:
        return f"https://github.com/ggml-org/llama.cpp/releases/download/{version}/{self.asset}"


# Clave: "{system}-{variant}" en minúsculas. variant=cpu|cuda según haya GPU NVIDIA.
LLAMACPP_ASSETS: dict[str, LlamaCppAsset] = {
    "windows-cpu": LlamaCppAsset("llama-b9631-bin-win-cpu-x64.zip"),
    "windows-cuda": LlamaCppAsset("llama-b9631-bin-win-cuda-12.4-x64.zip"),
    "linux-cpu": LlamaCppAsset("llama-b9631-bin-ubuntu-x64.tar.gz"),
}

RPC_PORT = 50052
HEARTBEAT_INTERVAL_S = 20
JOB_POLL_WAIT_S = 20
OFFLINE_THRESHOLD_S = 60
MODEL_LOAD_TIMEOUT_S = 600
HEARTBEAT_RETENTION_DAYS = 14

# Telemetría de logs remota (read-only): el agente manda sus logs piggyback en el
# heartbeat. Límites para que un agente buggy/malicioso no infle la BD ni el payload.
MAX_LOG_LINES_PER_HEARTBEAT = 50  # tope de líneas por heartbeat (cap del schema)
MAX_LOG_MESSAGE_CHARS = 2000  # truncado por línea (mensaje + traceback)
MAX_LOG_AGE_S = 300  # clamp de antigüedad: descarta timestamps absurdos del agente
NODE_LOG_RETENTION_DAYS = 7  # logs más charlatanes que heartbeats → retención más corta
MAX_CLUSTER_NODES = 4  # límite para la fuerza bruta de permutaciones (ADR-009)
PEER_LATENCY_MAX_MS = 150  # umbral por defecto; override via MESHNET_PEER_LATENCY_MAX_MS
MAX_PROMPT_BYTES = 8 * 1024
MAX_RESULT_BYTES = 64 * 1024  # tope del texto de respuesta del head
MAX_COMPLETION_TOKENS = 16384  # sanity-check anti-inflado de eval_count (PLAN.md)
MEMORY_HEADROOM_FACTOR = 1.2  # la suma de memoria libre debe superar tamaño_modelo x 1.2

# Modelos GGUF soportados. size_gb/n_layers verificados contra los model cards
# (Q4_K_M); sha256 None → se verifica al descargar tras confirmarlo la 1ª vez.
MODELS: dict[str, GgufModel] = {
    "qwen2.5-3b-q4": GgufModel(
        "qwen2.5-3b-q4",
        2.0,
        36,
        1.0,
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    ),
    "llama3.1-8b-q4": GgufModel(
        "llama3.1-8b-q4",
        4.9,
        32,
        2.5,
        repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "qwen2.5-32b-q4": GgufModel(
        "qwen2.5-32b-q4",
        20.0,
        64,
        6.0,
        repo="bartowski/Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    ),
    "llama3.1-70b-q4": GgufModel(
        "llama3.1-70b-q4",
        42.5,
        80,
        12.0,
        repo="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        filename="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    ),
}

BENCHMARK_MODEL = "qwen2.5-3b-q4"
BENCHMARK_SEED = 42
BENCHMARK_N_PREDICT = 256
BENCHMARK_RUNS = 3
# H5: benchmark real con llama-bench sobre el 3B (sustituye el proxy sintético H2).
BENCHMARK_VERSION = "h5-llamacpp-3b-v1"
