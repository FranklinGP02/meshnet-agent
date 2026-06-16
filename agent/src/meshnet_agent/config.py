"""Config persistente del agente: credenciales tras `register`.

Guardada en TOML bajo user_config_dir (Windows: %APPDATA%\\meshnet\\config.toml).
La api_key es el único secreto y solo se obtiene una vez al registrarse.
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w
from platformdirs import user_config_dir

log = logging.getLogger("meshnet.config")

_APP = "meshnet"


def config_path() -> Path:
    return Path(user_config_dir(_APP)) / "config.toml"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    coordinator_url: str
    node_id: int
    api_key: str


def save_config(config: AgentConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # tomli_w escapa los valores: sin inyección TOML aunque la URL traiga comillas.
    payload = {
        "coordinator_url": config.coordinator_url,
        "node_id": config.node_id,
        "api_key": config.api_key,
    }
    target.write_bytes(tomli_w.dumps(payload).encode("utf-8"))
    try:
        target.chmod(0o600)  # best-effort; en Windows la ACL de %APPDATA% ya es per-usuario
    except OSError:
        pass
    return target


def load_config(path: Path | None = None) -> AgentConfig | None:
    target = path or config_path()
    if not target.exists():
        return None
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
        return AgentConfig(
            coordinator_url=data["coordinator_url"],
            node_id=int(data["node_id"]),
            api_key=data["api_key"],
        )
    except (tomllib.TOMLDecodeError, KeyError, ValueError) as exc:
        log.error("config corrupta en %s: %s — ejecuta 'register' de nuevo", target, exc)
        return None
