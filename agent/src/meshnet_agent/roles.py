"""Máquina de estados del agente: idle ↔ worker ↔ head (ADR-003).

El agente es subordinado: ejecuta los comandos que llegan en la respuesta del
heartbeat reconciliando contra el proceso vivo (idempotente). La fuente de verdad
del DESEO es el coordinador; el agente publica el HECHO (role + llama_proc_alive)
en cada heartbeat. Nunca declara estado de clúster por su cuenta.
"""

import logging

from meshnet_agent.llamacpp import InferenceOutput, ProcessManager
from meshnet_shared.constants import NodeRole
from meshnet_shared.schemas import (
    ClusterCommand,
    StartHeadCommand,
    StartRpcServerCommand,
)

log = logging.getLogger("meshnet.roles")


class AgentState:
    """Estado local del agente, derivado de los procesos que gestiona."""

    def __init__(self, processes: ProcessManager) -> None:
        self._proc = processes
        self.role: NodeRole = NodeRole.IDLE
        self.cluster_id: int | None = None
        self.host: str = "127.0.0.1"  # IP del tailnet para enlazar rpc-server

    @property
    def llama_proc_alive(self) -> bool:
        return self._proc.is_alive()

    def run_inference(self, prompt: str) -> InferenceOutput:
        return self._proc.run_inference(prompt)

    def set_host(self, host: str) -> None:
        self.host = host

    def apply_commands(self, commands: tuple[ClusterCommand, ...]) -> None:
        """Aplica los comandos del heartbeat. Idempotente: el ProcessManager no
        relanza si la config coincide con el proceso vivo. Un fallo de arranque
        NO se propaga (el agente nunca muere); se refleja como proc no vivo."""
        for command in commands:
            try:
                self._apply_one(command)
            except Exception:  # resiliencia: un comando no debe tumbar el loop
                log.exception("fallo aplicando comando %s", getattr(command, "type", "?"))

    def _apply_one(self, command: ClusterCommand) -> None:
        # El rol solo se fija si el proceso arrancó de verdad: así un fallo de
        # arranque deja role=IDLE y el coordinador lo detecta de inmediato (no
        # espera el timeout de carga viendo un head "vivo" que nunca lo estuvo).
        if isinstance(command, StartRpcServerCommand):
            self._proc.start_rpc_server(self.host, command.port)
            if self._proc.is_alive():
                self.role = NodeRole.WORKER
                self.cluster_id = command.cluster_id
        elif isinstance(command, StartHeadCommand):
            self._proc.start_head(command.model, command.rpc_peers, command.layer_split)
            if self._proc.is_alive():
                self.role = NodeRole.HEAD
                self.cluster_id = command.cluster_id
        else:  # StopCommand / RebenchmarkCommand → liberar y volver a idle
            self._proc.stop()
            self.role = NodeRole.IDLE
            self.cluster_id = None
