# MeshNet Agent

Cliente que conviertes tu PC en un nodo de **MeshNet AI**: una red cooperativa
de cómputo donde varios PCs domésticos suman su potencia por internet para
correr modelos de IA que ninguno podría correr solo, acumulando créditos MESH
proporcionales a tu aporte.

Este repositorio contiene **solo el agente** (cliente). El coordinador
(backend) es privado.

## Instalación rápida (recomendado)

Pide a quien te invitó el enlace a la página `/join` de su coordinador.
Ahí generas un instalador PowerShell ya configurado con tu token de red.

## Instalación manual

```powershell
uv tool install "git+https://github.com/<ORG>/meshnet-agent#subdirectory=agent"
meshnet-agent register --coordinator-url https://TU-COORDINADOR --name mi-pc
meshnet-agent run
```

Requiere [uv](https://docs.astral.sh/uv/) y [Tailscale](https://tailscale.com/download)
instalados y la IP de Tailscale activa.

## Comandos

| Comando | Qué hace |
|---|---|
| `meshnet-agent prefetch --model <id>` | Descarga binarios de llama.cpp y el modelo GGUF |
| `meshnet-agent register` | Registra este PC en la red (pide el token de registro) |
| `meshnet-agent run` | Arranca el agente: heartbeats + gestión de roles del clúster |
| `meshnet-agent benchmark` | Recalibra tu `power_factor` |

## Privacidad

Este agente **nunca** ejecuta comandos arbitrarios del coordinador en tu PC.
Solo gestiona procesos de llama.cpp (inferencia) y reporta métricas/logs de
forma transparente — puedes leer todo el código en `agent/src/meshnet_agent/`.
