"""CLI del agente MeshNet (typer): register | run | benchmark.

register y run implementados en H1; benchmark real llega en H2.
"""

import logging
import threading

import httpx
import typer

from meshnet_agent.benchmark import (
    BenchmarkRunner,
    LlamaCppBenchmarkRunner,
    StubBenchmarkRunner,
    run_benchmark,
)
from meshnet_agent.config import AgentConfig, load_config, save_config
from meshnet_agent.hardware import detect_hardware
from meshnet_agent.heartbeat import AuthRevokedError, run_loop
from meshnet_agent.llamacpp import SubprocessLlamaManager
from meshnet_agent.logbuffer import install_ring_handler
from meshnet_agent.roles import AgentState
from meshnet_agent.worker import run_inference_loop
from meshnet_shared.constants import BENCHMARK_MODEL
from meshnet_shared.schemas import RegisterRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = typer.Typer(help="Agente MeshNet — comparte tu cómputo y acumula créditos MESH.")


def _describe_http_error(exc: httpx.HTTPError) -> str:
    """Incluye el cuerpo de la respuesta del coordinador en el mensaje: sin esto,
    un 422 solo dice 'Unprocessable Entity' sin decir qué campo falló la
    validación, lo que vuelve indepurable cualquier error de schema."""
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:1000]
        return f"{exc}\nRespuesta del servidor: {body}"
    return str(exc)


def _benchmark_runner(stub: bool) -> BenchmarkRunner:
    """Runner real (llama-bench con el 3B) o stub determinista para dev/CI."""
    if stub:
        return StubBenchmarkRunner()
    from pathlib import Path

    from meshnet_agent.downloads import ensure_binaries, ensure_model
    from meshnet_agent.platform_assets import current_llamacpp_asset

    binary_dir = ensure_binaries(current_llamacpp_asset())
    model_path = ensure_model(BENCHMARK_MODEL)
    return LlamaCppBenchmarkRunner(str(Path(binary_dir) / "llama-bench"), str(model_path))


@app.command()
def prefetch(
    model: str = typer.Option(BENCHMARK_MODEL, help="Modelo GGUF a descargar"),
) -> None:
    """Descarga (y cachea) los binarios de llama.cpp y el GGUF del modelo. Idempotente."""
    from meshnet_agent.downloads import ensure_binaries, ensure_model
    from meshnet_agent.platform_assets import current_llamacpp_asset

    try:
        typer.secho("Descargando binarios de llama.cpp…", fg=typer.colors.CYAN)
        ensure_binaries(current_llamacpp_asset())
        typer.secho(f"Descargando modelo {model} (puede tardar)…", fg=typer.colors.CYAN)
        ensure_model(model)
    except Exception as exc:
        typer.secho(f"prefetch falló: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho("Binarios y modelo listos en caché.", fg=typer.colors.GREEN)


@app.command()
def register(
    coordinator_url: str = typer.Option(..., help="URL del coordinador (https://...)"),
    token: str = typer.Option(
        ...,
        prompt=True,
        hide_input=True,
        envvar="MESHNET_REGISTRATION_TOKEN",
        help="Registration token (se pide por consola si no está en la env var)",
    ),
    name: str = typer.Option(..., help="Nombre de este nodo"),
    tailscale_ip: str | None = typer.Option(
        None, help="IP del tailnet (autodetectada de Tailscale si se omite)"
    ),
    stub_benchmark: bool = typer.Option(
        False, "--stub-benchmark", help="Usar benchmark stub (dev/CI, sin llama.cpp)"
    ),
) -> None:
    """Registra este PC en la red MeshNet y guarda la API key localmente."""
    if tailscale_ip is None:
        from meshnet_agent.tailscale import detect_self_ip

        tailscale_ip = detect_self_ip()
        if tailscale_ip is None:
            typer.secho(
                "No se detectó IP de Tailscale. Instala/loguea Tailscale o pasa --tailscale-ip.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    hardware = detect_hardware()
    benchmark = run_benchmark(_benchmark_runner(stub_benchmark))
    request = RegisterRequest(
        registration_token=token,
        name=name,
        hardware=hardware,
        tailscale_ip=tailscale_ip,
        benchmark=benchmark,
    )
    try:
        response = httpx.post(
            f"{coordinator_url}/api/v1/nodes/register",
            content=request.model_dump_json(),
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        typer.secho(f"register falló: {_describe_http_error(exc)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    data = response.json()
    try:
        path = save_config(
            AgentConfig(
                coordinator_url=coordinator_url, node_id=data["node_id"], api_key=data["api_key"]
            )
        )
    except OSError as exc:
        # El nodo YA está registrado en el servidor y la key se entrega una sola
        # vez: si no se puede guardar, imprímela para recuperación manual.
        typer.secho(
            f"Registrado (nodo #{data['node_id']}) pero NO se pudo guardar la config: {exc}\n"
            f"Guarda esta API key manualmente: {data['api_key']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"Registrado como nodo #{data['node_id']} (power_factor={data['power_factor']}). "
        f"Config en {path}",
        fg=typer.colors.GREEN,
    )


@app.command()
def run() -> None:
    """Arranca el agente: loop de heartbeats con retry/backoff."""
    config = load_config()
    if config is None:
        typer.secho(
            "No hay config; ejecuta `meshnet-agent register` primero.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.secho(f"Agente nodo #{config.node_id} → {config.coordinator_url}", fg=typer.colors.GREEN)
    # Telemetría de logs remota (read-only): retiene las últimas líneas de los
    # loggers meshnet.* y las manda piggyback en cada heartbeat para que el admin
    # pueda depurar nodos de colaboradores no técnicos sin pedirles su terminal.
    log_handler = install_ring_handler()
    state = AgentState(SubprocessLlamaManager())
    # rpc-server debe enlazar a la IP del tailnet (ADR-008), no a 127.0.0.1.
    from meshnet_agent.tailscale import detect_self_ip

    self_ip = detect_self_ip()
    if self_ip is not None:
        state.set_host(self_ip)
    # El long-poll de jobs corre en un hilo aparte para no retrasar el heartbeat
    # (ambos pueden bloquear ~20s). Se activa solo cuando el nodo es head.
    stop = threading.Event()
    job_thread = threading.Thread(
        target=run_inference_loop,
        args=(config, state),
        kwargs={"is_running": lambda: not stop.is_set()},
        daemon=True,
    )
    job_thread.start()
    try:
        run_loop(config, state=state, log_handler=log_handler)
    except AuthRevokedError:
        typer.secho("API key revocada; vuelve a registrar el nodo.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.secho("Detenido por el usuario.", fg=typer.colors.YELLOW)
    finally:
        stop.set()


@app.command()
def benchmark(
    stub_benchmark: bool = typer.Option(
        False, "--stub-benchmark", help="Usar benchmark stub (dev/CI, sin llama.cpp)"
    ),
) -> None:
    """Ejecuta el benchmark local y recalibra el power_factor en el coordinador."""
    result = run_benchmark(_benchmark_runner(stub_benchmark))
    typer.secho(f"Benchmark: {result.tokens_per_second} tok/s", fg=typer.colors.GREEN)

    config = load_config()
    if config is None:
        typer.secho(
            "No hay config; ejecuta `meshnet-agent register` primero.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        response = httpx.post(
            f"{config.coordinator_url}/api/v1/nodes/rebenchmark",
            content=result.model_dump_json(),
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        typer.secho(
            f"rebenchmark falló: {_describe_http_error(exc)}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"power_factor actualizado a {response.json()['power_factor']}", fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    app()
