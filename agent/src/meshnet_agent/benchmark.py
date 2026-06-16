"""Benchmark de potencia del nodo.

H2 usó un stub determinista (proxy CPU). H5 añade LlamaCppBenchmarkRunner que
ejecuta `llama-bench` con el 3B real (mismo interfaz BenchmarkRunner) y mide
tokens/s de verdad → power_factor real (ADR-006).
"""

import json
import logging
import os
import statistics
import subprocess
from typing import Protocol

from meshnet_shared.constants import (
    BENCHMARK_MODEL,
    BENCHMARK_N_PREDICT,
    BENCHMARK_RUNS,
    BENCHMARK_SEED,
    BENCHMARK_VERSION,
)
from meshnet_shared.schemas import BenchmarkResult

log = logging.getLogger("meshnet.benchmark")


class BenchmarkRunner(Protocol):
    def measure_tps(self) -> float:
        """Devuelve tokens/segundo de UNA medición."""
        ...


class StubBenchmarkRunner:
    """Proxy DETERMINISTA de H2: deriva un tps reproducible de las specs del
    hardware (núcleos), sin medir tiempo de pared. No mide velocidad real —
    eso es el LlamaCppBenchmarkRunner de H3 — pero distingue máquinas y da un
    power_factor verosímil con un valor estable entre ejecuciones."""

    def measure_tps(self) -> float:
        cores = os.cpu_count() or 1
        # ~5 tps/núcleo (un 3B en CPU ronda ese orden); estable y comparable.
        return float(cores * 5)


def _parse_llama_bench_tps(stdout: str) -> float:
    """tokens/s de generación de la salida JSON de `llama-bench -o json` (avg_ts).
    Convierte fallos de formato en ValueError con contexto (el registro debe fallar
    visiblemente, no con un KeyError opaco)."""
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"llama-bench no devolvió JSON válido. stdout: {stdout[:500]!r}") from exc
    if not rows:
        raise ValueError("llama-bench no devolvió filas")
    # Preferir la fila de generación (n_gen>0); si no, la primera con avg_ts.
    gen = next((r for r in rows if r.get("n_gen")), rows[0])
    if "avg_ts" not in gen:
        raise ValueError(f"campo 'avg_ts' ausente en la fila de llama-bench: {gen}")
    return float(gen["avg_ts"])


class LlamaCppBenchmarkRunner:
    """Benchmark REAL (H5): corre `llama-bench` con el 3B de referencia y lee tok/s.
    A diferencia del proceso de clúster, un fallo aquí SÍ se propaga (el registro
    debe fallar visiblemente si no se puede medir)."""

    def __init__(self, binary: str, model_path: str) -> None:
        self._binary = binary
        self._model_path = model_path

    def measure_tps(self) -> float:
        argv = [
            self._binary,
            "-m",
            self._model_path,
            "-p",
            "0",  # sin prompt eval; solo generación
            "-n",
            str(BENCHMARK_N_PREDICT),
            "--seed",
            str(BENCHMARK_SEED),
            "-r",
            "1",  # una repetición interna; la mediana la hace run_benchmark
            "-o",
            "json",
        ]
        result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=600)
        return _parse_llama_bench_tps(result.stdout)


def run_benchmark(runner: BenchmarkRunner, runs: int = BENCHMARK_RUNS) -> BenchmarkResult:
    """Mediana de `runs` mediciones (robusta a un run ruidoso)."""
    samples = [runner.measure_tps() for _ in range(runs)]
    return BenchmarkResult(
        tokens_per_second=statistics.median(samples),
        prompt_hash=BENCHMARK_VERSION,
        model_name=BENCHMARK_MODEL,
    )
