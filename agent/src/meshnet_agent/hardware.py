"""Detección de hardware y lectura de métricas locales.

pynvml es OPCIONAL: si falta o no hay GPU NVIDIA, la GPU se degrada a None,
PERO un fallo al leer CPU/RAM SÍ se propaga (no es un caso degradado normal).
"""

import logging
import platform

import psutil

from meshnet_shared.schemas import HardwareInfo

log = logging.getLogger("meshnet.hardware")

_GB = 1024**3


def _gpu_static() -> tuple[str | None, float | None]:
    """(nombre, vram_total_gb) o (None, None) si no hay GPU NVIDIA detectable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name_str = name.decode() if isinstance(name, bytes) else str(name)
            return name_str, round(mem.total / _GB, 2)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # sin GPU, sin driver, o pynvml ausente → degradar
        log.debug("GPU no disponible: %s", exc)
        return None, None


def detect_hardware() -> HardwareInfo:
    gpu, vram_gb = _gpu_static()
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        ram_gb=round(psutil.virtual_memory().total / _GB, 2),
        gpu=gpu,
        vram_gb=vram_gb,
    )


def _gpu_usage() -> tuple[float | None, float | None]:
    """(uso_gpu_pct, vram_libre_gb) o (None, None)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(util.gpu), round(mem.free / _GB, 2)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # GPU opcional: degradar a None, pero dejar rastro
        log.debug("uso de GPU no disponible: %s", exc)
        return None, None


def read_metrics() -> dict[str, float | None]:
    """Snapshot de uso para el heartbeat. CPU/RAM obligatorios; GPU opcional."""
    vm = psutil.virtual_memory()
    gpu_pct, vram_free_gb = _gpu_usage()
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_pct": vm.percent,
        "ram_free_gb": round(vm.available / _GB, 2),
        "gpu_pct": gpu_pct,
        "vram_free_gb": vram_free_gb,
    }
