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


def _cpu_name() -> str:
    """Nombre comercial de la CPU (p.ej. "AMD Ryzen 7 5800X 8-Core Processor").

    En Windows, platform.processor() solo da la firma de familia/modelo/stepping
    (p.ej. "AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD"), que es IDÉNTICA
    entre CPUs distintas de la misma generación — en el dashboard, dos PCs con
    CPUs diferentes pero de la misma gama aparentan tener "el mismo hardware".
    Se lee el nombre real del registro; si falla, se degrada al valor genérico
    (nunca debe romper el registro del nodo por esto)."""
    if platform.system() == "Windows":
        try:
            # importlib.import_module en vez de "import winreg": typeshed solo
            # expone los miembros de winreg cuando mypy resuelve para
            # sys.platform == "win32" — en el runner ubuntu-latest de la matriz
            # CI el stub aparece vacío (attr-defined). import_module() devuelve
            # ModuleType, cuyos atributos tipan Any en typeshed: evita el error
            # SIN type:ignore (que además mypy marcaría "no usado" en Windows,
            # donde sí hay atributos — strict=True lo trataría como error).
            import importlib

            winreg = importlib.import_module("winreg")
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            name = str(name).strip()
            if name:
                return name
        except OSError as exc:
            log.debug("no se pudo leer ProcessorNameString del registro: %s", exc)
    return platform.processor() or platform.machine()


def detect_hardware() -> HardwareInfo:
    gpu, vram_gb = _gpu_static()
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu=_cpu_name(),
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
