"""Selección del asset de binarios llama.cpp según el SO y si hay GPU NVIDIA."""

import logging
import platform

from meshnet_shared.constants import LLAMACPP_ASSETS, LlamaCppAsset

log = logging.getLogger("meshnet.platform")


def _has_nvidia_gpu() -> bool:
    """True si hay GPU NVIDIA usable. Distingue "no hay GPU" (esperado) de un error
    inesperado (que se loguea), para no elegir CPU en silencio con una GPU rota."""
    try:
        import pynvml
    except ImportError:
        return False
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return False  # driver ausente o no cargado
    try:
        return int(pynvml.nvmlDeviceGetCount()) > 0
    except pynvml.NVMLError:
        return False
    except Exception:
        log.warning("error inesperado consultando GPU NVIDIA; asumiendo CPU", exc_info=True)
        return False
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def current_llamacpp_asset() -> LlamaCppAsset:
    """Asset adecuado para esta máquina. Windows/Linux x cpu/cuda; cae a CPU si no
    hay variante CUDA para el SO o no hay GPU NVIDIA."""
    system = platform.system().lower()  # "windows" | "linux" | "darwin"
    variant = "cuda" if _has_nvidia_gpu() else "cpu"
    key = f"{system}-{variant}"
    if key not in LLAMACPP_ASSETS:
        fallback = f"{system}-cpu"
        if fallback not in LLAMACPP_ASSETS:
            raise RuntimeError(f"sin binarios de llama.cpp para {system}")
        log.warning("sin asset %s; usando %s", key, fallback)
        key = fallback
    return LLAMACPP_ASSETS[key]
