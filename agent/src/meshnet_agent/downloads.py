"""Descarga verificada de binarios llama.cpp y modelos GGUF (H5).

Streaming con sha256 incremental, reanudación por HTTP Range, caché con marcador
`.ok` (no re-descarga si ya está íntegro). Un checksum que no cuadra borra el
artefacto y lanza DownloadError SIN reintentar (mismatch = corrupción persistente
o pin mal puesto; reintentar solo quema ancho de banda). Archivos de 2-42 GB:
la reanudación es crítica en redes domésticas.

Defensas: extracción de zip sin path traversal (zip slip), verificación de que la
extracción dejó los binarios completos antes de marcar `.ok`, caché que revalida
tamaño/checksum, y bloqueo de redirects a HTTP no seguro.
"""

import hashlib
import logging
import shutil
import tarfile
import threading
import zipfile
from pathlib import Path

import httpx
from platformdirs import user_cache_dir

from meshnet_shared.constants import LLAMACPP_VERSION, MODELS, GgufModel, LlamaCppAsset

log = logging.getLogger("meshnet.downloads")

_CHUNK = 1024 * 1024  # 1 MiB
_REQUIRED_BINARIES = ("llama-server", "rpc-server", "llama-bench")
_binary_lock = threading.Lock()


class DownloadError(RuntimeError):
    """Descarga fallida o integridad no verificada."""


def cache_dir() -> Path:
    return Path(user_cache_dir("meshnet"))


def _ok_marker(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".ok")


def _block_insecure_redirect(response: httpx.Response) -> None:
    if response.is_redirect:
        loc = response.headers.get("location", "")
        if loc.startswith("http://"):
            raise DownloadError(f"redirect a HTTP no seguro bloqueado: {loc}")


def _new_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(30.0, read=300.0),
        follow_redirects=True,
        event_hooks={"response": [_block_insecure_redirect]},
    )


def download_file(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    expected_size: int | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Descarga `url` a `dest` (streaming). Verifica tamaño y sha256 si se dan.
    Cachea con `.ok`; en cache hit revalida tamaño/checksum. Reanuda por Range."""
    marker = _ok_marker(dest)
    if marker.exists() and dest.exists() and _cache_is_valid(dest, marker, sha256):
        log.info("cache hit: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    owns_client = client is None
    client = client or _new_client()
    try:
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        with client.stream("GET", url, headers=headers) as r:
            if r.status_code not in (200, 206):
                raise DownloadError(f"HTTP {r.status_code} descargando {url}")
            # 206 con Content-Range que no empieza donde pedimos → CDN roto: reiniciar.
            if resume_from and r.status_code == 206:
                crange = r.headers.get("content-range", "")
                if f"bytes {resume_from}-" not in crange:
                    log.warning("Content-Range inesperado (%s); reinicio limpio", crange)
                    resume_from = 0
            mode = "ab" if (resume_from and r.status_code == 206) else "wb"
            if mode == "wb":
                resume_from = 0
            with part.open(mode) as f:
                for chunk in r.iter_bytes(_CHUNK):
                    f.write(chunk)

        size = part.stat().st_size
        if expected_size is not None and size != expected_size:
            part.unlink(missing_ok=True)
            raise DownloadError(f"tamaño inesperado de {dest.name}: {size} != {expected_size}")
        if sha256 is not None:
            actual = _sha256_of(part)
            if actual.lower() != sha256.lower():
                part.unlink(missing_ok=True)
                raise DownloadError(f"checksum de {dest.name} no coincide ({actual} != {sha256})")
        else:
            log.warning("descarga de %s SIN verificación de checksum (sha256 no pinado)", dest.name)

        part.replace(dest)
        marker.write_text(sha256 or "unverified", encoding="utf-8")
        log.info("descargado %s (%d bytes)", dest.name, size)
        return dest
    finally:
        if owns_client:
            client.close()


def _cache_is_valid(dest: Path, marker: Path, sha256: str | None) -> bool:
    """En cache hit, revalida: archivo no vacío y, si hay sha pinado, que el marker
    lo confirme. Si no cuadra, invalida (borra) y fuerza re-descarga."""
    if dest.stat().st_size == 0:
        log.warning("archivo en caché vacío; re-descargando %s", dest.name)
        dest.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        return False
    if sha256 is not None:
        stored = marker.read_text(encoding="utf-8").strip().lower()
        if stored not in ("unverified", "") and stored != sha256.lower():
            log.warning("checksum del caché no coincide; re-descargando %s", dest.name)
            dest.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            return False
    return True


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(model_name: str, *, client: httpx.Client | None = None) -> Path:
    """Asegura el GGUF del modelo en caché y devuelve su ruta."""
    model: GgufModel = MODELS[model_name]
    dest = cache_dir() / "models" / model.filename
    return download_file(model.download_url, dest, sha256=model.sha256, client=client)


def ensure_binaries(
    asset: LlamaCppAsset, *, version: str = LLAMACPP_VERSION, client: httpx.Client | None = None
) -> Path:
    """Asegura los binarios de llama.cpp extraídos en caché. Devuelve el directorio
    con rpc-server/llama-server/llama-bench. Thread-safe (descargas concurrentes)."""
    bin_root = cache_dir() / "bin" / version
    extracted_ok = bin_root / ".extracted.ok"
    with _binary_lock:
        # Fast path: marcador presente Y un binario centinela existe y no está vacío.
        if extracted_ok.exists() and _find_binary(bin_root, "llama-server") is not None:
            return bin_root
        if extracted_ok.exists():
            log.warning(".extracted.ok existe pero faltan binarios; re-extrayendo")
            extracted_ok.unlink(missing_ok=True)

        archive = download_file(
            asset.url(version), bin_root / asset.asset, sha256=asset.sha256, client=client
        )
        try:
            _extract(archive, bin_root)
            for name in _REQUIRED_BINARIES:
                found = _find_binary(bin_root, name)
                if found is None:
                    raise DownloadError(f"extracción incompleta: falta {name} en {bin_root}")
        except Exception:
            # No dejar un bin_root a medias que un .ok falso certifique luego.
            extracted_ok.unlink(missing_ok=True)
            raise
        archive.unlink(missing_ok=True)  # liberar espacio (zips de cientos de MB)
        extracted_ok.write_text("ok", encoding="utf-8")
        return bin_root


def _find_binary(root: Path, name: str) -> Path | None:
    """Localiza un ejecutable (con o sin .exe) no vacío bajo `root`."""
    for cand in (*root.rglob(f"{name}.exe"), *root.rglob(name)):
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            _safe_extract_zip(z, dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")  # filter=data evita path traversal
    else:
        raise DownloadError(f"formato de archivo no soportado: {archive.name}")
    shutil.rmtree(dest / "__MACOSX", ignore_errors=True)


def _safe_extract_zip(z: zipfile.ZipFile, dest: Path) -> None:
    """Extrae un zip rechazando entradas que escapen de `dest` (zip slip)."""
    dest_root = dest.resolve()
    for member in z.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest_root)):
            raise DownloadError(f"zip slip detectado: {member.filename}")
    z.extractall(dest)
