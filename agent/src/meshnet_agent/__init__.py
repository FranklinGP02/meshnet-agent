"""Agente MeshNet: gestiona llama.cpp local y reporta al coordinador.

Debe ser ligero: typer + httpx + pydantic (+psutil/pynvml desde H1).
Prohibido importar SQLAlchemy/FastAPI (ADR-001). Para duraciones usa
time.monotonic(); los timestamps los pone siempre el coordinador.
"""

__version__ = "0.0.0"
