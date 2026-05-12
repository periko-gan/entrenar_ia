#!/usr/bin/env python3
"""Utilidades para resolver el dispositivo de ejecución (CPU/GPU)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceResolution:
    requested: str
    resolved: str
    warning: str | None = None


def _safe_torch_info() -> tuple[bool, int] | None:
    """Devuelve (cuda_available, cuda_count) o None si torch no está disponible."""
    try:
        import torch
    except Exception:
        return None

    try:
        return bool(torch.cuda.is_available()), int(torch.cuda.device_count())
    except Exception:
        return None


def resolve_device(requested_device: str) -> DeviceResolution:
    """Normaliza y valida el parametro --device con fallback seguro a CPU.

    Reglas:
    - `auto`: usa GPU 0 si CUDA está disponible, si no CPU.
    - `cpu`: fuerza CPU.
    - `0`, `0,1`, ...: si no hay CUDA, hace fallback a CPU con warning.
    """
    requested = (requested_device or "auto").strip().lower()

    if requested in ("", "auto"):
        torch_info = _safe_torch_info()
        if torch_info and torch_info[0] and torch_info[1] > 0:
            return DeviceResolution(requested="auto", resolved="0")
        return DeviceResolution(requested="auto", resolved="cpu")

    if requested == "cpu":
        return DeviceResolution(requested=requested, resolved="cpu")

    torch_info = _safe_torch_info()
    if not torch_info:
        return DeviceResolution(
            requested=requested,
            resolved="cpu",
            warning="No se pudo importar torch para validar CUDA; se usara CPU.",
        )

    cuda_available, cuda_count = torch_info
    if not cuda_available or cuda_count == 0:
        return DeviceResolution(
            requested=requested,
            resolved="cpu",
            warning=(
                f"Se solicito device={requested}, pero CUDA no esta disponible "
                "(torch.cuda.is_available()=False). Se usara CPU."
            ),
        )

    for token in (part.strip() for part in requested.split(",")):
        if not token:
            continue
        if token.isdigit() and int(token) >= cuda_count:
            raise ValueError(
                f"Device CUDA invalido: '{requested}'. GPUs detectadas: {cuda_count}."
            )

    return DeviceResolution(requested=requested, resolved=requested)

