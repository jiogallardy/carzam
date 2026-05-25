"""Auto-select GPIO backend based on detected board."""
from __future__ import annotations
import os


def _detect_board() -> str:
    """Return 'pi' for a Raspberry Pi, 'lepotato' for AML-S905X-CC, else 'unknown'."""
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().strip("\0").lower()
    except OSError:
        return "unknown"
    if "raspberry pi" in model:
        return "pi"
    if "libre" in model or "aml-s905x" in model or "le potato" in model:
        return "lepotato"
    return "unknown"


_board = os.environ.get("EPD_BOARD") or _detect_board()
if _board == "pi":
    from . import gpio_pi as GPIO  # noqa: F401
else:
    from . import gpio_lepotato as GPIO  # noqa: F401

from .spectra6 import Spectra6, WIDTH, HEIGHT  # noqa: F401,E402
