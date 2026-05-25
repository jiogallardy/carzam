"""RPi.GPIO-compatible shim backed by gpiozero.

gpiozero auto-selects the right pin factory for the Pi model (works on
Pi 3/4 with RPi.GPIO, Pi 5 with lgpio). Same API surface as gpio_lepotato.
"""

from __future__ import annotations

import atexit
from typing import Dict

BCM = "BCM"
BOARD = "BOARD"
OUT = "out"
IN = "in"
HIGH = 1
LOW = 0

_outs: Dict[int, "object"] = {}
_ins: Dict[int, "object"] = {}


def setmode(mode):
    if mode != BCM:
        raise NotImplementedError("Only BCM mode supported")


def setwarnings(_):
    pass


def setup(pin, direction, initial=None, pull_up_down=None):
    import gpiozero
    if direction == OUT:
        dev = gpiozero.LED(pin, initial_value=bool(initial) if initial is not None else False)
        _outs[pin] = dev
    elif direction == IN:
        dev = gpiozero.Button(pin, pull_up=False)
        _ins[pin] = dev
    else:
        raise ValueError(f"unknown direction {direction}")


def output(pin, value):
    dev = _outs.get(pin)
    if dev is None:
        raise RuntimeError(f"pin {pin} not configured as OUT")
    if value:
        dev.on()
    else:
        dev.off()


def input(pin):
    dev = _ins.get(pin)
    if dev is None:
        raise RuntimeError(f"pin {pin} not configured as IN")
    return int(dev.value)


def cleanup(pin=None):
    targets = [pin] if pin is not None else list(_outs) + list(_ins)
    for p in targets:
        dev = _outs.pop(p, None) or _ins.pop(p, None)
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass


atexit.register(cleanup)
