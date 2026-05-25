"""RPi.GPIO-compatible shim for Libre Computer AML-S905X-CC (Le Potato).

Waveshare's e-Paper Python code imports `RPi.GPIO as GPIO` and calls
`GPIO.setmode(GPIO.BCM)`, `GPIO.setup(pin, GPIO.OUT)`, etc., using Pi BCM
pin numbers. The Le Potato has no BCM pins; instead each physical header
pin maps to a (gpiochip, line) tuple in libgpiod.

This module maps the BCM numbers Waveshare uses to Le Potato gpiochip lines
discovered with `gpioinfo` on the device, and exposes a small subset of the
RPi.GPIO API (just what the Waveshare e-Paper driver uses).
"""

from __future__ import annotations

import atexit
from typing import Dict, Tuple

import gpiod

BCM = "BCM"
BOARD = "BOARD"
OUT = "out"
IN = "in"
HIGH = 1
LOW = 0

_BCM_TO_PIN_HEADER: Dict[int, int] = {
    17: 11,
    18: 12,
    24: 18,
    25: 22,
}

_HEADER_PIN_TO_GPIOD: Dict[int, Tuple[str, int]] = {
    11: ("gpiochip0", 8),
    12: ("gpiochip0", 6),
    18: ("gpiochip1", 94),
    22: ("gpiochip1", 79),
}

_chips: Dict[str, gpiod.Chip] = {}
_lines: Dict[int, gpiod.Line] = {}
_directions: Dict[int, str] = {}


def _line_for_bcm(bcm: int) -> gpiod.Line:
    if bcm in _lines:
        return _lines[bcm]
    header = _BCM_TO_PIN_HEADER.get(bcm)
    if header is None:
        raise ValueError(f"BCM pin {bcm} not mapped on Le Potato")
    chip_name, offset = _HEADER_PIN_TO_GPIOD[header]
    if chip_name not in _chips:
        _chips[chip_name] = gpiod.Chip(chip_name)
    line = _chips[chip_name].get_line(offset)
    _lines[bcm] = line
    return line


def setmode(mode):
    if mode != BCM:
        raise NotImplementedError("Only BCM mode supported")


def setwarnings(_):
    pass


def setup(pin, direction, initial=None, pull_up_down=None):
    line = _line_for_bcm(pin)
    if direction == OUT:
        default = LOW if initial is None else int(initial)
        line.request(consumer="epd", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[default])
        _directions[pin] = OUT
    elif direction == IN:
        line.request(consumer="epd", type=gpiod.LINE_REQ_DIR_IN)
        _directions[pin] = IN
    else:
        raise ValueError(f"Unknown direction {direction}")


def output(pin, value):
    line = _line_for_bcm(pin)
    line.set_value(int(value))


def input(pin):
    line = _line_for_bcm(pin)
    return line.get_value()


def cleanup(pin=None):
    targets = [pin] if pin is not None else list(_lines.keys())
    for bcm in targets:
        line = _lines.get(bcm)
        if line is None:
            continue
        try:
            line.release()
        except Exception:
            pass
        _lines.pop(bcm, None)
        _directions.pop(bcm, None)
    if pin is None:
        for chip in _chips.values():
            try:
                chip.close()
            except Exception:
                pass
        _chips.clear()


atexit.register(cleanup)
