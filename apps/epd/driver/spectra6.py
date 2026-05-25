"""Waveshare 4-inch e-Paper Spectra 6 (E6 / 4in0e) driver.

Init sequence ported from the actual 4-inch reference driver at
e-Paper/E-paper_Separate_Program/4inch_e-Paper_E/.../epd4in0e.py — the
4-inch panel has a different init order and a critical extra 0x06
booster command in the display turn-on sequence vs the 7.3-inch.

Panel is natively 400 (W) x 600 (H) portrait. Image data is sent as
4 bits/pixel, two pixels per byte (high nibble = left/top pixel).
"""

from __future__ import annotations

import time

from . import GPIO

WIDTH = 400
HEIGHT = 600

RST_PIN = 17   # BCM 17 -> header pin 11
DC_PIN = 25    # BCM 25 -> header pin 22
BUSY_PIN = 24  # BCM 24 -> header pin 18
PWR_PIN = 18   # BCM 18 -> header pin 12


class Spectra6:
    def __init__(self, spi_bus: int = 0, spi_device: int = 0, spi_hz: int = 4_000_000):
        import spidev
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = spi_hz
        self.spi.mode = 0
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(RST_PIN, GPIO.OUT)
        GPIO.setup(DC_PIN, GPIO.OUT)
        GPIO.setup(PWR_PIN, GPIO.OUT)
        GPIO.setup(BUSY_PIN, GPIO.IN)
        GPIO.output(PWR_PIN, GPIO.HIGH)
        time.sleep(0.1)

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass
        try:
            GPIO.output(PWR_PIN, GPIO.LOW)
        except Exception:
            pass
        GPIO.cleanup()

    def _send_command(self, cmd: int):
        GPIO.output(DC_PIN, GPIO.LOW)
        self.spi.writebytes([cmd])

    def _send_data_byte(self, data: int):
        GPIO.output(DC_PIN, GPIO.HIGH)
        self.spi.writebytes([data])

    def _send_data_bulk(self, data):
        GPIO.output(DC_PIN, GPIO.HIGH)
        chunk = 4096
        mv = memoryview(data)
        for i in range(0, len(mv), chunk):
            self.spi.writebytes2(mv[i:i + chunk])

    def _reset(self):
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.020)
        GPIO.output(RST_PIN, GPIO.LOW)
        time.sleep(0.002)
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.020)

    def _wait_busy(self, timeout_s: float = 60.0):
        deadline = time.monotonic() + timeout_s
        while GPIO.input(BUSY_PIN) == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("e-paper BUSY did not release")
            time.sleep(0.005)
        time.sleep(0.200)

    def _cmd(self, c, *data):
        self._send_command(c)
        for b in data:
            self._send_data_byte(b)

    def init(self):
        self._reset()
        self._wait_busy()
        time.sleep(0.030)

        self._cmd(0xAA, 0x49, 0x55, 0x20, 0x08, 0x09, 0x18)
        self._cmd(0x01, 0x3F)
        self._cmd(0x00, 0x5F, 0x69)
        self._cmd(0x05, 0x40, 0x1F, 0x1F, 0x2C)
        self._cmd(0x08, 0x6F, 0x1F, 0x1F, 0x22)
        self._cmd(0x06, 0x6F, 0x1F, 0x17, 0x17)
        self._cmd(0x03, 0x00, 0x54, 0x00, 0x44)
        self._cmd(0x60, 0x02, 0x00)
        self._cmd(0x30, 0x08)
        self._cmd(0x50, 0x3F)
        # TRES: height(600=0x0190) then width(400=0x0190)... actually
        # for the 4in panel the bytes are H_high H_low W_high W_low
        # H=600=0x0258, W=400=0x0190
        self._cmd(0x61, 0x01, 0x90, 0x02, 0x58)
        self._cmd(0xE3, 0x2F)
        self._cmd(0x84, 0x01)
        self._wait_busy()

    def _turn_on(self):
        self._cmd(0x04)
        self._wait_busy()
        self._cmd(0x06, 0x6F, 0x1F, 0x17, 0x27)
        time.sleep(0.200)
        self._cmd(0x12, 0x00)
        self._wait_busy()
        self._cmd(0x02, 0x00)
        self._wait_busy()

    def display(self, packed: bytes):
        expected = WIDTH * HEIGHT // 2
        if len(packed) != expected:
            raise ValueError(f"expected {expected} bytes, got {len(packed)}")
        self._send_command(0x10)
        self._send_data_bulk(packed)
        self._turn_on()

    def clear(self, color: int = 0x1):
        nibble = (color << 4) | color
        buf = bytes([nibble]) * (WIDTH * HEIGHT // 2)
        self.display(buf)

    def sleep(self):
        self._send_command(0x07)
        self._send_data_byte(0xA5)
        time.sleep(2.0)
        GPIO.output(PWR_PIN, GPIO.LOW)
