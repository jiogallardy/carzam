"""Aggressive panel test:
- Drop SPI to 500 kHz
- Run init, watching BUSY at each command for low pulses
- If we ever see BUSY go LOW, the panel is responding
"""
from __future__ import annotations

import time

import spidev

from epd.driver import gpio_lepotato as GPIO
from epd.driver.spectra6 import BUSY_PIN, DC_PIN, PWR_PIN, RST_PIN


def watch_busy(label, duration_s=0.5):
    """Sample BUSY for duration_s and print the min/max/transitions."""
    start = time.monotonic()
    samples = []
    while time.monotonic() - start < duration_s:
        samples.append(GPIO.input(BUSY_PIN))
    lo, hi = min(samples), max(samples)
    print(f"  [{label}] BUSY: min={lo} max={hi} samples={len(samples)} {'**TRANSITION**' if lo != hi else ''}")


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DC_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUSY_PIN, GPIO.IN)

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 500_000
    spi.mode = 0

    print("\n== Power on ==")
    GPIO.output(PWR_PIN, GPIO.HIGH)
    time.sleep(0.2)
    watch_busy("after PWR")

    print("\n== Reset ==")
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.020)
    GPIO.output(RST_PIN, GPIO.LOW)
    watch_busy("RST low (10ms)", 0.010)
    GPIO.output(RST_PIN, GPIO.HIGH)
    watch_busy("RST high - post reset (1s)", 1.0)

    print("\n== Send 0xAA init magic + data ==")
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0xAA])
    GPIO.output(DC_PIN, GPIO.HIGH)
    for b in (0x49, 0x55, 0x20, 0x08, 0x09, 0x18):
        spi.writebytes([b])
    watch_busy("after 0xAA + 6 bytes (1s)", 1.0)

    print("\n== Send POWER_ON 0x04 ==")
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0x04])
    watch_busy("right after 0x04", 0.5)
    watch_busy("0x04 + 2s", 2.0)
    watch_busy("0x04 + 4s", 2.0)

    print("\n== Send DISPLAY_REFRESH 0x12 0x00 ==")
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0x12])
    GPIO.output(DC_PIN, GPIO.HIGH)
    spi.writebytes([0x00])
    watch_busy("right after 0x12", 0.5)
    watch_busy("0x12 + 5s", 5.0)
    watch_busy("0x12 + 10s", 10.0)
    watch_busy("0x12 + 20s", 10.0)

    print("\n== DEEP_SLEEP 0x07 0xA5 ==")
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0x07])
    GPIO.output(DC_PIN, GPIO.HIGH)
    spi.writebytes([0xA5])
    time.sleep(1)
    watch_busy("post sleep", 0.5)

    spi.close()
    GPIO.output(PWR_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("\nDONE")


if __name__ == "__main__":
    main()
