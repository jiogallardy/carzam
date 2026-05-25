"""Try every SPI mode and a range of speeds. For each, send the 0x12 refresh
command and wait up to 5s for BUSY to go LOW. If we EVER see BUSY drop,
that combo is the right one.
"""
from __future__ import annotations
import time

import spidev

from epd.driver import gpio_lepotato as GPIO
from epd.driver.spectra6 import BUSY_PIN, DC_PIN, PWR_PIN, RST_PIN


def reset_panel():
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.020)
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.005)
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.030)


def wait_for_busy_drop(timeout=5.0):
    """Return how long until BUSY drops to 0, or None if it never does."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if GPIO.input(BUSY_PIN) == 0:
            return time.monotonic() - start
    return None


def try_combo(mode, hz):
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = hz
    spi.mode = mode

    # full reset
    reset_panel()
    time.sleep(0.05)

    # send the same init from Waveshare reference
    def cmd(c, *data):
        GPIO.output(DC_PIN, GPIO.LOW)
        spi.writebytes([c])
        if data:
            GPIO.output(DC_PIN, GPIO.HIGH)
            spi.writebytes(list(data))

    cmd(0xAA, 0x49, 0x55, 0x20, 0x08, 0x09, 0x18)
    cmd(0x01, 0x3F)
    cmd(0x00, 0x5F, 0x69)
    cmd(0x03, 0x00, 0x54, 0x00, 0x44)
    cmd(0x05, 0x40, 0x1F, 0x1F, 0x2C)
    cmd(0x06, 0x6F, 0x1F, 0x17, 0x49)
    cmd(0x08, 0x6F, 0x1F, 0x1F, 0x22)
    cmd(0x30, 0x03)
    cmd(0x50, 0x3F)
    cmd(0x60, 0x02, 0x00)
    cmd(0x61, 0x02, 0x58, 0x01, 0x90)
    cmd(0x84, 0x01)
    cmd(0xE3, 0x2F)
    cmd(0x04)  # POWER_ON

    # quick whitespace check
    pre = wait_for_busy_drop(0.5)
    if pre is not None:
        print(f"  >>> BUSY dropped {pre:.3f}s after POWER_ON (mode={mode} hz={hz})")

    # one full-image write of all-white (small data)
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0x10])
    GPIO.output(DC_PIN, GPIO.HIGH)
    # send 1000 bytes of 0x11 (white) to keep it small; not a full frame
    spi.writebytes([0x11] * 1000)

    # trigger refresh
    cmd(0x12, 0x00)
    dt = wait_for_busy_drop(5.0)
    spi.close()
    return dt


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DC_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUSY_PIN, GPIO.IN)

    GPIO.output(PWR_PIN, GPIO.HIGH)
    time.sleep(0.3)

    for hz in (1_000_000, 2_000_000, 4_000_000, 8_000_000):
        for mode in (0,):
            print(f"\n-- mode={mode} hz={hz} --")
            dt = try_combo(mode, hz)
            if dt is None:
                print(f"  BUSY never dropped within 5s post-refresh")
            else:
                print(f"  *** BUSY DROPPED at +{dt:.3f}s !!! mode={mode} hz={hz} ***")

    GPIO.output(PWR_PIN, GPIO.LOW)
    GPIO.cleanup()


if __name__ == "__main__":
    main()
