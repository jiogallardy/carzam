"""Quick hardware diagnostic for the Waveshare HAT+ on Le Potato.

Probes BUSY/RST/DC/PWR pins to verify wiring and reads SPI to see if
the panel responds. Run on the Libre board.
"""
from __future__ import annotations

import time

from epd.driver import gpio_lepotato as GPIO
from epd.driver.spectra6 import BUSY_PIN, DC_PIN, PWR_PIN, RST_PIN


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    print("Setting up output pins (RST, DC, PWR)...")
    GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DC_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWR_PIN, GPIO.OUT, initial=GPIO.LOW)

    print("Setting up input pin (BUSY)...")
    GPIO.setup(BUSY_PIN, GPIO.IN)

    print()
    print("Initial pin reads:")
    print(f"  BUSY = {GPIO.input(BUSY_PIN)} (with PWR=0, panel unpowered)")

    print()
    print("Powering on (PWR=HIGH, RST=HIGH)...")
    GPIO.output(PWR_PIN, GPIO.HIGH)
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.1)
    print(f"  BUSY = {GPIO.input(BUSY_PIN)} (after PWR up, RST high)")

    print()
    print("Pulsing RST low->high (reset sequence)...")
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.02)
    GPIO.output(RST_PIN, GPIO.HIGH)
    print(f"  BUSY immediately after reset = {GPIO.input(BUSY_PIN)}")
    for ms in (10, 50, 100, 500, 1000, 2000):
        time.sleep((ms - sum([10, 50, 100, 500, 1000, 2000][:[10, 50, 100, 500, 1000, 2000].index(ms)])) / 1000)
        print(f"  BUSY at t={ms}ms = {GPIO.input(BUSY_PIN)}")

    print()
    print("Try SPI write to send 0x71 (get_status) and see if anything happens...")
    import spidev
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1_000_000
    spi.mode = 0
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0x71])
    print(f"  BUSY right after 0x71 = {GPIO.input(BUSY_PIN)}")
    time.sleep(0.1)
    print(f"  BUSY 100ms later     = {GPIO.input(BUSY_PIN)}")

    print()
    print("Powering off...")
    GPIO.output(PWR_PIN, GPIO.LOW)
    GPIO.output(RST_PIN, GPIO.LOW)
    GPIO.cleanup()


if __name__ == "__main__":
    main()
