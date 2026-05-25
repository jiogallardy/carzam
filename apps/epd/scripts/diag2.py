"""Sanity-check the gpiochip1 line 94 -> pin 18 mapping for BUSY.

Reads BUSY as input WITHOUT panel involvement, then toggles a *different*
known-good output (RST on pin 11). BUSY should not move with RST toggles
if the HAT+ isn't connected (panel is what drives BUSY). If BUSY moves
in response to RST toggles, the line mapping is wrong.

Also reads BUSY many times to see if it's stable HIGH or noisy.
"""
from __future__ import annotations
import time

from epd.driver import gpio_lepotato as GPIO
from epd.driver.spectra6 import BUSY_PIN, RST_PIN


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUSY_PIN, GPIO.IN)

    print("Reading BUSY 20 times at 50ms intervals while RST is LOW:")
    for i in range(20):
        print(f"  t={i*50}ms BUSY={GPIO.input(BUSY_PIN)}")
        time.sleep(0.05)

    print("\nNow toggling RST 5 times — BUSY should NOT change if mapping correct:")
    for i in range(5):
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.05)
        b1 = GPIO.input(BUSY_PIN)
        GPIO.output(RST_PIN, GPIO.LOW)
        time.sleep(0.05)
        b2 = GPIO.input(BUSY_PIN)
        print(f"  cycle {i}: BUSY(RST=H)={b1}  BUSY(RST=L)={b2}")

    GPIO.cleanup()


if __name__ == "__main__":
    main()
