"""Probe whether RST/DC pins on the GPIO chip actually reach the HAT+.

Set each control pin as INPUT and read it. If the HAT+ is connected, it
typically has weak pull-ups on RST and DC, so we'd expect to read HIGH.
If we read LOW or floating noise, the GPIO line isn't actually electrically
connected to the HAT+ pin.
"""
from __future__ import annotations
import time

from epd.driver import gpio_lepotato as GPIO
from epd.driver.spectra6 import BUSY_PIN, DC_PIN, PWR_PIN, RST_PIN


def read_as_input(name, bcm):
    GPIO.setup(bcm, GPIO.IN)
    samples = [GPIO.input(bcm) for _ in range(100)]
    hi = sum(samples)
    print(f"  {name} (BCM {bcm}): {hi}/100 reads HIGH  -> {'STABLE_HIGH' if hi >= 95 else 'STABLE_LOW' if hi <= 5 else 'FLOATING'}")


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    print("== With PWR=LOW (panel cold) ==")
    GPIO.setup(PWR_PIN, GPIO.OUT, initial=GPIO.LOW)
    time.sleep(0.2)
    read_as_input("RST", RST_PIN)
    read_as_input("DC", DC_PIN)
    read_as_input("BUSY", BUSY_PIN)

    print("\n== With PWR=HIGH (panel powered) ==")
    GPIO.setup(PWR_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.output(PWR_PIN, GPIO.HIGH)
    time.sleep(0.5)
    read_as_input("RST", RST_PIN)
    read_as_input("DC", DC_PIN)
    read_as_input("BUSY", BUSY_PIN)

    GPIO.cleanup()


if __name__ == "__main__":
    main()
