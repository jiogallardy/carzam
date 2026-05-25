"""Hard power-cycle + long-reset attempt at waking the panel."""
from __future__ import annotations
import time

import spidev

from epd.driver import gpio_lepotato as GPIO
from epd.driver.spectra6 import BUSY_PIN, DC_PIN, PWR_PIN, RST_PIN


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DC_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUSY_PIN, GPIO.IN)

    print("Holding PWR LOW + RST LOW for 2s (full panel power-down)...")
    time.sleep(2.0)
    print(f"  BUSY = {GPIO.input(BUSY_PIN)} (should be 0/floating - unpowered)")

    print("\nApply PWR HIGH (panel boot)...")
    GPIO.output(PWR_PIN, GPIO.HIGH)
    time.sleep(0.5)
    print(f"  BUSY = {GPIO.input(BUSY_PIN)} (should be 1 - ready)")

    print("\nLong reset: RST HIGH 100ms, LOW 100ms, HIGH 100ms...")
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(RST_PIN, GPIO.HIGH)
    print("  BUSY during 1s after reset (sampling)...")
    for ms in (0, 50, 100, 200, 500, 1000):
        time.sleep(ms / 1000 if ms == 0 else 0.050)
        print(f"    t+{ms}ms BUSY={GPIO.input(BUSY_PIN)}")

    print("\nOpen SPI at 1 MHz, send init + try refresh...")
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1_000_000
    spi.mode = 0

    def cmd(c, *data):
        GPIO.output(DC_PIN, GPIO.LOW)
        spi.writebytes([c])
        if data:
            GPIO.output(DC_PIN, GPIO.HIGH)
            spi.writebytes(list(data))

    # full init (4in0e correct order)
    cmd(0xAA, 0x49, 0x55, 0x20, 0x08, 0x09, 0x18)
    cmd(0x01, 0x3F)
    cmd(0x00, 0x5F, 0x69)
    cmd(0x05, 0x40, 0x1F, 0x1F, 0x2C)
    cmd(0x08, 0x6F, 0x1F, 0x1F, 0x22)
    cmd(0x06, 0x6F, 0x1F, 0x17, 0x17)
    cmd(0x03, 0x00, 0x54, 0x00, 0x44)
    cmd(0x60, 0x02, 0x00)
    cmd(0x30, 0x08)
    cmd(0x50, 0x3F)
    cmd(0x61, 0x01, 0x90, 0x02, 0x58)
    cmd(0xE3, 0x2F)
    cmd(0x84, 0x01)
    print("  init sent, watching BUSY for 3s...")
    for i in range(30):
        time.sleep(0.1)
        b = GPIO.input(BUSY_PIN)
        if i % 5 == 0:
            print(f"    t+{i*100}ms BUSY={b}")

    print("\nSend tiny 100-byte test frame + 0x04 + 0x06 booster + 0x12 refresh...")
    GPIO.output(DC_PIN, GPIO.LOW)
    spi.writebytes([0x10])
    GPIO.output(DC_PIN, GPIO.HIGH)
    spi.writebytes([0x11] * 100)  # 100 bytes of white
    cmd(0x04)  # POWER_ON
    print("  after 0x04, watching BUSY for 3s...")
    for i in range(30):
        time.sleep(0.1)
        if GPIO.input(BUSY_PIN) == 0:
            print(f"    *** BUSY DROPPED at t+{i*100}ms ***")
            break

    cmd(0x06, 0x6F, 0x1F, 0x17, 0x27)
    time.sleep(0.2)
    cmd(0x12, 0x00)
    print("  after 0x12, watching BUSY for 30s...")
    busy_seen = False
    for i in range(300):
        time.sleep(0.1)
        b = GPIO.input(BUSY_PIN)
        if b == 0:
            if not busy_seen:
                print(f"    *** BUSY DROPPED at t+{i*100}ms ***")
                busy_seen = True
        elif busy_seen:
            print(f"    *** BUSY back HIGH at t+{i*100}ms ***")
            break
    if not busy_seen:
        print("  BUSY NEVER dropped over 30s — panel ignored 0x12")

    spi.close()
    GPIO.output(PWR_PIN, GPIO.LOW)
    GPIO.cleanup()


if __name__ == "__main__":
    main()
