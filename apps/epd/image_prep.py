"""Image preprocessing for the Waveshare 4-inch Spectra 6 (E6) e-paper.

Pipeline: PIL Image -> EXIF-orient -> resize/letterbox to 600x400 ->
Floyd-Steinberg dither to the 6-color panel palette -> pack 4 bits/pixel
into the wire format the panel expects (high nibble = left pixel).

The Spectra 6 panel can only reproduce six colors. Dithering with a fixed
palette gives the best visual result for cartoons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

# Panel is natively 400x600 portrait. We accept user images in landscape
# (3:2, like 600x400) and rotate before packing so the image displays in
# landscape orientation when the panel is held with its long edge horizontal.
DISPLAY_W = 600   # what the user thinks of as width (landscape view)
DISPLAY_H = 400
PANEL_W = 400     # panel native width (portrait)
PANEL_H = 600     # panel native height
WIDTH = DISPLAY_W
HEIGHT = DISPLAY_H

PANEL_BLACK = 0x0
PANEL_WHITE = 0x1
PANEL_YELLOW = 0x2
PANEL_RED = 0x3
PANEL_BLUE = 0x5
PANEL_GREEN = 0x6

PALETTE_RGB = [
    (0, 0, 0),        # 0 black
    (255, 255, 255),  # 1 white
    (255, 243, 56),   # 2 yellow
    (191, 0, 0),      # 3 red
    (0, 0, 0),        # 4 (unused)
    (0, 0, 191),      # 5 blue
    (0, 153, 0),      # 6 green
]

_PANEL_INDEX_FOR_PALETTE_SLOT = [
    PANEL_BLACK,
    PANEL_WHITE,
    PANEL_YELLOW,
    PANEL_RED,
    PANEL_BLUE,
    PANEL_GREEN,
]

_DITHER_PALETTE_FLAT = [c for rgb in [
    PALETTE_RGB[PANEL_BLACK],
    PALETTE_RGB[PANEL_WHITE],
    PALETTE_RGB[PANEL_YELLOW],
    PALETTE_RGB[PANEL_RED],
    PALETTE_RGB[PANEL_BLUE],
    PALETTE_RGB[PANEL_GREEN],
] for c in rgb]

_DITHER_PALETTE_FLAT += [0] * (768 - len(_DITHER_PALETTE_FLAT))

_PALETTE_IMG = Image.new("P", (1, 1))
_PALETTE_IMG.putpalette(_DITHER_PALETTE_FLAT)


def _fit(img: Image.Image, mode: Literal["letterbox", "cover"]) -> Image.Image:
    src_ratio = img.width / img.height
    dst_ratio = DISPLAY_W / DISPLAY_H
    if mode == "letterbox":
        if src_ratio > dst_ratio:
            new_w = DISPLAY_W
            new_h = round(DISPLAY_W / src_ratio)
        else:
            new_h = DISPLAY_H
            new_w = round(DISPLAY_H * src_ratio)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (DISPLAY_W, DISPLAY_H), (255, 255, 255))
        canvas.paste(resized, ((DISPLAY_W - new_w) // 2, (DISPLAY_H - new_h) // 2))
        return canvas
    if mode == "cover":
        if src_ratio > dst_ratio:
            new_h = DISPLAY_H
            new_w = round(DISPLAY_H * src_ratio)
        else:
            new_w = DISPLAY_W
            new_h = round(DISPLAY_W / src_ratio)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - DISPLAY_W) // 2
        top = (new_h - DISPLAY_H) // 2
        return resized.crop((left, top, left + DISPLAY_W, top + DISPLAY_H))
    raise ValueError(f"unknown fit mode {mode}")


def load_and_prepare(
    path: str | Path,
    *,
    fit: Literal["letterbox", "cover"] = "letterbox",
    rotate: int = 0,
) -> Image.Image:
    """Load an image and convert to a 600x400 RGB image."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if rotate:
        img = img.rotate(rotate, expand=True)
    return _fit(img, fit)


def dither_to_palette(img: Image.Image) -> Image.Image:
    """Floyd-Steinberg dither down to the 6-color panel palette."""
    return img.quantize(palette=_PALETTE_IMG, dither=Image.FLOYDSTEINBERG)


def pack_for_panel(quantized: Image.Image) -> bytes:
    """Pack a quantized image into Spectra 6 wire format.

    Input is expected at PANEL native orientation (400 x 600 portrait),
    NOT landscape display orientation. Use rotate_to_panel() to flip
    landscape display images first.

    Two pixels per byte: high nibble = left pixel, low nibble = right pixel.
    """
    if quantized.mode != "P":
        raise ValueError("expected palette-mode image")
    if quantized.size != (PANEL_W, PANEL_H):
        raise ValueError(f"expected {PANEL_W}x{PANEL_H} (panel native), got {quantized.size}")

    raw = quantized.tobytes()
    out = bytearray(PANEL_W * PANEL_H // 2)
    for i in range(0, len(raw), 2):
        a = _PANEL_INDEX_FOR_PALETTE_SLOT[raw[i]]
        b = _PANEL_INDEX_FOR_PALETTE_SLOT[raw[i + 1]]
        out[i // 2] = (a << 4) | b
    return bytes(out)


def rotate_to_panel(img: Image.Image) -> Image.Image:
    """Rotate landscape display image (600x400) to panel native (400x600)."""
    if img.size == (PANEL_W, PANEL_H):
        return img
    if img.size == (DISPLAY_W, DISPLAY_H):
        return img.rotate(-90, expand=True)
    raise ValueError(f"unexpected size {img.size}")


def render_preview(quantized: Image.Image, out_path: str | Path) -> None:
    """Save a PNG showing exactly what the panel will display."""
    quantized.convert("RGB").save(out_path)


def prepare_and_pack(
    path: str | Path,
    *,
    fit: Literal["letterbox", "cover"] = "letterbox",
    rotate: int = 0,
    preview_path: str | Path | None = None,
) -> bytes:
    """Full pipeline from file path to wire-format bytes for the panel."""
    img = load_and_prepare(path, fit=fit, rotate=rotate)
    if preview_path:
        # render the landscape view the user sees, dithered to the panel palette
        landscape_q = dither_to_palette(img)
        render_preview(landscape_q, preview_path)
    panel_img = rotate_to_panel(img)
    quantized = dither_to_palette(panel_img)
    return pack_for_panel(quantized)
