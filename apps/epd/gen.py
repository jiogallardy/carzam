"""Generate cartoon car images via Gemini's image-gen models.

Uses gemini-2.5-flash-image-preview (Gemini multimodal image gen) by default
because it's free-tier accessible. Falls back to imagen-3.0-generate-002 if
the user sets EPD_IMAGE_MODEL=imagen.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

DEFAULT_PROMPT_TEMPLATE = (
    "Flat-color cartoon illustration of {subject}, viewed from a 45-degree "
    "rear three-quarter angle, parked on a {setting} street. Bold black outlines, "
    "solid color fills, no gradients, no shading, no halftone, no anti-aliasing. "
    "Limited color palette using only colors close to: pure black, white, red, "
    "yellow, blue, green. Light or white background. In the top-right corner, "
    "render the title \"{title}\" in a bold serif badging font similar to the "
    "classic Porsche crest typeface — confident, slightly italic, with strong "
    "letter weight. Aspect ratio 3:2, designed for a 600 by 400 pixel color e-ink display."
)


def build_prompt(
    subject: str,
    title: Optional[str] = None,
    setting: str = "european cobblestone",
) -> str:
    if title is None:
        title = subject.upper()
    return DEFAULT_PROMPT_TEMPLATE.format(subject=subject, title=title, setting=setting)


def generate_png(
    subject: str,
    *,
    title: Optional[str] = None,
    setting: str = "european cobblestone",
    out_path: str | Path,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Path:
    """Generate an image and save as PNG. Returns the path."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    model = model or os.environ.get("EPD_IMAGE_MODEL", "gemini")
    prompt = build_prompt(subject=subject, title=title, setting=setting)

    if model.startswith("imagen"):
        imagen_model = model if "-" in model and model != "imagen" else "imagen-4.0-generate-001"
        png_bytes = _gen_via_imagen(prompt, api_key, imagen_model)
    else:
        gemini_model = model if model.startswith("gemini-") else "gemini-2.5-flash-image"
        png_bytes = _gen_via_gemini(prompt, api_key, gemini_model)

    out_path = Path(out_path)
    out_path.write_bytes(png_bytes)
    return out_path


def _gen_via_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash-image") -> bytes:
    """Use Gemini multimodal image generation."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for cand in response.candidates or []:
        for part in (cand.content.parts if cand.content else []):
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                data = inline.data
                if isinstance(data, str):
                    data = base64.b64decode(data)
                return data
    raise RuntimeError("Gemini returned no image part")


def _gen_via_imagen(prompt: str, api_key: str, model: str = "imagen-4.0-generate-001") -> bytes:
    """Use Imagen 4 (higher quality, may require paid tier)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="3:2"),
    )
    img = response.generated_images[0].image
    data = img.image_bytes
    if isinstance(data, str):
        data = base64.b64decode(data)
    return data
