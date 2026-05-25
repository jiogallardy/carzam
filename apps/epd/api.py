"""FastAPI push service for the e-paper.

POST /image (multipart) with field `file` -> dithers + displays.
Rate-limited to one refresh per `EPD_MIN_INTERVAL_S` seconds (default 60)
to protect the panel; supply `?force=1` to override.
Auth via `Authorization: Bearer <EPD_TOKEN>` if EPD_TOKEN is set.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .driver import Spectra6
from .image_prep import prepare_and_pack

app = FastAPI(title="epd-push")

_lock = threading.Lock()
_last_refresh = 0.0


def _require_token(authorization: str | None = Header(default=None)):
    expected = os.environ.get("EPD_TOKEN")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="bad token")


@app.get("/health")
def health():
    return {"ok": True, "last_refresh_s_ago": time.time() - _last_refresh if _last_refresh else None}


@app.post("/image")
async def push_image(
    file: UploadFile = File(...),
    fit: str = Query("letterbox", regex="^(letterbox|cover)$"),
    rotate: int = Query(0),
    force: bool = Query(False),
    _: None = Depends(_require_token),
):
    global _last_refresh
    min_interval = float(os.environ.get("EPD_MIN_INTERVAL_S", "60"))
    now = time.time()
    if not force and (now - _last_refresh) < min_interval:
        retry = min_interval - (now - _last_refresh)
        raise HTTPException(
            status_code=429,
            detail=f"too soon; retry in {retry:.0f}s or pass ?force=1",
        )

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty file")

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "img").suffix, delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name

    try:
        packed = prepare_and_pack(tmp_path, fit=fit, rotate=rotate)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="display busy")
    try:
        panel = Spectra6()
        try:
            panel.init()
            t0 = time.monotonic()
            panel.display(packed)
            elapsed = time.monotonic() - t0
        finally:
            panel.sleep()
            panel.close()
        _last_refresh = time.time()
    finally:
        _lock.release()

    return JSONResponse({"ok": True, "refresh_s": round(elapsed, 2)})


class PromptBody(BaseModel):
    subject: str
    title: str | None = None
    setting: str = "european cobblestone"
    force: bool = False


@app.post("/prompt")
async def push_prompt(body: PromptBody, _: None = Depends(_require_token)):
    global _last_refresh
    min_interval = float(os.environ.get("EPD_MIN_INTERVAL_S", "60"))
    now = time.time()
    if not body.force and (now - _last_refresh) < min_interval:
        retry = min_interval - (now - _last_refresh)
        raise HTTPException(429, f"too soon; retry in {retry:.0f}s or pass force=true")

    from .gen import generate_png
    tmp_path = tempfile.mkstemp(suffix=".png")[1]
    try:
        generate_png(subject=body.subject, title=body.title, setting=body.setting, out_path=tmp_path)
        packed = prepare_and_pack(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "display busy")
    try:
        panel = Spectra6()
        try:
            panel.init()
            t0 = time.monotonic()
            panel.display(packed)
            elapsed = time.monotonic() - t0
        finally:
            panel.sleep()
            panel.close()
        _last_refresh = time.time()
    finally:
        _lock.release()

    return JSONResponse({"ok": True, "refresh_s": round(elapsed, 2)})
