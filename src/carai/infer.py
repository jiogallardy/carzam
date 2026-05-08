import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from carai.audio import load_wav, resample_to
from carai.data.dataset import SAMPLE_RATE, compute_logmel
from carai.models.multihead import CARS, STATES, CarAudioModel

WINDOW_SECONDS = 5.0
OTHER_THRESHOLD = 0.40


def _prepare_windows(path: Path, hop_seconds: float = 2.5) -> tuple[list[torch.Tensor], list[float]]:
    """Slice the input clip into 5s windows. Returns (windows, start_times_in_seconds).
    For clips <= 5s, returns one padded window. For longer clips, returns overlapping
    windows hopping every hop_seconds.
    """
    audio, sr = load_wav(path)
    audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
    target_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop_len = int(hop_seconds * SAMPLE_RATE)

    if len(audio) < int(2.0 * SAMPLE_RATE):
        raise ValueError(f"clip too short: {len(audio) / SAMPLE_RATE:.1f}s, need >= 2s")

    windows: list[torch.Tensor] = []
    starts: list[float] = []
    if len(audio) <= target_len:
        if len(audio) < target_len:
            pad = np.zeros(target_len - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        windows.append(torch.from_numpy(np.ascontiguousarray(audio)))
        starts.append(0.0)
    else:
        start = 0
        while start + target_len <= len(audio):
            chunk = audio[start : start + target_len]
            windows.append(torch.from_numpy(np.ascontiguousarray(chunk)))
            starts.append(start / SAMPLE_RATE)
            start += hop_len
    return windows, starts


def _is_url(path_or_url: str | Path) -> bool:
    s = str(path_or_url)
    return s.startswith("http://") or s.startswith("https://")


def _download_url_to_wav(url: str, out_dir: Path) -> Path:
    out = out_dir / "clip.wav"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-o", str(out.with_suffix(".%(ext)s")),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not out.exists():
        cands = list(out_dir.glob("*.wav"))
        if cands:
            cands[0].rename(out)
    return out


def predict_clip(
    path_or_url: str | Path,
    checkpoint: Path,
    device: torch.device | None = None,
) -> dict:
    """Predict car + state with test-time aggregation.

    For clips longer than 5s, slides 5s windows with 2.5s hop, runs the model on
    each, and averages the softmaxes. Per-window predictions are also returned
    so callers can inspect the timeline.
    """
    device = device or torch.device("cpu")
    if _is_url(path_or_url):
        with tempfile.TemporaryDirectory() as td:
            wav = _download_url_to_wav(str(path_or_url), Path(td))
            return predict_clip(wav, checkpoint, device)

    windows, starts = _prepare_windows(Path(path_or_url))

    model = CarAudioModel(weights_path=None).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    car_probs_list: list[np.ndarray] = []
    state_probs_list: list[np.ndarray] = []
    per_window: list[dict] = []
    with torch.no_grad():
        for w, t in zip(windows, starts):
            logmel = compute_logmel(w).unsqueeze(0).to(device)
            cl, sl = model(logmel)
            cp = F.softmax(cl, dim=-1).squeeze(0).cpu().numpy()
            sp = F.softmax(sl, dim=-1).squeeze(0).cpu().numpy()
            car_probs_list.append(cp)
            state_probs_list.append(sp)
            i = int(np.argmax(cp))
            j = int(np.argmax(sp))
            per_window.append({
                "start": float(t),
                "car": CARS[i],
                "car_confidence": float(cp[i]),
                "state": STATES[j],
                "state_confidence": float(sp[j]),
            })

    car_probs = np.mean(car_probs_list, axis=0)
    state_probs = np.mean(state_probs_list, axis=0)

    car_top1 = int(np.argmax(car_probs))
    state_top1 = int(np.argmax(state_probs))
    car_conf = float(car_probs[car_top1])
    car_label = CARS[car_top1] if car_conf >= OTHER_THRESHOLD else "other"

    car_order = np.argsort(-car_probs)
    state_order = np.argsort(-state_probs)
    return {
        "car": car_label,
        "car_confidence": car_conf,
        "state": STATES[state_top1],
        "state_confidence": float(state_probs[state_top1]),
        "car_top3": [(CARS[i], float(car_probs[i])) for i in car_order[:3]],
        "state_all": [(STATES[i], float(state_probs[i])) for i in state_order],
        "n_windows": len(windows),
        "windows": per_window,
    }
