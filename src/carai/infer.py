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


def _prepare_audio(path: Path) -> torch.Tensor:
    audio, sr = load_wav(path)
    audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
    target_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(audio) < int(2.0 * SAMPLE_RATE):
        raise ValueError(f"clip too short: {len(audio) / SAMPLE_RATE:.1f}s, need >= 2s")
    if len(audio) < target_len:
        pad = np.zeros(target_len - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    elif len(audio) > target_len:
        excess = len(audio) - target_len
        start = excess // 2
        audio = audio[start : start + target_len]
    return torch.from_numpy(np.ascontiguousarray(audio))


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
    device = device or torch.device("cpu")
    if _is_url(path_or_url):
        with tempfile.TemporaryDirectory() as td:
            wav = _download_url_to_wav(str(path_or_url), Path(td))
            return predict_clip(wav, checkpoint, device)

    audio = _prepare_audio(Path(path_or_url))
    logmel = compute_logmel(audio).unsqueeze(0).to(device)

    model = CarAudioModel(weights_path=None).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        car_logits, state_logits = model(logmel)
        car_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()
        state_probs = F.softmax(state_logits, dim=-1).squeeze(0).cpu().numpy()

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
    }
