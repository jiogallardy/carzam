import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

from carzam.audio import load_wav, resample_to
from carzam.data.dataset import SAMPLE_RATE, compute_logmel
from carzam.models.multihead import CARS, STATES, CarAudioModel


def _load_classes(checkpoint: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read classes.json next to the checkpoint, falling back to current defaults."""
    classes_path = Path(checkpoint).parent / "classes.json"
    if classes_path.exists():
        data = json.loads(classes_path.read_text())
        return tuple(data["cars"]), tuple(data["states"])
    return CARS, STATES


def _load_model(checkpoint: Path, device: torch.device) -> tuple[CarAudioModel, tuple[str, ...], tuple[str, ...]]:
    cars, states = _load_classes(checkpoint)
    # Auto-detect family + embedding heads from the saved state_dict so the
    # CLI handles every shipped generation (v6 fine-only, v7 family, v9-v11
    # family+contrastive). Same trick as the deployed API loader.
    sd = torch.load(checkpoint, map_location=device)
    n_families = sd["family_head.weight"].shape[0] if "family_head.weight" in sd else None
    embedding_dim: int | None = None
    classes_path = Path(checkpoint).parent / "classes.json"
    if classes_path.exists():
        classes_payload = json.loads(classes_path.read_text())
        embedding_dim = classes_payload.get("embedding_dim")
    if embedding_dim is None and "embedding_head.2.weight" in sd:
        embedding_dim = sd["embedding_head.2.weight"].shape[0]

    model = CarAudioModel(
        weights_path=None,
        n_cars=len(cars),
        n_states=len(states),
        n_families=n_families,
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cars, states

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


def _download_url_to_wav(
    url: str,
    out_dir: Path,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    out = out_dir / "clip.wav"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-o", str(out.with_suffix(".%(ext)s")),
    ]
    if start is not None and duration is not None:
        cmd += ["--download-sections", f"*{start}-{start + duration}"]
    cmd.append(url)
    subprocess.run(cmd, check=True)
    if not out.exists():
        cands = list(out_dir.glob("*.wav"))
        if cands:
            cands[0].rename(out)
    return out


def _predict_audio(
    audio: torch.Tensor,
    checkpoint: Path,
    device: torch.device,
) -> dict:
    """Predict on a single, already-prepared 5s audio tensor."""
    model, cars, states = _load_model(checkpoint, device)
    logmel = compute_logmel(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        car_logits, state_logits, _ = model(logmel)
        car_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()
        state_probs = F.softmax(state_logits, dim=-1).squeeze(0).cpu().numpy()
    car_top1 = int(np.argmax(car_probs))
    state_top1 = int(np.argmax(state_probs))
    car_conf = float(car_probs[car_top1])
    car_label = cars[car_top1] if car_conf >= OTHER_THRESHOLD else "other"
    car_order = np.argsort(-car_probs)
    state_order = np.argsort(-state_probs)
    return {
        "car": car_label,
        "car_confidence": car_conf,
        "state": states[state_top1],
        "state_confidence": float(state_probs[state_top1]),
        "car_top3": [(cars[i], float(car_probs[i])) for i in car_order[:3]],
        "state_all": [(states[i], float(state_probs[i])) for i in state_order],
        "n_windows": 1,
        "windows": [],
    }


def listen_and_predict(
    checkpoint: Path,
    duration: float = WINDOW_SECONDS,
    device: torch.device | None = None,
    sr: int = 16000,
) -> dict:
    """Record `duration` seconds from the default mic and predict."""
    device = device or torch.device("cpu")
    n_samples = int(duration * sr)
    rec = sd.rec(n_samples, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio_np = rec.squeeze()
    audio_np = resample_to(audio_np, src_sr=sr, dst_sr=SAMPLE_RATE)
    target_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(audio_np) < target_len:
        audio_np = np.concatenate([audio_np, np.zeros(target_len - len(audio_np), dtype=np.float32)])
    elif len(audio_np) > target_len:
        audio_np = audio_np[:target_len]
    audio = torch.from_numpy(np.ascontiguousarray(audio_np))
    return _predict_audio(audio, checkpoint, device)


def predict_clip(
    path_or_url: str | Path,
    checkpoint: Path,
    device: torch.device | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> dict:
    """Predict car + state with test-time aggregation.

    For clips longer than 5s, slides 5s windows with 2.5s hop, runs the model on
    each, and averages the softmaxes. Per-window predictions are also returned
    so callers can inspect the timeline.

    `start` + `duration` (in seconds) trim the input before prediction. For URLs
    we pass them to yt-dlp so only that slice is downloaded; for local files we
    trim after loading.
    """
    device = device or torch.device("cpu")
    if _is_url(path_or_url):
        with tempfile.TemporaryDirectory() as td:
            wav = _download_url_to_wav(
                str(path_or_url), Path(td), start=start, duration=duration
            )
            # already trimmed at download time; don't re-trim
            return predict_clip(wav, checkpoint, device)

    windows, starts = _prepare_windows(Path(path_or_url))
    model, cars, states = _load_model(checkpoint, device)

    car_probs_list: list[np.ndarray] = []
    state_probs_list: list[np.ndarray] = []
    per_window: list[dict] = []
    with torch.no_grad():
        for w, t in zip(windows, starts):
            logmel = compute_logmel(w).unsqueeze(0).to(device)
            cl, sl, _ = model(logmel)
            cp = F.softmax(cl, dim=-1).squeeze(0).cpu().numpy()
            sp = F.softmax(sl, dim=-1).squeeze(0).cpu().numpy()
            car_probs_list.append(cp)
            state_probs_list.append(sp)
            i = int(np.argmax(cp))
            j = int(np.argmax(sp))
            per_window.append({
                "start": float(t),
                "car": cars[i],
                "car_confidence": float(cp[i]),
                "state": states[j],
                "state_confidence": float(sp[j]),
            })

    car_probs = np.mean(car_probs_list, axis=0)
    state_probs = np.mean(state_probs_list, axis=0)

    car_top1 = int(np.argmax(car_probs))
    state_top1 = int(np.argmax(state_probs))
    car_conf = float(car_probs[car_top1])
    car_label = cars[car_top1] if car_conf >= OTHER_THRESHOLD else "other"

    car_order = np.argsort(-car_probs)
    state_order = np.argsort(-state_probs)
    return {
        "car": car_label,
        "car_confidence": car_conf,
        "state": states[state_top1],
        "state_confidence": float(state_probs[state_top1]),
        "car_top3": [(cars[i], float(car_probs[i])) for i in car_order[:3]],
        "state_all": [(states[i], float(state_probs[i])) for i in state_order],
        "n_windows": len(windows),
        "windows": per_window,
    }
