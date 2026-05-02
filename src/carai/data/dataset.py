import random

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from carai.audio import load_wav, resample_to
from carai.data.manifest import ManifestRow
from carai.models.multihead import CARS, STATES

SAMPLE_RATE = 32000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320  # 32000 * 5 / 320 ~= 500 frames per 5s window


def compute_logmel(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Returns log-mel of shape (frames, n_mels)."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=20.0,
        f_max=sample_rate / 2,
    )(audio)
    log_mel = torch.log(mel + 1e-6)
    return log_mel.squeeze(0).transpose(0, 1)  # (frames, n_mels)


class CarAudioDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        train: bool,
        seed: int = 0,
    ) -> None:
        self.rows = [r for r in rows if r.label is not None]
        self.train = train
        self.rng = random.Random(seed)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=40)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio, sr = load_wav(row.path)
        if sr != SAMPLE_RATE:
            audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
        audio_t = torch.from_numpy(np.ascontiguousarray(audio))

        if self.train:
            gain_db = (self.rng.random() * 12.0) - 6.0
            audio_t = audio_t * (10.0 ** (gain_db / 20.0))

        logmel = compute_logmel(audio_t)

        if self.train:
            logmel = logmel.transpose(0, 1).unsqueeze(0)  # (1, mel, frames)
            logmel = self.freq_mask(logmel)
            logmel = self.time_mask(logmel)
            logmel = logmel.squeeze(0).transpose(0, 1)

        return {
            "logmel": logmel,
            "car_idx": CARS.index(row.car),
            "state_idx": STATES.index(row.label),  # type: ignore[arg-type]
        }
