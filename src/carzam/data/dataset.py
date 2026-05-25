import random

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from carzam.audio import load_wav, resample_to
from carzam.data.manifest import ManifestRow
from carzam.models.multihead import CARS, STATES, family_index_for_car

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
        use_families_as_classes: bool = False,
        cars_subset: tuple[str, ...] | None = None,
    ) -> None:
        """If `cars_subset` is given, only rows whose `car` is in that subset
        are kept, and `car_idx` is computed relative to the subset (not the
        global CARS tuple). Used for training family-specialist models in the
        v8 cascade — e.g. for the `na_flat6` specialist, cars_subset is
        ("porsche_gt3", "porsche_gt4") and the model has 2 outputs.
        """
        rows = [r for r in rows if r.label is not None]
        if cars_subset is not None:
            allowed = set(cars_subset)
            rows = [r for r in rows if r.car in allowed]
        self.rows = rows
        self.cars_subset = cars_subset
        self.train = train
        self.rng = random.Random(seed)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=40)
        self.use_families = use_families_as_classes

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

            # Background noise injection (50% of windows). Gaussian + pink mix at
            # random SNR; teaches the model to find engine signal under noise.
            if self.rng.random() < 0.5:
                snr_db = self.rng.uniform(0.0, 25.0)  # higher SNR = quieter noise
                signal_rms = float(torch.sqrt(torch.mean(audio_t * audio_t)).item()) + 1e-8
                noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
                # Mix of white + pink-ish noise
                noise = torch.randn_like(audio_t) * noise_rms
                if self.rng.random() < 0.5:
                    # Coarse "pink" approximation: low-pass filter the noise via cumsum
                    noise = torch.cumsum(noise, dim=0) - torch.cumsum(noise, dim=0).mean()
                    noise = noise / (noise.std() + 1e-8) * noise_rms
                audio_t = audio_t + noise

        logmel = compute_logmel(audio_t)

        if self.train:
            logmel = logmel.transpose(0, 1).unsqueeze(0)  # (1, mel, frames)
            logmel = self.freq_mask(logmel)
            logmel = self.time_mask(logmel)
            logmel = logmel.squeeze(0).transpose(0, 1)

        family_idx = family_index_for_car(row.car)
        if self.cars_subset is not None:
            car_idx = self.cars_subset.index(row.car)  # index within the family
        elif self.use_families:
            car_idx = family_idx
        else:
            car_idx = CARS.index(row.car)
        return {
            "logmel": logmel,
            "car_idx": car_idx,
            "family_idx": family_idx,
            "state_idx": STATES.index(row.label),  # type: ignore[arg-type]
        }
