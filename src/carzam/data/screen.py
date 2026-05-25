"""Audio cleanliness gates for the auto-label pipeline.

A 5s window passes screening if it looks like real engine sound and not
voiceover, music, or ambient junk. We do NOT try to surgically remove
contamination — we just drop bad windows. YouTube supply is effectively
infinite, so it's cheaper to throw away 70% of a video than to wrestle
with source separation artifacts.

Gates:
  * silero VAD       — drop if speech ratio > 0.20
  * LAION CLAP       — prompt-based; engine score must beat music + voice
  * harmonicity      — engines have strong harmonics; pure-noise/wind doesn't

The caller composes these into a verdict. Each gate is also independently
useful (e.g. you might want to keep a window that has speech but is clearly
engine-dominated, by raising the VAD threshold).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# CLAP wants 48kHz mono. silero VAD wants 16kHz mono.
CLAP_SR = 48_000
VAD_SR = 16_000


@dataclass
class CleanlinessVerdict:
    is_clean: bool
    speech_ratio: float
    clap_scores: dict[str, float] = field(default_factory=dict)
    harmonicity: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_clean": self.is_clean,
            "speech_ratio": self.speech_ratio,
            "clap_scores": self.clap_scores,
            "harmonicity": self.harmonicity,
            "reasons": self.reasons,
        }


# ---------- silero VAD ----------

_VAD_CACHE: dict[str, object] = {}


def load_silero_vad():
    """Returns the silero-vad model (PyTorch). Cached after first call.

    Uses the `silero-vad` pip package which ships the ONNX weights.
    """
    if "model" in _VAD_CACHE:
        return _VAD_CACHE["model"]
    from silero_vad import load_silero_vad as _load

    model = _load()  # torch.jit module
    _VAD_CACHE["model"] = model
    return model


def vad_speech_ratio(audio_16k: np.ndarray, vad_model, threshold: float = 0.5) -> float:
    """Fraction of the clip that silero VAD calls speech.

    `audio_16k` must be 16kHz mono float32 in [-1, 1].
    """
    from silero_vad import get_speech_timestamps

    if audio_16k.dtype != np.float32:
        audio_16k = audio_16k.astype(np.float32)
    audio_t = torch.from_numpy(audio_16k)
    speech = get_speech_timestamps(
        audio_t, vad_model,
        sampling_rate=VAD_SR,
        threshold=threshold,
        # Default min_speech_duration_ms is 250ms — keeps single-word grunts out
    )
    total = len(audio_16k) / VAD_SR
    if total <= 0:
        return 0.0
    speech_secs = sum((s["end"] - s["start"]) / VAD_SR for s in speech)
    return min(1.0, speech_secs / total)


# ---------- CLAP zero-shot scoring ----------

_CLAP_CACHE: dict[str, object] = {}

# Multi-prompt averaging for each label. CLAP's training mixed brand names
# with descriptive phrasing — both forms carry signal.
CLAP_PROMPTS: dict[str, list[str]] = {
    "engine": [
        "the sound of a car engine revving",
        "a sports car engine accelerating",
        "loud car exhaust note",
        "a high-performance car engine",
    ],
    "music": [
        "music playing",
        "background music with drums and bass",
        "electronic dance music",
        "rock music with guitars",
    ],
    "voice": [
        "a person talking",
        "a man speaking",
        "a woman speaking",
        "voice-over narration",
    ],
    "wind": [
        "wind noise",
        "windy weather",
        "wind blowing into a microphone",
    ],
    "silence": [
        "silence",
        "quiet background ambient noise",
    ],
}


def load_clap(model_id: str = "laion/clap-htsat-fused", device: torch.device | None = None):
    """Loads LAION-CLAP and pre-computes text embeddings for the gate prompts.

    Returns (model, processor, text_embeds: dict[label]->(D,)) all on `device`.
    """
    from transformers import ClapModel, ClapProcessor

    cache_key = f"{model_id}@{device}"
    if cache_key in _CLAP_CACHE:
        return _CLAP_CACHE[cache_key]

    if device is None:
        device = torch.device("cpu")
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device)
    model.eval()

    text_embeds: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for label, prompts in CLAP_PROMPTS.items():
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            emb = model.get_text_features(**inputs)
            # transformers 5.x wraps the embedding in BaseModelOutputWithPooling;
            # 4.x returned the raw tensor. Accept both.
            emb = _as_pooled(emb)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            text_embeds[label] = emb.mean(dim=0)  # prompt-ensemble
    payload = (model, processor, text_embeds, device)
    _CLAP_CACHE[cache_key] = payload
    return payload


def _as_pooled(out) -> torch.Tensor:
    """Extract the (B, D) embedding from a CLAP feature call.

    transformers 4.x: tensor directly. 5.x: BaseModelOutputWithPooling.
    """
    if torch.is_tensor(out):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "audio_embeds") and out.audio_embeds is not None:
        return out.audio_embeds
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    raise TypeError(f"Cannot extract pooled embedding from {type(out).__name__}")


@torch.no_grad()
def clap_label_scores(audio_48k: np.ndarray, clap_loaded) -> dict[str, float]:
    """Cosine sim of audio against each ensembled text-label embedding.

    Returns scores in (0, 1)-ish range (cosine of two unit vectors mapped
    to [-1, 1]; in practice CLAP gives small positives). We return softmax
    over labels for easier thresholding.
    """
    model, processor, text_embeds, device = clap_loaded
    inputs = processor(audio=[audio_48k.astype(np.float32)],
                       sampling_rate=CLAP_SR, return_tensors="pt").to(device)
    audio_emb = model.get_audio_features(**inputs)
    audio_emb = _as_pooled(audio_emb)
    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1).squeeze(0)

    labels = list(text_embeds.keys())
    sims = torch.stack([torch.dot(audio_emb, text_embeds[l]) for l in labels])
    probs = torch.softmax(sims / 0.05, dim=-1)  # low temperature → confident gate
    return {l: float(p) for l, p in zip(labels, probs.tolist())}


# ---------- harmonicity ----------

def harmonicity_score(audio: np.ndarray, sr: int) -> float:
    """Crude harmonic-energy ratio via autocorrelation.

    Engines have strong harmonic structure (cylinder firing → periodic).
    Wind, road noise, and white noise have weak autocorrelation peaks.
    Voice has harmonics too — that's why we have a separate VAD gate.

    Returns approximately [0, 1] — higher means more harmonic.
    """
    x = audio.astype(np.float32)
    if x.std() < 1e-6:
        return 0.0
    x = x - x.mean()
    # Limit autocorr lag to 40-400Hz fundamental range
    max_lag = sr // 40
    min_lag = sr // 400
    n = min(len(x), sr)  # cap at 1s for speed
    x = x[:n]
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    peak = float(ac[min_lag:max_lag].max()) if max_lag > min_lag else 0.0
    return max(0.0, min(1.0, peak))


# ---------- combined gate ----------

@dataclass
class ScreenThresholds:
    max_speech_ratio: float = 0.20
    min_engine_score: float = 0.40       # CLAP engine probability
    min_engine_margin_over_music: float = 0.10
    min_engine_margin_over_voice: float = 0.05
    min_harmonicity: float = 0.10


def screen_window(
    audio_16k: np.ndarray,
    audio_48k: np.ndarray,
    vad_model,
    clap_loaded,
    thresholds: ScreenThresholds = ScreenThresholds(),
) -> CleanlinessVerdict:
    """Apply all gates to one 5s window and return the verdict.

    Caller is responsible for providing both samplerates of the audio
    (resample once in the orchestrator, not per-gate).
    """
    reasons: list[str] = []

    speech = vad_speech_ratio(audio_16k, vad_model)
    if speech > thresholds.max_speech_ratio:
        reasons.append(f"speech={speech:.2f}")

    clap = clap_label_scores(audio_48k, clap_loaded)
    eng = clap.get("engine", 0.0)
    mus = clap.get("music", 0.0)
    voc = clap.get("voice", 0.0)
    if eng < thresholds.min_engine_score:
        reasons.append(f"engine={eng:.2f}<min")
    if eng - mus < thresholds.min_engine_margin_over_music:
        reasons.append(f"music_competes={mus:.2f}")
    if eng - voc < thresholds.min_engine_margin_over_voice:
        reasons.append(f"voice_competes={voc:.2f}")

    harm = harmonicity_score(audio_16k, VAD_SR)
    if harm < thresholds.min_harmonicity:
        reasons.append(f"harm={harm:.2f}")

    return CleanlinessVerdict(
        is_clean=not reasons,
        speech_ratio=speech,
        clap_scores=clap,
        harmonicity=harm,
        reasons=reasons,
    )
