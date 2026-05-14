"""Visual gate for auto-labeling.

Two-stage:
  1. YOLOv8 detects cars in a frame (just "is there a car, where").
  2. DINOv2 embeds each car crop and we cosine-sim it against a curated
     library of reference photos under data/references/<car>/*.{jpg,png}.
     Top-1 reference wins the crop.

Why DINOv2 + references instead of a Stanford-Cars classifier: the dataset
is from 2013 (no 488, no Huracan, no 720S). DINOv2 self-supervised
embeddings are SOTA for nearest-neighbor classification on novel classes
and we can extend by dropping new photos in a folder — no retraining.

Decision policy for a window of frames sampled during a 5s audio window:
  * "target visible"     → accept (strongest signal)
  * "no car visible"     → accept (POV/cockpit/hood cam — best audio)
  * "different car"      → reject (chase footage, vs comparison)
  * "mixed / multi-car"  → reject (can't isolate audio)
"""
from __future__ import annotations

import io
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# COCO class indices that we treat as "a car-like object"
_COCO_VEHICLE_IDS = {2, 5, 7}  # car, bus, truck — yolo's COCO mapping


@dataclass
class CarDetection:
    bbox: tuple[float, float, float, float]  # xyxy in pixels
    confidence: float
    coco_class: int


@dataclass
class FrameVerdict:
    """What we concluded about one sampled frame.

    Three useful states for the auto-label gate:
      * is_target          → target car visible in frame (best signal)
      * is_interior        → no exterior car visible, but the frame looks like
                             a car interior (POV cockpit, dash cam, hood cam).
                             The audio is presumably from inside the car.
      * is_empty_unrelated → no car AND not interior — random footage, reject.
    """
    n_cars_detected: int
    best_match_car: str | None     # car name from reference library
    best_match_score: float        # cosine sim, [0,1]
    is_target: bool                # best_match_car == target with high score
    is_empty: bool                 # no cars detected at all
    other_car_dominant: bool       # best match is a different car w/ high conf
    is_interior: bool = False      # whole-frame matches interior references
    interior_score: float = 0.0


@dataclass
class ReferenceIndex:
    cars: tuple[str, ...]
    # (N, D) L2-normalized embeddings, plus a parallel (N,) car-name array
    embeddings: torch.Tensor
    car_per_embedding: tuple[str, ...]
    encoder_name: str
    # Interior reference embeddings (whole-frame), used when no cars detected
    interior_embeddings: torch.Tensor | None = None
    encoder: object = field(repr=False, default=None)
    processor: object = field(repr=False, default=None)


# ---------- model loaders ----------

_DETECTOR_CACHE: dict[str, object] = {}


def load_car_detector(model_name: str = "yolov8n.pt"):
    """Ultralytics YOLOv8 (or v11). 'n' = nano, fastest. Lazy-load + cache.

    The first call downloads weights into the ultralytics cache dir.
    """
    if model_name in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[model_name]
    from ultralytics import YOLO  # local import — heavy

    model = YOLO(model_name)
    _DETECTOR_CACHE[model_name] = model
    return model


def load_image_encoder(model_name: str = "facebook/dinov2-small"):
    """DINOv2 (small) image encoder. Returns (model, processor).

    Small variant: ~22M params, 384-dim embeddings, plenty for nearest-neighbor.
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, processor


# ---------- inference primitives ----------

@torch.no_grad()
def _embed_images(
    images: list[Image.Image],
    encoder,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """Returns L2-normalized (N, D) embeddings."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    out = encoder(**inputs)
    # CLS token from last_hidden_state — DINOv2's image-level embedding
    emb = out.last_hidden_state[:, 0]
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


def detect_cars(
    image: Image.Image | np.ndarray,
    detector,
    min_confidence: float = 0.35,
    min_area_ratio: float = 0.01,
) -> list[CarDetection]:
    """Run YOLO on an image, keep only car/truck/bus boxes above thresholds."""
    if isinstance(image, Image.Image):
        arr = np.array(image)
    else:
        arr = image
    h, w = arr.shape[:2]
    min_area = min_area_ratio * h * w

    results = detector.predict(arr, verbose=False)
    dets: list[CarDetection] = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            if cls not in _COCO_VEHICLE_IDS:
                continue
            conf = float(boxes.conf[i].item())
            if conf < min_confidence:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            dets.append(CarDetection((x1, y1, x2, y2), conf, cls))
    return dets


def crop_box(image: Image.Image, box: tuple[float, float, float, float],
             pad_ratio: float = 0.04) -> Image.Image:
    """Crop with a small symmetric padding so DINOv2 sees a bit of context."""
    w, h = image.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    return image.crop((cx1, cy1, cx2, cy2))


# ---------- reference library ----------

INTERIOR_DIR_NAME = "_interior"  # special bucket: not a car class


def _gather_reference_photos(refs_dir: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """Walks data/references/. Returns (car -> photos, interior_photos).

    Folders beginning with '_' are treated as special buckets — currently
    only `_interior` is used. Everything else is a car class.
    """
    cars: dict[str, list[Path]] = {}
    interior: list[Path] = []
    if not refs_dir.exists():
        return cars, interior
    for sub in sorted(refs_dir.iterdir()):
        if not sub.is_dir():
            continue
        photos: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG"):
            photos.extend(sorted(sub.glob(ext)))
        if not photos:
            continue
        if sub.name == INTERIOR_DIR_NAME:
            interior = photos
        elif sub.name.startswith("_"):
            continue  # reserved for future special buckets
        else:
            cars[sub.name] = photos
    return cars, interior


def build_reference_index(
    refs_dir: Path,
    device: torch.device,
    encoder_name: str = "facebook/dinov2-small",
    detector_name: str = "yolov8n.pt",
    detect_in_references: bool = True,
) -> ReferenceIndex:
    """Encode every reference photo into a single tensor index.

    Exterior car references: YOLO-cropped, then DINOv2-embedded. The crop step
    is more robust than whole-image because reference photos often have
    background that dilutes the signal. Falls back to whole-image if YOLO
    finds nothing (rare — covers studio shots that confuse the detector).

    Interior references (data/references/_interior/*): whole-image embedding,
    no YOLO. Used for "you're inside a car" detection when no exterior car
    is visible — distinguishes POV cockpit / dash cam / hood cam (good audio)
    from unrelated empty footage (random pocket cam etc.).
    """
    photos_by_car, interior_photos = _gather_reference_photos(refs_dir)
    if not photos_by_car:
        raise FileNotFoundError(
            f"No reference photos found under {refs_dir}. "
            f"Expected data/references/<car>/*.jpg"
        )

    encoder, processor = load_image_encoder(encoder_name)
    encoder.to(device)
    detector = load_car_detector(detector_name) if detect_in_references else None

    all_crops: list[Image.Image] = []
    car_per_emb: list[str] = []

    for car, photos in photos_by_car.items():
        for p in photos:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            crop = img
            if detector is not None:
                dets = detect_cars(img, detector, min_confidence=0.25, min_area_ratio=0.01)
                if dets:
                    # Largest detected car
                    dets.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                              reverse=True)
                    crop = crop_box(img, dets[0].bbox)
            all_crops.append(crop)
            car_per_emb.append(car)

    # Embed in batches to control memory
    embs: list[torch.Tensor] = []
    batch = 16
    for i in range(0, len(all_crops), batch):
        embs.append(_embed_images(all_crops[i:i + batch], encoder, processor, device))
    embeddings = torch.cat(embs, dim=0)

    # Interior references — whole-image embedding.
    interior_embeddings: torch.Tensor | None = None
    if interior_photos:
        imgs: list[Image.Image] = []
        for p in interior_photos:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                continue
        if imgs:
            chunks = [
                _embed_images(imgs[i:i + batch], encoder, processor, device)
                for i in range(0, len(imgs), batch)
            ]
            interior_embeddings = torch.cat(chunks, dim=0)

    return ReferenceIndex(
        cars=tuple(sorted(photos_by_car.keys())),
        embeddings=embeddings,
        car_per_embedding=tuple(car_per_emb),
        encoder_name=encoder_name,
        interior_embeddings=interior_embeddings,
        encoder=encoder,
        processor=processor,
    )


@torch.no_grad()
def score_interior(
    image: Image.Image,
    ref_index: ReferenceIndex,
    device: torch.device,
) -> float:
    """Cosine sim of the whole frame against interior references.
    Returns 0.0 if no interior references were provided (gate is permissive
    in that case — caller decides whether to require interior detection)."""
    if ref_index.interior_embeddings is None or ref_index.interior_embeddings.numel() == 0:
        return 0.0
    emb = _embed_images([image], ref_index.encoder, ref_index.processor, device)
    sims = (emb @ ref_index.interior_embeddings.T).squeeze(0)
    return float(sims.max().item())


# ---------- per-frame classification ----------

@torch.no_grad()
def classify_crops(
    crops: list[Image.Image],
    ref_index: ReferenceIndex,
    device: torch.device,
) -> list[tuple[str, float]]:
    """For each crop returns (best_car, cosine_sim). Cosine = dot since both
    sides are L2-normalized."""
    if not crops:
        return []
    emb = _embed_images(crops, ref_index.encoder, ref_index.processor, device)
    sims = emb @ ref_index.embeddings.T  # (B, N)
    best_idx = sims.argmax(dim=-1)
    best_sim = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    return [
        (ref_index.car_per_embedding[int(i)], float(s))
        for i, s in zip(best_idx.tolist(), best_sim.tolist())
    ]


def classify_frame(
    image: Image.Image,
    target_car: str,
    detector,
    ref_index: ReferenceIndex,
    device: torch.device,
    target_match_threshold: float = 0.55,
    other_match_threshold: float = 0.55,
    interior_threshold: float = 0.55,
) -> FrameVerdict:
    """Detect cars in `image`, classify each, decide what this frame means
    for the target car.

    If no car is detected, fall back to interior detection — embed the whole
    frame and check against the interior reference set. This is what catches
    POV cockpit / dash cam / hood cam (good audio source) and distinguishes
    them from unrelated empty footage like pocket cams.
    """
    dets = detect_cars(image, detector)
    if not dets:
        interior_sim = score_interior(image, ref_index, device)
        is_interior = interior_sim >= interior_threshold
        return FrameVerdict(
            n_cars_detected=0,
            best_match_car=None,
            best_match_score=0.0,
            is_target=False,
            is_empty=True,
            other_car_dominant=False,
            is_interior=is_interior,
            interior_score=interior_sim,
        )

    crops = [crop_box(image, d.bbox) for d in dets]
    matches = classify_crops(crops, ref_index, device)
    # Best crop = highest-scoring assignment overall
    best_car, best_score = max(matches, key=lambda m: m[1])

    is_target = (best_car == target_car) and (best_score >= target_match_threshold)
    other_dominant = (best_car != target_car) and (best_score >= other_match_threshold)
    return FrameVerdict(
        n_cars_detected=len(dets),
        best_match_car=best_car,
        best_match_score=best_score,
        is_target=is_target,
        is_empty=False,
        other_car_dominant=other_dominant,
    )


# ---------- video-level helpers ----------

def sample_frames_with_ffmpeg(
    video_path: Path,
    timestamps: list[float],
) -> list[Image.Image]:
    """Pull single frames at the given timestamps using ffmpeg.

    Cheaper than decoding the whole video. One ffmpeg call per timestamp
    keeps memory low; if you're pulling hundreds, switch to a fps filter.
    """
    frames: list[Image.Image] = []
    for ts in timestamps:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True)
            frames.append(Image.open(io.BytesIO(result.stdout)).convert("RGB"))
        except (subprocess.CalledProcessError, Exception):
            continue
    return frames


def sample_frames_uniform_with_ffmpeg(
    video_path: Path,
    n_frames: int,
    duration: float | None = None,
) -> tuple[list[float], list[Image.Image]]:
    """Sample `n_frames` evenly across the video. Returns (timestamps, frames)."""
    if duration is None:
        duration = probe_video_duration(video_path)
    if duration <= 0:
        return [], []
    # Inset slightly from start/end to avoid blank intros/outros
    inset = min(2.0, 0.05 * duration)
    timestamps = list(np.linspace(inset, duration - inset, n_frames))
    return timestamps, sample_frames_with_ffmpeg(video_path, timestamps)


def probe_video_duration(video_path: Path) -> float:
    """Use ffprobe to get duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def screen_video_for_target(
    video_path: Path,
    target_car: str,
    ref_index: ReferenceIndex,
    detector,
    device: torch.device,
    n_frames: int = 10,
    min_target_ratio: float = 0.30,
) -> dict:
    """Video-level prefilter. Returns a dict with keep/drop and per-frame
    breakdown.

    Logic (sketches the chain the user asked for):
      * A frame is "acceptable" if it shows the target car OR is detected
        as a car interior.
      * A frame is "rejected" if a different car dominates the frame.
      * Empty non-interior frames are "unrelated" — neither acceptable nor
        rejected on their own, but they count against the acceptable ratio.

    We keep the video if `acceptable / total >= min_target_ratio` and no
    other car is dominating.
    """
    timestamps, frames = sample_frames_uniform_with_ffmpeg(video_path, n_frames)
    verdicts = [
        classify_frame(f, target_car, detector, ref_index, device)
        for f in frames
    ]

    n_total = max(1, len(verdicts))
    n_target = sum(1 for v in verdicts if v.is_target)
    n_other = sum(1 for v in verdicts if v.other_car_dominant)
    n_interior = sum(1 for v in verdicts if v.is_empty and v.is_interior)
    n_empty_unrelated = sum(
        1 for v in verdicts
        if v.is_empty and not v.is_interior
    )
    # "Acceptable" = target visible OR interior shot. Both yield good audio.
    n_acceptable = n_target + n_interior
    acceptable_ratio = n_acceptable / n_total
    other_ratio = n_other / n_total

    # Majority non-empty car (debug)
    car_counter: Counter[str] = Counter()
    for v in verdicts:
        if v.best_match_car and not v.is_empty:
            car_counter[v.best_match_car] += 1
    majority_car = car_counter.most_common(1)[0][0] if car_counter else None

    keep = (acceptable_ratio >= min_target_ratio) and (other_ratio <= 0.5)

    return {
        "keep": keep,
        "n_frames": n_total,
        "n_target": n_target,
        "n_other": n_other,
        "n_interior": n_interior,
        "n_empty_unrelated": n_empty_unrelated,
        "target_ratio": n_target / n_total,
        "interior_ratio": n_interior / n_total,
        "acceptable_ratio": acceptable_ratio,
        "other_ratio": other_ratio,
        "verdicts": verdicts,
        "timestamps": timestamps,
        "majority_car": majority_car,
    }


def screen_window_visual(
    video_path: Path,
    window_start: float,
    window_duration: float,
    target_car: str,
    ref_index: ReferenceIndex,
    detector,
    device: torch.device,
    n_frames_per_window: int = 3,
    require_acceptable: bool = True,
) -> dict:
    """Per-5s-window visual check. Samples a few frames inside the window
    and applies the same target/interior/other rule.

    If `require_acceptable` is True, the window must show either the target
    car or a car interior in at least one frame. With False, the window is
    only rejected if a different car dominates — useful for permissive runs
    where you trust the audio gate to catch unrelated junk.

    Called from auto_label.py after a window has passed the audio gate.
    """
    end = window_start + window_duration
    # Inset a hair so we don't grab the boundary frame of the prior window
    timestamps = list(np.linspace(window_start + 0.2, end - 0.2, n_frames_per_window))
    frames = sample_frames_with_ffmpeg(video_path, timestamps)
    verdicts = [
        classify_frame(f, target_car, detector, ref_index, device)
        for f in frames
    ]
    n = len(verdicts)
    if n == 0:
        # ffmpeg failed for this window — be permissive, let audio gate decide.
        return {"keep": True, "reason": "no-frames", "verdicts": []}

    n_target = sum(1 for v in verdicts if v.is_target)
    n_other = sum(1 for v in verdicts if v.other_car_dominant)
    n_interior = sum(1 for v in verdicts if v.is_empty and v.is_interior)
    n_unrelated = sum(1 for v in verdicts if v.is_empty and not v.is_interior)
    n_acceptable = n_target + n_interior

    if n_other / n > 0.5:
        return {
            "keep": False, "reason": "other-car-visible",
            "verdicts": verdicts, "n_other": n_other, "n_target": n_target,
            "n_interior": n_interior, "n_unrelated": n_unrelated,
        }

    if require_acceptable and n_acceptable == 0:
        # No target visible AND no interior — likely unrelated footage.
        return {
            "keep": False, "reason": "no-target-no-interior",
            "verdicts": verdicts, "n_other": n_other, "n_target": n_target,
            "n_interior": n_interior, "n_unrelated": n_unrelated,
        }

    reason = (
        "target-visible" if n_target > 0
        else "interior-only" if n_interior > 0
        else "permissive-pass"
    )
    return {
        "keep": True, "reason": reason,
        "verdicts": verdicts, "n_other": n_other, "n_target": n_target,
        "n_interior": n_interior, "n_unrelated": n_unrelated,
    }
