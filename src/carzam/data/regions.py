from dataclasses import dataclass
from pathlib import Path

import yaml

REGION_LABELS = ("idle", "accel", "decel", "skip")


@dataclass
class Region:
    start: float
    end: float
    label: str

    def __post_init__(self):
        if self.label not in REGION_LABELS:
            raise ValueError(f"label must be one of {REGION_LABELS}, got {self.label!r}")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")


def write_regions(
    path: Path | str,
    video_id: str,
    duration: float,
    regions: list[Region],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "duration": float(duration),
        "ranges": [
            {"start": float(r.start), "end": float(r.end), "label": r.label}
            for r in regions
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def read_regions(path: Path | str) -> tuple[str, float, list[Region]]:
    data = yaml.safe_load(Path(path).read_text())
    regions = [Region(r["start"], r["end"], r["label"]) for r in data.get("ranges", [])]
    return data["video_id"], float(data["duration"]), regions


def regions_path_for(regions_dir: Path | str, car: str, video_id: str) -> Path:
    return Path(regions_dir) / car / f"{video_id}.yaml"


def label_for_time(
    regions: list[Region],
    t: float,
    window_seconds: float = 0.0,
) -> str | None:
    """Return the label of the region containing the time interval [t, t + window_seconds].
    Returns None if no single region fully contains it (skip regions count too).
    """
    end = t + window_seconds
    for r in regions:
        if r.start <= t and end <= r.end:
            return r.label
    return None
