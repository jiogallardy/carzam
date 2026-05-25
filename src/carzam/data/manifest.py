import csv
from dataclasses import asdict, dataclass
from pathlib import Path

LABELS = ("idle", "accel", "decel")
FIELDS = ("path", "car", "source_video", "start_time", "label")


@dataclass
class ManifestRow:
    path: str
    car: str
    source_video: str
    start_time: float
    label: str | None  # None means unlabeled

    def __post_init__(self):
        if self.label is not None and self.label not in LABELS:
            raise ValueError(
                f"label must be one of {LABELS} or None, got {self.label!r}"
            )


def write_manifest(csv_path: Path | str, rows: list[ManifestRow]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            d["label"] = "" if d["label"] is None else d["label"]
            writer.writerow(d)


def read_manifest(csv_path: Path | str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(csv_path).open() as f:
        for r in csv.DictReader(f):
            rows.append(
                ManifestRow(
                    path=r["path"],
                    car=r["car"],
                    source_video=r["source_video"],
                    start_time=float(r["start_time"]),
                    label=r["label"] if r["label"] else None,
                )
            )
    return rows


def append_row(csv_path: Path | str, row: ManifestRow) -> None:
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        d = asdict(row)
        d["label"] = "" if d["label"] is None else d["label"]
        writer.writerow(d)
