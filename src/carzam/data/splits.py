import math
import random
from collections import defaultdict
from dataclasses import dataclass

from carzam.data.manifest import ManifestRow


@dataclass
class SplitResult:
    train: list[ManifestRow]
    val: list[ManifestRow]
    test: list[ManifestRow]


def split_by_video(
    rows: list[ManifestRow],
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitResult:
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    rows = [r for r in rows if r.label is not None]

    by_car_video: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.source_video not in seen[r.car]:
            seen[r.car].add(r.source_video)
            by_car_video[r.car].append(r.source_video)

    rng = random.Random(seed)
    train_videos: set[str] = set()
    val_videos: set[str] = set()
    test_videos: set[str] = set()
    for _, videos in by_car_video.items():
        videos = sorted(videos)
        rng.shuffle(videos)
        n = len(videos)
        if n == 1:
            train_videos.update(videos)
            continue
        if n == 2:
            train_videos.add(videos[0])
            test_videos.add(videos[1])
            continue
        # Guarantee at least 1 video in each of val and test, then fill train.
        n_test = max(1, int(round(ratios[2] * n)))
        n_val = max(1, int(round(ratios[1] * n)))
        n_train = n - n_test - n_val
        if n_train < 1:
            # too few videos to honor ratios — minimal viable split
            n_train, n_val, n_test = max(1, n - 2), 1, 1
        train_videos.update(videos[:n_train])
        val_videos.update(videos[n_train : n_train + n_val])
        test_videos.update(videos[n_train + n_val :])

    train, val, test = [], [], []
    for r in rows:
        if r.source_video in train_videos:
            train.append(r)
        elif r.source_video in val_videos:
            val.append(r)
        else:
            test.append(r)
    return SplitResult(train, val, test)
