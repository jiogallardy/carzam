import pytest

from carzam.data.manifest import ManifestRow
from carzam.data.splits import split_by_video


def make_rows(car: str, video: str, n: int, label: str = "idle") -> list[ManifestRow]:
    return [
        ManifestRow(
            path=f"data/windows/{car}/{video}_{i}.wav",
            car=car,
            source_video=video,
            start_time=float(i) * 2.5,
            label=label,
        )
        for i in range(n)
    ]


def test_no_video_leaks_across_splits():
    rows = []
    for car in ["ferrari_812", "porsche_gt3"]:
        for v in range(10):
            rows.extend(make_rows(car, f"{car}_v{v}", 5))
    res = split_by_video(rows, ratios=(0.7, 0.15, 0.15), seed=0)
    train_videos = {r.source_video for r in res.train}
    val_videos = {r.source_video for r in res.val}
    test_videos = {r.source_video for r in res.test}
    assert train_videos.isdisjoint(val_videos)
    assert train_videos.isdisjoint(test_videos)
    assert val_videos.isdisjoint(test_videos)


def test_each_car_present_in_train():
    rows = []
    for car in ["ferrari_812", "porsche_gt3", "amg_c63_m177"]:
        for v in range(10):
            rows.extend(make_rows(car, f"{car}_v{v}", 5))
    res = split_by_video(rows, (0.7, 0.15, 0.15), seed=42)
    train_cars = {r.car for r in res.train}
    assert train_cars == {"ferrari_812", "porsche_gt3", "amg_c63_m177"}


def test_seed_is_deterministic():
    rows = []
    for v in range(20):
        rows.extend(make_rows("ferrari_812", f"v{v}", 3))
    res1 = split_by_video(rows, (0.7, 0.15, 0.15), seed=7)
    res2 = split_by_video(rows, (0.7, 0.15, 0.15), seed=7)
    paths1 = [r.path for r in res1.train]
    paths2 = [r.path for r in res2.train]
    assert paths1 == paths2


def test_unlabeled_rows_excluded():
    rows = make_rows("ferrari_812", "v1", 5, label="idle")
    rows.extend([ManifestRow("x.wav", "ferrari_812", "v2", 0.0, None)])
    res = split_by_video(rows, (0.7, 0.15, 0.15), seed=0)
    all_kept = res.train + res.val + res.test
    assert all(r.label is not None for r in all_kept)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        split_by_video([], (0.6, 0.2, 0.1), seed=0)
