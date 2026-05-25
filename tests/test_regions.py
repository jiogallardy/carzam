import pytest

from carzam.data.regions import Region, label_for_time, read_regions, write_regions


def test_write_then_read_roundtrip(tmp_path):
    regions = [
        Region(start=0.0, end=10.0, label="skip"),
        Region(start=10.0, end=45.5, label="idle"),
        Region(start=45.5, end=120.0, label="accel"),
    ]
    p = tmp_path / "v1.yaml"
    write_regions(p, video_id="v1", duration=200.0, regions=regions)
    loaded_id, loaded_dur, loaded = read_regions(p)
    assert loaded_id == "v1"
    assert loaded_dur == 200.0
    assert len(loaded) == 3
    assert loaded[1].label == "idle"
    assert loaded[2].start == 45.5


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        Region(0.0, 5.0, "bogus")


def test_invalid_range_rejected():
    with pytest.raises(ValueError):
        Region(10.0, 5.0, "idle")


def test_label_for_time_inside_range():
    regions = [
        Region(0.0, 10.0, "idle"),
        Region(10.0, 30.0, "accel"),
    ]
    assert label_for_time(regions, 5.0) == "idle"
    assert label_for_time(regions, 15.0) == "accel"


def test_label_for_time_outside_range():
    regions = [Region(10.0, 20.0, "idle")]
    assert label_for_time(regions, 5.0) is None
    assert label_for_time(regions, 25.0) is None


def test_label_for_time_window_must_fit():
    """A window starting at t with length 5s must fit ENTIRELY in one range."""
    regions = [
        Region(0.0, 12.0, "accel"),
        Region(12.0, 30.0, "idle"),
    ]
    # 5s window starting at 8 ends at 13, straddles boundary -> None
    assert label_for_time(regions, 8.0, window_seconds=5.0) is None
    # 5s window starting at 5 ends at 10, fits in accel
    assert label_for_time(regions, 5.0, window_seconds=5.0) == "accel"
    # 5s window starting at 13 ends at 18, fits in idle
    assert label_for_time(regions, 13.0, window_seconds=5.0) == "idle"
