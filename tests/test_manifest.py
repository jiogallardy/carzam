import pytest

from carzam.data.manifest import (
    LABELS,
    ManifestRow,
    append_row,
    read_manifest,
    write_manifest,
)


def test_labels_constant():
    assert set(LABELS) == {"idle", "accel", "decel"}


def test_write_then_read_roundtrip(tmp_path):
    rows = [
        ManifestRow(
            path="data/windows/ferrari_812/abc_47.5.wav",
            car="ferrari_812",
            source_video="abc",
            start_time=47.5,
            label="accel",
        ),
        ManifestRow(
            path="data/windows/porsche_gt3/xyz_12.0.wav",
            car="porsche_gt3",
            source_video="xyz",
            start_time=12.0,
            label=None,
        ),
    ]
    csv_path = tmp_path / "manifest.csv"
    write_manifest(csv_path, rows)
    loaded = read_manifest(csv_path)
    assert len(loaded) == 2
    assert loaded[0].label == "accel"
    assert loaded[1].label is None
    assert loaded[1].start_time == 12.0


def test_append_row_creates_file(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    row = ManifestRow(
        path="data/windows/x.wav",
        car="other",
        source_video="x",
        start_time=0.0,
        label="idle",
    )
    append_row(csv_path, row)
    loaded = read_manifest(csv_path)
    assert len(loaded) == 1
    assert loaded[0].car == "other"


def test_append_row_extends_file(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    write_manifest(
        csv_path,
        [ManifestRow("a.wav", "ferrari_812", "v1", 0.0, None)],
    )
    append_row(csv_path, ManifestRow("b.wav", "ferrari_812", "v1", 5.0, "idle"))
    loaded = read_manifest(csv_path)
    assert len(loaded) == 2


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        ManifestRow("a.wav", "ferrari_812", "v1", 0.0, "bogus")
