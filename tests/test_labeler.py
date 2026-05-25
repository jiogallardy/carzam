from carzam.data.labeler import unlabeled_rows
from carzam.data.manifest import ManifestRow


def test_filters_to_unlabeled():
    rows = [
        ManifestRow("a.wav", "ferrari_812", "v1", 0.0, "idle"),
        ManifestRow("b.wav", "ferrari_812", "v1", 2.5, None),
        ManifestRow("c.wav", "porsche_gt3", "v2", 0.0, None),
        ManifestRow("d.wav", "porsche_gt3", "v2", 5.0, "accel"),
    ]
    out = unlabeled_rows(rows)
    assert len(out) == 2
    assert {r.path for r in out} == {"b.wav", "c.wav"}
