import yaml

from carzam.data.regions import write_regions, Region
from carzam.data.web_reviewer import build_queue


def test_build_queue_marks_reviewed(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        yaml.safe_dump(
            {
                "ferrari_812": [
                    "https://www.youtube.com/watch?v=abc12345xyz",
                    "https://youtu.be/zzz9876aaaa",
                ],
                "porsche_gt3": ["https://www.youtube.com/watch?v=qrs7777nnn"],
            }
        )
    )
    regions_dir = tmp_path / "regions"
    # mark abc12345xyz as already reviewed
    (regions_dir / "ferrari_812").mkdir(parents=True)
    write_regions(
        regions_dir / "ferrari_812" / "abc12345xyz.yaml",
        video_id="abc12345xyz",
        duration=120.0,
        regions=[Region(0.0, 30.0, "idle")],
    )

    q = build_queue(sources_yaml, regions_dir)
    assert len(q) == 3
    by_id = {v["video_id"]: v for v in q}
    assert by_id["abc12345xyz"]["reviewed"] is True
    assert by_id["zzz9876aaaa"]["reviewed"] is False
    assert by_id["qrs7777nnn"]["car"] == "porsche_gt3"


def test_build_queue_empty_regions_not_reviewed(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        yaml.safe_dump({"ferrari_812": ["https://youtu.be/abcdef12345"]})
    )
    regions_dir = tmp_path / "regions"
    (regions_dir / "ferrari_812").mkdir(parents=True)
    # write a regions file with NO ranges -> should still count as not-reviewed
    write_regions(
        regions_dir / "ferrari_812" / "abcdef12345.yaml",
        video_id="abcdef12345",
        duration=60.0,
        regions=[],
    )
    q = build_queue(sources_yaml, regions_dir)
    assert q[0]["reviewed"] is False
