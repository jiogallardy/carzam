from carzam.data.download import (
    is_already_downloaded,
    target_path_for,
    video_id_from_url,
)


def test_video_id_watch_url():
    assert video_id_from_url("https://www.youtube.com/watch?v=abc123XYZ_-") == "abc123XYZ_-"


def test_video_id_short_url():
    assert video_id_from_url("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_target_path_uses_video_id(tmp_path):
    out = target_path_for(tmp_path, "ferrari_812", "https://youtube.com/watch?v=abc123XYZ_-")
    assert out.parent == tmp_path / "ferrari_812"
    assert out.name == "abc123XYZ_-.wav"


def test_idempotency_check(tmp_path):
    car_dir = tmp_path / "ferrari_812"
    car_dir.mkdir()
    (car_dir / "abc.wav").write_bytes(b"x")
    assert is_already_downloaded(tmp_path, "ferrari_812", "https://youtu.be/abc")
    assert not is_already_downloaded(tmp_path, "ferrari_812", "https://youtu.be/zzz")
