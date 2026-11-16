"""The train/val/test split must be leak-free: frames of one video may never land in two splits."""
import pytest

from ml.split import scan_dataset, split_by_video, summarize


@pytest.fixture
def dataset(tmp_path):
    """20 fake and 10 real videos, 3 frames each (named <video>_<frame>.png), plus an ignored folder."""
    for cls, n_videos in (("fake", 20), ("real", 10)):
        d = tmp_path / cls
        d.mkdir()
        for v in range(n_videos):
            for frame in range(3):
                (d / f"vid{v:02d}_{frame}.png").write_bytes(b"")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x_0.png").write_bytes(b"")
    (tmp_path / "real" / "notes.txt").write_text("not an image")
    return tmp_path


def test_scan_labels_and_video_ids(dataset):
    items = scan_dataset(dataset)
    assert len(items) == (20 + 10) * 3
    labels = {vid: label for _, label, vid in items}
    assert labels["fake/vid00"] == 1 and labels["real/vid00"] == 0
    assert not any(vid.startswith("other/") for vid in labels)


def test_same_video_id_in_both_classes_is_kept_apart(dataset):
    vids = {vid for _, _, vid in scan_dataset(dataset)}
    assert {"fake/vid00", "real/vid00"} <= vids


def test_scan_empty_dataset_raises(tmp_path):
    (tmp_path / "real").mkdir()
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path)


def test_no_video_in_two_splits_and_nothing_lost(dataset):
    items = scan_dataset(dataset)
    train, val, test = split_by_video(items)
    assert len(train) + len(val) + len(test) == len(items)
    split_videos = [{v for _, _, v in rows} for rows in (train, val, test)]
    assert not (split_videos[0] & split_videos[1])
    assert not (split_videos[0] & split_videos[2])
    assert not (split_videos[1] & split_videos[2])


def test_all_frames_of_a_video_stay_together(dataset):
    train, val, test = split_by_video(scan_dataset(dataset))
    for rows in (train, val, test):
        counts = {}
        for _, _, vid in rows:
            counts[vid] = counts.get(vid, 0) + 1
        assert set(counts.values()) <= {3}


def test_both_classes_appear_in_every_split(dataset):
    for rows in split_by_video(scan_dataset(dataset)):
        assert {label for _, label, _ in rows} == {0, 1}


def test_split_sizes_follow_fractions(dataset):
    train, val, test = split_by_video(scan_dataset(dataset), val_frac=0.2, test_frac=0.2)
    n_videos = lambda rows: len({v for _, _, v in rows})  # noqa: E731
    assert (n_videos(train), n_videos(val), n_videos(test)) == (18, 6, 6)


def test_split_is_deterministic_per_seed(dataset):
    items = scan_dataset(dataset)
    first = split_by_video(items, seed=7)
    assert first == split_by_video(items, seed=7)
    assert first != split_by_video(items, seed=8)


def test_summarize_reports_counts(dataset):
    train, _, _ = split_by_video(scan_dataset(dataset))
    text = summarize("train", train)
    assert text.startswith("train: ") and "frames" in text and "videos" in text
