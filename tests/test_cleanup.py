import os
import time

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.cleanup import purge_stale_files


def _age(path, hours):
    t = time.time() - hours * 3600
    os.utime(path, (t, t))


def test_purges_only_old_entries(tmp_path):
    old, new = tmp_path / "old.mp4", tmp_path / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    old_dir = tmp_path / "frames_old"
    old_dir.mkdir()
    (old_dir / "0.png").write_bytes(b"x")
    _age(old, 48)
    _age(old_dir, 48)
    assert purge_stale_files(tmp_path, 24) == 2
    assert not old.exists() and not old_dir.exists() and new.exists()


def test_keeps_dotfiles_and_can_be_disabled(tmp_path):
    keep = tmp_path / ".gitkeep"
    keep.write_bytes(b"")
    _age(keep, 999)
    stale = tmp_path / "a.mp4"
    stale.write_bytes(b"x")
    _age(stale, 999)
    assert purge_stale_files(tmp_path, 0) == 0 and stale.exists()
    assert purge_stale_files(tmp_path, 1) == 1 and keep.exists()


def test_missing_directory_is_ignored(tmp_path):
    assert purge_stale_files(tmp_path / "nope", 1) == 0


def test_startup_purges_stale_uploads(monkeypatch):
    stale = settings.upload_dir / "leftover.mp4"
    stale.write_bytes(b"x")
    _age(stale, 100)
    monkeypatch.setattr(settings, "stale_file_ttl_hours", 24)
    with TestClient(app):
        pass
    assert not stale.exists()


def test_one_undeletable_entry_does_not_stop_the_purge(tmp_path, monkeypatch):
    from pathlib import Path

    stuck, ok = tmp_path / "stuck.mp4", tmp_path / "ok.mp4"
    for f in (stuck, ok):
        f.write_bytes(b"x")
        _age(f, 100)
    real_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if self.name == "stuck.mp4":
            raise PermissionError("in use")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    assert purge_stale_files(tmp_path, 24) == 1
    assert stuck.exists() and not ok.exists()
