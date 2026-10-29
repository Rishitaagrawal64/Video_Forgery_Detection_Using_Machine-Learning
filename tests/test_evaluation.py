import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.services.evaluation import compute_metrics, roc_auc

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_and_inverted_auc():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_ties_and_single_class():
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert roc_auc(np.array([1, 1]), np.array([0.2, 0.9])) is None


def test_compute_metrics_counts():
    m = compute_metrics([0, 0, 1, 1, 1], [0.1, 0.6, 0.4, 0.9, 0.8], threshold=0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == 0.6
    assert m["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert m["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert m["majority_class_accuracy"] == 0.6


def test_compute_metrics_validates_input():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5])


def test_evaluate_script_runs_end_to_end(tmp_path, sample_video):
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    for d in ("real", "fake"):
        (tmp_path / d / "a.avi").write_bytes(Path(sample_video).read_bytes())
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "scripts/evaluate_pipeline.py", "--data", str(tmp_path), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["metrics"]["n"] == 2
    assert report["failed_videos"] == []
