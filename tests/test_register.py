import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from presnap.video import register as R  # noqa: E402


def test_sideline_pairs_use_the_vertical_midline():
    # two flat lines at y = 200 and y = 800 in a 1920x1080 frame, plus a stands railing at y = 1000
    flat = [np.array([0.0, 1.0, -200.0]), np.array([0.0, 1.0, -800.0]), np.array([0.0, 1.0, -1000.0])]
    pairs = list(R._pairs(([], flat, 1080, 1920)))
    assert len(pairs) == 2  # (200, 800) and (200, 1000); (800, 1000) is too close
    assert all(abs(R._cross(a, np.array([1.0, 0, -960]))[1] - 200) < 1e-6 for a, _ in pairs)
