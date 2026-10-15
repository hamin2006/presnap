import numpy as np
import pytest


from presnap.augment import drift, swap, velocities  # noqa: E402


def test_velocity_from_linear_motion():
    T = 6
    X = np.zeros((1, T, 1, 15), np.float32)
    X[0, :, 0, 0] = np.arange(T) * 0.5 / 10       # 0.5 yd per frame = 5 yd/s downfield
    M = np.ones((1, T, 1), bool)
    v = velocities(X, M)[0, :, 0, 2] * 5
    assert np.allclose(v, 5.0)


def test_drift_zero_is_identity_and_swap_keeps_positions_set():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 5, 3, 15)).astype(np.float32)
    M = np.ones((2, 5, 3), bool)
    assert drift(X, M, 0, rng) is X
    S = swap(X, M, 1.0, rng)
    # an identity switch moves positions between slots but never invents new ones
    assert np.allclose(np.sort(S[..., 0], axis=-1), np.sort(X[..., 0], axis=-1))
