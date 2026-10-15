import pytest

torch = pytest.importorskip("torch")
from presnap.model import CoverageModel  # noqa: E402


def test_outputs_are_causal_in_time():
    torch.manual_seed(0)
    m = CoverageModel(n_feat=15, n_sit=19, d=32, t_max=6).eval()
    X, M, S = torch.randn(2, 6, 5, 15), torch.ones(2, 6, 5, dtype=torch.bool), torch.randn(2, 19)
    a = m(X, M, S)
    X2 = X.clone(); X2[:, 3:] = torch.randn_like(X2[:, 3:])  # change only frames 3..5
    b = m(X2, M, S)
    assert a["cov"].shape == (2, 6, 7)
    assert torch.allclose(a["cov"][:, :3], b["cov"][:, :3], atol=1e-5)   # past unaffected
    assert not torch.allclose(a["cov"][:, 3:], b["cov"][:, 3:])


def test_padded_players_are_ignored():
    torch.manual_seed(0)
    m = CoverageModel(n_feat=15, n_sit=19, d=32, t_max=4).eval()
    X, S = torch.randn(1, 4, 6, 15), torch.randn(1, 19)
    M = torch.ones(1, 4, 6, dtype=torch.bool); M[..., 5] = False
    X2 = X.clone(); X2[..., 5, :] = 99.0  # garbage in the padded slot
    assert torch.allclose(m(X, M, S)["man"], m(X2, M, S)["man"], atol=1e-5)
