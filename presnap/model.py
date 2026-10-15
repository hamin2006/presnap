"""Coverage model (Phase 1C): a transformer over players within each frame, then a causal transformer over
frames, so the output at frame t uses only frames 0..t. One forward pass gives predictions at every time step.

Our own implementation; the per-frame player encoder follows the idea in SumerSports' SportsTrackingTransformer
(that repo has no license, so no code is copied).
"""
import torch
from torch import nn

HEADS = {"cov": 7, "man": 2, "shell": 3, "blitz": 2}


class CoverageModel(nn.Module):
    def __init__(self, n_feat, n_sit, d=64, heads=4, player_layers=2, time_layers=2, dropout=0.1, t_max=41):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.ReLU(), nn.LayerNorm(d))
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.players = nn.TransformerEncoder(layer(), player_layers, enable_nested_tensor=False)
        self.sit = nn.Linear(n_sit, d)
        self.time_pos = nn.Embedding(t_max, d)
        self.frame_proj = nn.Linear(2 * d, d)
        self.time = nn.TransformerEncoder(layer(), time_layers, enable_nested_tensor=False)
        self.out = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n)) for k, n in HEADS.items()})

    def forward(self, X, M, S):
        """X [B,T,P,F], M [B,T,P] player present, S [B,n_sit] -> {head: logits [B,T,n]}"""
        B, T, P, _ = X.shape
        h = self.embed(X.view(B * T, P, -1))
        m = M.view(B * T, P)
        empty = ~m.any(1)
        m = m.clone()
        m[empty, 0] = True  # frames past the throw are fully padded; keep attention defined
        h = self.players(h, src_key_padding_mask=~m)
        w = m.unsqueeze(-1).float()
        is_def = X.view(B * T, P, -1)[..., 4:5] * w
        mean_all = (h * w).sum(1) / w.sum(1).clamp(min=1)
        mean_def = (h * is_def).sum(1) / is_def.sum(1).clamp(min=1)
        f = self.frame_proj(torch.cat([mean_all, mean_def], -1)).view(B, T, -1)
        f = f + self.time_pos(torch.arange(T, device=X.device)) + self.sit(S).unsqueeze(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=X.device), 1)
        f = self.time(f, mask=causal)
        return {k: head(f) for k, head in self.out.items()}
