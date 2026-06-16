from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from entmax import entmax_bisect


def normalize_adj_add_self_loop(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.float()
    adj = adj + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
    degree = adj.sum(dim=1).clamp_min(1e-6)
    d_inv_sqrt = degree.pow(-0.5)
    return adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def cheb_filter(x: torch.Tensor, adj: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    # x: [B, Fin, N], adj: [N, N], weight: [Fout, K, Fin]
    supports = [x]
    if weight.size(1) > 1:
        supports.append(torch.einsum("ij,bfj->bfi", adj, x))
    for _ in range(2, weight.size(1)):
        supports.append(2 * torch.einsum("ij,bfj->bfi", adj, supports[-1]) - supports[-2])
    stacked = torch.stack(supports[: weight.size(1)], dim=2)
    out = torch.einsum("bfkn,okf->bon", stacked, weight)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out


class DGRNCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_input_dim: int,
        temporal_embed_dim: int,
        k_in: int,
        k_rec: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_in = k_in
        self.k_rec = k_rec
        self.weight_in = nn.Parameter(torch.empty(hidden_dim, k_in, input_dim))
        self.weight_rec = nn.Parameter(torch.empty(hidden_dim, k_rec, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.time_encoder = nn.Sequential(
            nn.Linear(time_input_dim, temporal_embed_dim),
            nn.ReLU(),
        )
        self.gamma_in = nn.Linear(temporal_embed_dim, hidden_dim)
        self.gamma_rec = nn.Linear(temporal_embed_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound_in = 1.0 / (self.weight_in.size(2) * self.k_in) ** 0.5
        self.weight_in.data.uniform_(-bound_in, bound_in)
        with torch.no_grad():
            self.weight_rec.zero_()
            eye_dim = min(self.weight_rec.size(0), self.weight_rec.size(2))
            self.weight_rec[:eye_dim, 0, :eye_dim] = torch.eye(eye_dim)
            if self.k_rec > 1:
                self.weight_rec[:, 1:, :].uniform_(-1e-3, 1e-3)
            self.gamma_in.bias.fill_(1.0)
            self.gamma_rec.bias.fill_(1.0)

    def forward(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor,
        adj_phy: torch.Tensor,
        adj_latent: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        h = h0
        states = []
        for step in range(x_seq.size(1)):
            x_t = x_seq[:, step]
            t_t = time_features[:, step]
            term_in = cheb_filter(x_t, adj_phy, self.weight_in)
            term_rec = cheb_filter(h, adj_latent, self.weight_rec)
            time_embed = self.time_encoder(t_t)
            gamma_in = torch.sigmoid(self.gamma_in(time_embed)).unsqueeze(-1)
            gamma_rec = torch.sigmoid(self.gamma_rec(time_embed)).unsqueeze(-1)
            term_in = gamma_in * term_in
            term_rec = gamma_rec * term_rec
            h = torch.tanh(term_in + term_rec + self.bias.view(1, -1, 1))
            states.append(h.unsqueeze(1))
        return torch.cat(states, dim=1)


class DGRNBlock(nn.Module):
    def __init__(
        self,
        close_len: int,
        pred_len: int,
        num_nodes: int,
        hidden_dim: int,
        time_input_dim: int,
        temporal_embed_dim: int,
        k_in: int,
        k_rec: int,
    ):
        super().__init__()
        self.cell = DGRNCell(1, hidden_dim, time_input_dim, temporal_embed_dim, k_in, k_rec)
        self.hidden_dim = hidden_dim
        self.forecast = nn.Sequential(nn.Linear(hidden_dim, pred_len), nn.ReLU())
        self.backcast = nn.Sequential(nn.Linear(hidden_dim, close_len), nn.ReLU())

    def forward(
        self,
        x: torch.Tensor,
        adj_phy: torch.Tensor,
        adj_latent: torch.Tensor,
        time_features: torch.Tensor,
    ):
        batch_size, num_nodes, _ = x.shape
        x_seq = x.permute(0, 2, 1).unsqueeze(2)
        h0 = torch.zeros(batch_size, self.hidden_dim, num_nodes, device=x.device)
        states = self.cell(x_seq, h0, adj_phy, adj_latent, time_features)
        h_last = states[:, -1].permute(0, 2, 1)
        return self.forecast(h_last), self.backcast(h_last)


class DGRN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        close_len: int,
        pred_len: int,
        physical_adj: np.ndarray,
        hidden_dim: int = 64,
        time_input_dim: int = 9,
        temporal_embed_dim: int = 64,
        k_in: int = 3,
        k_rec: int = 3,
        num_blocks: int = 2,
        latent_init: np.ndarray | None = None,
        entmax_alpha: float = 1.5,
        graph_l1: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.close_len = close_len
        self.pred_len = pred_len
        self.entmax_alpha = entmax_alpha
        self.graph_l1 = graph_l1
        self.register_buffer(
            "adj_phy", normalize_adj_add_self_loop(torch.from_numpy(physical_adj).float())
        )

        edge_count = num_nodes * (num_nodes - 1) // 2
        if latent_init is None:
            logits = torch.empty(edge_count)
            nn.init.xavier_uniform_(logits.unsqueeze(0))
            logits = logits.squeeze(0)
        else:
            values = []
            clipped = np.clip(latent_init.astype(np.float32), 1e-4, 1 - 1e-4)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    values.append(clipped[i, j])
            logits = torch.logit(torch.tensor(values, dtype=torch.float32))
        self.edge_logits = nn.Parameter(logits)

        self.blocks = nn.ModuleList(
            [
                DGRNBlock(
                    close_len,
                    pred_len,
                    num_nodes,
                    hidden_dim,
                    time_input_dim,
                    temporal_embed_dim,
                    k_in,
                    k_rec,
                )
                for _ in range(num_blocks)
            ]
        )
    def edge_weights(self) -> torch.Tensor:
        logits = torch.nan_to_num(self.edge_logits, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            -30.0, 30.0
        )
        return entmax_bisect(logits.unsqueeze(0), alpha=self.entmax_alpha, dim=-1).squeeze(0)

    def latent_adj(self) -> torch.Tensor:
        adj = torch.zeros(
            self.num_nodes,
            self.num_nodes,
            dtype=self.edge_logits.dtype,
            device=self.edge_logits.device,
        )
        triu = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=adj.device)
        weights = self.edge_weights()
        adj[triu[0], triu[1]] = weights
        adj = adj + adj.t()
        return normalize_adj_add_self_loop(adj)

    def regularization_loss(self) -> torch.Tensor:
        if self.graph_l1 <= 0:
            return self.edge_logits.new_tensor(0.0)
        return self.graph_l1 * self.edge_weights().mean()

    def model_parameters(self):
        return [p for name, p in self.named_parameters() if name != "edge_logits"]

    def graph_parameters(self):
        return [self.edge_logits]

    @torch.no_grad()
    def project_recurrent_taps(self, delta: float = 0.05) -> None:
        # Enforce C_H < 1 through aggregate spectral projection of the recurrent
        # Chebyshev taps after inner updates.
        c_max = 1.0 - delta
        for block in self.blocks:
            weight = block.cell.weight_rec
            k_rec = weight.size(1)
            # Because FiLM gains use sigmoid, K_gamma,r <= 1. The threshold
            # matches tau_r=(1-delta)/(K_gamma,r*sqrt(K_r)).
            tau = c_max / (k_rec ** 0.5)
            tap_norms = []
            for tap_idx in range(k_rec):
                tap = weight[:, tap_idx, :]
                tap_norms.append(torch.linalg.matrix_norm(tap, ord=2))
            aggregate = torch.sqrt(sum(norm.square() for norm in tap_norms))
            if aggregate > tau:
                weight.mul_(tau / aggregate.clamp_min(1e-8))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        residual = x * mask
        total_forecast = 0.0
        adj_latent = self.latent_adj()
        for block in self.blocks:
            forecast, backcast = block(residual, self.adj_phy, adj_latent, time_features)
            residual = residual - backcast
            total_forecast = total_forecast + forecast
        return total_forecast

    def save_latent_graph(self, path: str | Path) -> None:
        adj = self.latent_adj().detach().cpu().numpy()
        np.savetxt(path, adj, delimiter=",")
