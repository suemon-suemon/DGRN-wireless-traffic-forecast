from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from entmax import entmax_bisect
torch.sparse.check_sparse_tensor_invariants.disable()


def normalize_adj_add_self_loop(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.float()
    adj = adj + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
    degree = adj.sum(dim=1).clamp_min(1e-6)
    d_inv_sqrt = degree.pow(-0.5)
    return adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def normalize_adj_sparse_add_self_loop(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops and symmetrically normalize a sparse adjacency matrix."""
    num_nodes = adj.size(0)
    device = adj.device
    indices = adj.coalesce().indices()
    values = adj.coalesce().values()
    self_indices = torch.arange(num_nodes, device=device)
    all_indices = torch.cat(
        [indices, torch.stack([self_indices, self_indices])], dim=1
    )
    all_values = torch.cat([values, torch.ones(num_nodes, device=device, dtype=values.dtype)])
    adj = torch.sparse_coo_tensor(
        all_indices, all_values, (num_nodes, num_nodes), device=device
    ).coalesce()
    degree = torch.zeros(num_nodes, device=device, dtype=adj.dtype).scatter_add_(
        0, adj.indices()[0], adj.values()
    )
    d_inv_sqrt = degree.clamp_min(1e-6).pow(-0.5)
    normalized_values = adj.values() * d_inv_sqrt[adj.indices()[0]] * d_inv_sqrt[adj.indices()[1]]
    return torch.sparse_coo_tensor(
        adj.indices(), normalized_values, adj.shape, device=device
    ).coalesce()


def graph_multiply(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply an [N, N] dense or sparse graph shift to x shaped [B, F, N]."""
    if not adj.is_sparse:
        return torch.einsum("ij,bfj->bfi", adj, x)
    num_nodes = x.size(-1)
    flattened = x.permute(2, 0, 1).reshape(num_nodes, -1)
    return torch.sparse.mm(adj, flattened).reshape(num_nodes, *x.shape[:2]).permute(1, 2, 0)


def cheb_filter(x: torch.Tensor, adj: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    # x: [B, Fin, N], adj: [N, N], weight: [Fout, K, Fin]
    supports = [x]
    if weight.size(1) > 1:
        supports.append(graph_multiply(adj, x))
    for _ in range(2, weight.size(1)):
        supports.append(2 * graph_multiply(adj, supports[-1]) - supports[-2])
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
        # self.bias = nn.Parameter(torch.zeros(hidden_dim))
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
            h = torch.tanh(term_in + term_rec)
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
        latent_graph: str = "dense",
        topk_k: int = 12,
        graph_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.close_len = close_len
        self.pred_len = pred_len
        self.entmax_alpha = entmax_alpha
        self.graph_l1 = graph_l1
        if latent_graph not in {"dense", "topk"}:
            raise ValueError(f"latent_graph must be 'dense' or 'topk', got {latent_graph!r}")
        if topk_k < 1:
            raise ValueError("topk_k must be positive")
        if graph_embed_dim < 1:
            raise ValueError("graph_embed_dim must be positive")
        self.latent_graph = latent_graph
        self.topk_k = topk_k
        self.graph_embed_dim = graph_embed_dim
        self._cached_inference_adj: torch.Tensor | None = None
        self.register_buffer(
            "adj_phy",
            normalize_adj_add_self_loop(torch.from_numpy(physical_adj).float())
            if latent_graph == "dense"
            else normalize_adj_sparse_add_self_loop(torch.from_numpy(physical_adj).float().to_sparse_coo()),
        )

        edge_count = num_nodes * (num_nodes - 1) // 2
        if latent_graph == "topk":
            logits = torch.randn(num_nodes, 2 * graph_embed_dim) * 0.1
        elif latent_init is None:
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
        if self.latent_graph == "topk":
            raise RuntimeError("edge_weights is only defined for the dense latent graph")
        logits = torch.nan_to_num(self.edge_logits, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            -30.0, 30.0
        )
        return entmax_bisect(logits.unsqueeze(0), alpha=self.entmax_alpha, dim=-1).squeeze(0)

    def latent_adj(self) -> torch.Tensor:
        if self.latent_graph == "topk":
            if not self.training and self._cached_inference_adj is not None:
                return self._cached_inference_adj
            adj = self._topk_latent_adj()
            if not self.training:
                self._cached_inference_adj = adj.detach()
                return self._cached_inference_adj
            return adj
        adj = torch.zeros(
            self.num_nodes,
            self.num_nodes,
            dtype=self.edge_logits.dtype,
            device=self.edge_logits.device,
        )
        triu = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=adj.device)
        weights = self.edge_weights()
        adj[triu[0], triu[1]] = weights
        adj = (adj + adj.t()) * self.num_nodes * 0.5
        return normalize_adj_add_self_loop(adj)

    def _topk_latent_adj(self) -> torch.Tensor:
        num_nodes = self.num_nodes
        k = min(self.topk_k, num_nodes - 1)
        if k == 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=self.edge_logits.device)
            empty_values = torch.empty(0, dtype=self.edge_logits.dtype, device=self.edge_logits.device)
            return normalize_adj_sparse_add_self_loop(
                torch.sparse_coo_tensor(empty_indices, empty_values, (num_nodes, num_nodes)).coalesce()
            )

        first = self.edge_logits[:, : self.graph_embed_dim]
        second = self.edge_logits[:, self.graph_embed_dim :]
        with torch.no_grad():
            scores = first @ second.t() + second @ first.t()
            scores.fill_diagonal_(float("-inf"))
            neighbor_positions = torch.topk(scores, k, dim=-1).indices
            rows = torch.arange(num_nodes, device=scores.device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = neighbor_positions.reshape(-1)
            linear_indices = torch.unique(torch.cat([rows * num_nodes + cols, cols * num_nodes + rows]))
            row_indices = torch.div(linear_indices, num_nodes, rounding_mode="floor")
            col_indices = linear_indices % num_nodes

        values = torch.sigmoid(
            (first[row_indices] * second[col_indices]).sum(-1)
            + (first[col_indices] * second[row_indices]).sum(-1)
        )
        adj = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]), values, (num_nodes, num_nodes), device=values.device
        ).coalesce()
        return normalize_adj_sparse_add_self_loop(adj)

    def train(self, mode: bool = True):
        if mode:
            self._cached_inference_adj = None
        return super().train(mode)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        self._cached_inference_adj = None
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def regularization_loss(self) -> torch.Tensor:
        if self.latent_graph == "topk" or self.graph_l1 <= 0:
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

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        residual = x
        total_forecast = 0.0
        adj_latent = self.latent_adj()
        for block in self.blocks:
            forecast, backcast = block(residual, self.adj_phy, adj_latent, time_features)
            residual = residual - backcast
            total_forecast = total_forecast + forecast
        return total_forecast

    def save_latent_graph(self, path: str | Path) -> None:
        adj = self.latent_adj().detach()
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = adj.cpu().numpy()
        np.savetxt(path, adj, delimiter=",")
