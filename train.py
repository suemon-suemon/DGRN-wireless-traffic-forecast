from __future__ import annotations

import argparse
import copy
import json
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dgrn.data import build_dataloaders, load_matrix
from dgrn.model import DGRN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bilevel DGRN training loop")
    parser.add_argument("--config", default="configs/milan.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--latent-graph",
        choices=("dense", "topk"),
        default="dense",
        help="latent graph generator; this command-line option intentionally does not read YAML",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, y)


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_flat = pred.detach().cpu().reshape(-1).float()
    target_flat = target.detach().cpu().reshape(-1).float()
    diff = pred_flat - target_flat
    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff.square())).item()
    ss_res = diff.square().sum()
    ss_tot = (target_flat - target_flat.mean()).square().sum()
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


@torch.no_grad()
def evaluate(model: DGRN, loader, scaler, device: str) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    total_count = 0
    for x, time_fea, y in loader:
        x, time_fea, y = (
            x.to(device),
            time_fea.to(device),
            y.to(device),
        )
        pred = model(x, time_fea)
        total_loss += F.l1_loss(pred, y, reduction="sum").item()
        total_count += y.numel()
        preds.append(scaler.inverse_transform_tensor(pred).cpu())
        targets.append(scaler.inverse_transform_tensor(y).cpu())

    out = metrics(torch.cat(preds), torch.cat(targets))
    out["_scaled_l1"] = total_loss / total_count
    return out


def build_model(cfg: dict, project_dir: Path, device: str, latent_graph: str = "dense") -> DGRN:
    physical_adj = load_matrix(project_dir / cfg["physical_adj_csv"])
    latent_init = None
    if cfg.get("latent_init_csv"):
        latent_path = project_dir / cfg["latent_init_csv"]
        if latent_path.exists():
            latent_init = load_matrix(latent_path)

    return DGRN(
        num_nodes=int(cfg["num_nodes"]),
        close_len=int(cfg["close_len"]),
        pred_len=int(cfg["pred_len"]),
        physical_adj=physical_adj,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        time_input_dim=int(cfg.get("time_input_dim", cfg.get("time_dim", 9))),
        temporal_embed_dim=int(cfg.get("temporal_embed_dim", 64)),
        k_in=int(cfg.get("k_in", 3)),
        k_rec=int(cfg.get("k_rec", 3)),
        num_blocks=int(cfg.get("num_blocks", 3)),
        latent_init=latent_init,
        entmax_alpha=float(cfg.get("entmax_alpha", 1.5)),
        graph_l1=float(cfg.get("graph_l1", 0.0)),
        latent_graph=latent_graph,
    ).to(device)


def move_batch(batch, device: str):
    x, time_fea, y = batch
    return x.to(device), time_fea.to(device), y.to(device)


def collect_epoch_batches(loader, num_batches: int, device: str):
    if num_batches <= 0:
        return [move_batch(batch, device) for batch in loader]
    return [move_batch(batch, device) for batch in itertools.islice(loader, num_batches)]


def average_loss(model: DGRN, batches) -> torch.Tensor:
    losses = []
    for x, time_fea, y in batches:
        pred = model(x, time_fea)
        losses.append(training_loss(pred, y))
    return torch.stack(losses).mean()


def bilevel_outer_step(
    model: DGRN,
    inner_lr: float,
    outer_optimizer: torch.optim.Optimizer,
    train_batches,
    val_batches,
    ema_buffer: torch.Tensor | None,
    ema_beta: float,
    clip_tau: float,
    armijo_mu: float,
    max_backtracking_steps: int,
    backtracking_factor: float,
) -> tuple[torch.Tensor, bool, float]:
    model.train()
    model_params = model.model_parameters()
    graph_param = model.edge_logits

    val_loss = average_loss(model, val_batches)
    direct_grad = torch.autograd.grad(val_loss, graph_param, retain_graph=True)[0]

    # First-order DARTS hypergradient (Eq. (10) / Algorithm 1): the direct
    # validation gradient plus the one-step indirect term that accounts for the
    # dependence of the inner optimum on phi. The residual of this one-step
    # approximation is the bias constant c_FO analyzed in the supplement.
    val_model_grads = torch.autograd.grad(
        val_loss, model_params, retain_graph=True, allow_unused=True
    )
    train_loss = average_loss(model, train_batches)
    train_model_grads = torch.autograd.grad(
        train_loss, model_params, create_graph=True, allow_unused=True
    )
    grad_dot = graph_param.new_tensor(0.0)
    for train_grad, val_grad in zip(train_model_grads, val_model_grads):
        if train_grad is None or val_grad is None:
            continue
        grad_dot = grad_dot + (train_grad * val_grad.detach()).sum()
    indirect_grad = torch.autograd.grad(grad_dot, graph_param, allow_unused=True)[0]
    if indirect_grad is None:
        indirect_grad = torch.zeros_like(graph_param)
    hyper_grad = direct_grad - inner_lr * indirect_grad
    if ema_buffer is None:
        ema_buffer = torch.zeros_like(hyper_grad)
    ema_buffer = ema_beta * ema_buffer + (1.0 - ema_beta) * hyper_grad.detach()

    clipped_grad = ema_buffer.clamp(min=-clip_tau, max=clip_tau)
    old_graph = graph_param.detach().clone()
    old_state = copy.deepcopy(outer_optimizer.state_dict())
    old_lrs = [group["lr"] for group in outer_optimizer.param_groups]
    old_val_loss = val_loss.detach()
    accepted = False

    for attempt in range(max_backtracking_steps + 1):
        graph_param.data.copy_(old_graph)
        outer_optimizer.load_state_dict(old_state)
        trial_lrs = [lr * (backtracking_factor**attempt) for lr in old_lrs]
        for group, lr in zip(outer_optimizer.param_groups, trial_lrs):
            group["lr"] = lr

        outer_optimizer.zero_grad(set_to_none=True)
        graph_param.grad = clipped_grad.clone()
        outer_optimizer.step()
        graph_param.grad = None

        with torch.no_grad():
            new_val_loss = average_loss(model, val_batches)
            step_norm_sq = (graph_param.detach() - old_graph).pow(2).sum()
            sufficient_decrease = old_val_loss - armijo_mu * step_norm_sq
            if new_val_loss <= sufficient_decrease:
                accepted = True
                break

    if not accepted:
        graph_param.data.copy_(old_graph)
        outer_optimizer.load_state_dict(old_state)
        trial_lrs = [lr * backtracking_factor for lr in old_lrs]

    for group, lr in zip(outer_optimizer.param_groups, trial_lrs):
        group["lr"] = lr

    return ema_buffer.detach(), accepted, trial_lrs[0]


def run_inner_updates(
    model: DGRN,
    loader,
    optimizer: torch.optim.Optimizer,
    inner_steps: int,
    device: str,
    projection_delta: float,
    grad_clip: float,
) -> None:
    model.train()
    batches = itertools.cycle(loader)
    for _ in range(inner_steps):
        x, time_fea, y = move_batch(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, time_fea)
        loss = training_loss(pred, y) + model.regularization_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model_parameters(), grad_clip)
        optimizer.step()
        model.edge_logits.grad = None
        model.project_recurrent_taps(delta=projection_delta)


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    cfg = load_config(project_dir / args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = project_dir / args.output_dir / cfg["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, scaler, data_shape = build_dataloaders(cfg, project_dir)
    model = build_model(cfg, project_dir, args.device, latent_graph=args.latent_graph)

    inner_lr = float(cfg.get("inner_lr", cfg.get("learning_rate", 1e-3)))
    outer_lr = float(cfg.get("outer_lr", 0.1))
    inner_steps = int(cfg.get("inner_steps", 3))
    outer_train_batches = int(cfg.get("outer_train_batches", 5))
    outer_val_batches = int(cfg.get("outer_val_batches", 5))
    ema_beta = float(cfg.get("ema_beta", 0.9))
    clip_tau = float(cfg.get("outer_clip_tau", 1.0))
    projection_delta = float(cfg.get("projection_delta", 0.05))
    armijo_mu = float(cfg.get("armijo_mu", 1e-4))
    max_backtracking_steps = int(cfg.get("outer_ls_max_trials", 2))
    backtracking_factor = float(cfg.get("outer_lr_backtracking_factor", 0.8))
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError("outer_lr_backtracking_factor must be in (0, 1).")

    inner_optimizer = torch.optim.AdamW(
        model.model_parameters(),
        lr=inner_lr,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    outer_optimizer = torch.optim.AdamW(model.graph_parameters(), lr=outer_lr)

    best_val = float("inf")
    best_state = None
    patience = int(cfg.get("patience", 20))
    patience_left = patience
    ema_buffer = None

    print(
        f"Dataset={cfg['dataset']} shape={data_shape} train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)} device={args.device}"
    )
    print(
        f"Bilevel settings: inner_steps={inner_steps}, inner_lr={inner_lr}, "
        f"outer_lr={outer_lr}, ema_beta={ema_beta}, outer_train_batches={outer_train_batches}, "
        f"outer_val_batches={outer_val_batches}"
    )
    print(
        f"Scaler fit on train: min={scaler.data_min:.6g}, "
        f"max={scaler.data_max:.6g}, range={scaler.data_range:.6g}"
    )
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Seed={seed} | Model params: total={num_params:,} trainable={num_trainable:,}")

    train_start = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epochs_run = epoch
        run_inner_updates(
            model=model,
            loader=train_loader,
            optimizer=inner_optimizer,
            inner_steps=inner_steps,
            device=args.device,
            projection_delta=projection_delta,
            grad_clip=float(cfg.get("grad_clip", 5.0)),
        )

        train_batches = collect_epoch_batches(train_loader, outer_train_batches, args.device)
        val_batches = collect_epoch_batches(val_loader, outer_val_batches, args.device)
        ema_buffer, accepted, current_outer_lr = bilevel_outer_step(
            model=model,
            inner_lr=inner_lr,
            outer_optimizer=outer_optimizer,
            train_batches=train_batches,
            val_batches=val_batches,
            ema_buffer=ema_buffer,
            ema_beta=ema_beta,
            clip_tau=clip_tau,
        armijo_mu=armijo_mu,
        max_backtracking_steps=max_backtracking_steps,
        backtracking_factor=backtracking_factor,
        )

        val = evaluate(
            model,
            val_loader,
            scaler,
            args.device,
        )
        print(
            f"epoch={epoch:03d} "
            f"val_MAE={val['MAE']:.4f} val_RMSE={val['RMSE']:.4f} "
            f"val_R2={val['R2']:.4f} outer_accepted={int(accepted)} "
            f"outer_lr={current_outer_lr:.6g}"
        )

        if val["_scaled_l1"] < best_val:
            best_val = val["_scaled_l1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    train_seconds = time.perf_counter() - train_start
    print(
        f"Training finished: epochs_run={epochs_run} "
        f"total_time={train_seconds:.1f}s ({train_seconds / 60:.2f} min)"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(
        model,
        test_loader,
        scaler,
        args.device,
    )
    test.pop("_scaled_l1", None)
    test["seed"] = seed
    test["num_params"] = num_params
    test["num_trainable_params"] = num_trainable
    test["epochs_run"] = epochs_run
    test["train_seconds"] = round(train_seconds, 2)
    print("METRIC_JSON:", json.dumps(test, sort_keys=True))

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    model.save_latent_graph(output_dir / "learned_adj.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, sort_keys=True)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
