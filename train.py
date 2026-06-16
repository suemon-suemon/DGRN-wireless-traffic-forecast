from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dgrn.data import build_dataloaders, load_matrix
from dgrn.model import DGRN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DGRN training loop")
    parser.add_argument("--config", default="configs/milan.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    pred_np = np.maximum(pred_np, 0.1)
    mae = float(np.mean(np.abs(pred_np - target_np)))
    rmse = float(math.sqrt(np.mean((pred_np - target_np) ** 2)))
    ss_res = float(np.sum((target_np - pred_np) ** 2))
    ss_tot = float(np.sum((target_np - target_np.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


@torch.no_grad()
def evaluate(
    model: DGRN,
    loader,
    scaler,
    device: str,
    include_history_in_metric: bool = True,
) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    total_count = 0
    for x, mask, time_fea, y in loader:
        x = x.to(device)
        mask = mask.to(device)
        time_fea = time_fea.to(device)
        y = y.to(device)
        pred = model(x, mask, time_fea)
        if include_history_in_metric:
            pred_for_metric = torch.cat([x * mask, pred], dim=-1)
            target_for_metric = torch.cat([x, y], dim=-1)
        else:
            pred_for_metric = pred
            target_for_metric = y
        total_loss += F.l1_loss(pred_for_metric, target_for_metric, reduction="sum").item()
        total_count += target_for_metric.numel()
        preds.append(scaler.inverse_transform_tensor(pred_for_metric).cpu())
        targets.append(scaler.inverse_transform_tensor(target_for_metric).cpu())
    out = metrics(torch.cat(preds), torch.cat(targets))
    out["_scaled_l1"] = total_loss / total_count
    return out


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    cfg_path = project_dir / args.config
    cfg = load_config(cfg_path)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    output_dir = project_dir / args.output_dir / cfg["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, scaler, data_shape = build_dataloaders(cfg, project_dir)
    physical_adj = load_matrix(project_dir / cfg["physical_adj_csv"])
    latent_init = None
    if cfg.get("latent_init_csv"):
        latent_path = project_dir / cfg["latent_init_csv"]
        if latent_path.exists():
            latent_init = load_matrix(latent_path)

    model = DGRN(
        num_nodes=int(cfg["num_nodes"]),
        close_len=int(cfg["close_len"]),
        pred_len=int(cfg["pred_len"]),
        physical_adj=physical_adj,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        time_input_dim=int(cfg.get("time_input_dim", cfg.get("time_dim", 9))),
        temporal_embed_dim=int(cfg.get("temporal_embed_dim", 64)),
        k_in=int(cfg.get("k_in", 3)),
        k_rec=int(cfg.get("k_rec", 3)),
        num_blocks=int(cfg.get("num_blocks", 2)),
        latent_init=latent_init,
        entmax_alpha=float(cfg.get("entmax_alpha", 1.5)),
        graph_l1=float(cfg.get("graph_l1", 0.0)),
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=int(cfg.get("lr_patience", 4)), factor=0.5
    )

    best_val = float("inf")
    best_state = None
    patience = int(cfg.get("patience", 8))
    patience_left = patience
    start_time = time.time()
    print(
        f"Dataset={cfg['dataset']} shape={data_shape} train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)} device={args.device}"
    )
    print(
        f"Scaler fit on train: min={scaler.data_min:.6g}, "
        f"max={scaler.data_max:.6g}, range={scaler.data_range:.6g}"
    )

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for x, mask, time_fea, y in train_loader:
            x = x.to(args.device)
            mask = mask.to(args.device)
            time_fea = time_fea.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, mask, time_fea)
            loss = F.l1_loss(pred, y) + model.regularization_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 5.0)))
            optimizer.step()
            epoch_loss += loss.item() * y.numel()
            epoch_count += y.numel()

        val = evaluate(
            model,
            val_loader,
            scaler,
            args.device,
            include_history_in_metric=bool(cfg.get("include_history_in_metric", True)),
        )
        scheduler.step(val["_scaled_l1"])
        print(
            f"epoch={epoch:03d} "
            f"val_MAE={val['MAE']:.4f} val_RMSE={val['RMSE']:.4f} val_R2={val['R2']:.4f}"
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

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(
        model,
        test_loader,
        scaler,
        args.device,
        include_history_in_metric=bool(cfg.get("include_history_in_metric", True)),
    )
    test.pop("_scaled_l1", None)
    print("METRIC_JSON:", json.dumps(test, sort_keys=True))

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    model.save_latent_graph(output_dir / "learned_adj.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, sort_keys=True)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
