#!/usr/bin/env python3
"""Run DGRN repeatedly and aggregate the test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
METRICS = ("RMSE", "MAE", "R2")
# 10 runs for accuracy comparison in TABLE IV
DEFAULT_SEEDS = (
    873162450, 815294637, 425917638, 314857206, 841903276,
    168935742, 782410593, 638274591, 183649570, 350761928,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated DGRN evaluations and aggregate test metrics."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/taiwan.yaml", "configs/finland.yaml", "configs/milan.yaml"],
        help="Dataset configurations to run (default: the three bundled datasets).",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seeds; must contain exactly --runs values.",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42, help="First seed when --seeds is omitted."
    )
    parser.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto.")
    parser.add_argument(
        "--output-dir",
        default="runs_multiple_evaluation",
        help="Directory, relative to the repository, for per-run outputs and summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a prior output directory before starting. Required to replace results.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed trials and continue with the remaining trials.",
    )
    return parser.parse_args()


def load_config(relative_path: str) -> tuple[Path, dict[str, Any]]:
    path = (PROJECT_DIR / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return path, yaml.safe_load(f)


def resolve_device(requested: str) -> tuple[str, str]:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            return requested, "CUDA was requested but is unavailable."
        if requested == "mps" and not torch.backends.mps.is_available():
            return requested, "MPS was requested but is unavailable."
        return requested, "Requested device is available."
    if torch.cuda.is_available():
        return "cuda", "CUDA selected automatically."
    if torch.backends.mps.is_available():
        return "mps", "MPS selected automatically."
    return "cpu", "No GPU backend is available; CPU selected."


def format_mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ({std:.3f})"


def write_summaries(output_dir: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["dataset", "run", "seed", *METRICS, "epochs_run", "train_seconds", "metrics_path"]
    with (output_dir / "per_run_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_rows = []
    for dataset in sorted({row["dataset"] for row in records}):
        dataset_rows = [row for row in records if row["dataset"] == dataset]
        if len(dataset_rows) < 2:
            continue
        row: dict[str, Any] = {"dataset": dataset, "runs": len(dataset_rows)}
        for metric in METRICS:
            values = [float(item[metric]) for item in dataset_rows]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values)  # Sample standard deviation (n - 1).
            row[f"{metric}_mean_std"] = format_mean_std(values)
        summary_rows.append(row)

    summary_fields = ["dataset", "runs"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std")
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    by_dataset = {row["dataset"]: row for row in summary_rows}
    with (output_dir / "dgrn_multi_run.md").open("w", encoding="utf-8") as f:
        f.write("# DGRN multi-run evaluation\n\n")
        f.write("Sample standard deviations (n-1) over independent seeds.\n\n")
        f.write("| Model | Vehicular GCT (RMSE / MAE / R2) | Urban Wi-Fi (RMSE / MAE / R2) | Telecom Internet (RMSE / MAE / R2) |\n")
        f.write("|---|---|---|---|\n")
        cells = []
        for dataset in ("taiwan", "finland", "milan"):
            row = by_dataset.get(dataset)
            if row is None:
                cells.append("not completed")
            else:
                cells.append(" / ".join(row[f"{metric}_mean_std"] for metric in METRICS))
        f.write(f"| DGRN ({len(records) // max(len(by_dataset), 1)} runs per completed dataset) | {' | '.join(cells)} |\n")


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to compute a sample standard deviation.")
    if args.seeds is not None:
        seeds = args.seeds
    elif args.runs == len(DEFAULT_SEEDS):
        seeds = list(DEFAULT_SEEDS)
    else:
        seeds = list(range(args.base_seed, args.base_seed + args.runs))
    if len(seeds) != args.runs:
        raise SystemExit("--seeds must contain exactly --runs values.")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Seeds must be distinct.")

    configs = [load_config(config_path) for config_path in args.configs]
    device, device_detail = resolve_device(args.device)
    if "unavailable" in device_detail.lower():
        raise SystemExit(device_detail)
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory exists and contains results: {output_dir}. Use --overwrite to replace it.")

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if device == "mps":
        run_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    records: list[dict[str, Any]] = []
    for config_path, cfg in configs:
        dataset = cfg["dataset"]
        for run_index, seed in enumerate(seeds, start=1):
            run_root = output_dir / "artifacts" / f"run_{run_index:02d}_seed_{seed}"
            metrics_path = run_root / dataset / "metrics.json"
            log_path = logs_dir / f"{dataset}_run_{run_index:02d}_seed_{seed}.log"
            command = [
                sys.executable, "train.py", "--config", str(config_path.relative_to(PROJECT_DIR)),
                "--seed", str(seed), "--device", device, "--output-dir", str(run_root.relative_to(PROJECT_DIR)),
            ]
            print(f"\n=== {dataset}: run {run_index}/{args.runs}, seed={seed} ===")
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    env=run_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            if completed.returncode != 0 or not metrics_path.is_file():
                message = f"Trial failed (exit={completed.returncode}); inspect {log_path}"
                print(message)
                if not args.continue_on_error:
                    raise SystemExit(message)
                continue
            with metrics_path.open("r", encoding="utf-8") as f:
                metric_data = json.load(f)
            missing = [metric for metric in METRICS if metric not in metric_data]
            if missing:
                raise SystemExit(f"Missing metrics {missing} in {metrics_path}")
            records.append({
                "dataset": dataset, "run": run_index, "seed": seed,
                **{metric: metric_data[metric] for metric in METRICS},
                "epochs_run": metric_data.get("epochs_run"),
                "train_seconds": metric_data.get("train_seconds"),
                "metrics_path": str(metrics_path.relative_to(PROJECT_DIR)),
            })
            write_summaries(output_dir, records)
    write_summaries(output_dir, records)
    print(f"Completed {len(records)} successful trials. Summary: {output_dir / 'dgrn_multi_run.md'}")


if __name__ == "__main__":
    main()
