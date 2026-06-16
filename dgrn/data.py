from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class MinMaxScaler:
    data_min: float
    data_max: float

    @property
    def data_range(self) -> float:
        return max(self.data_max - self.data_min, 1e-8)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.data_min) / self.data_range

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.data_range + self.data_min


class TrafficWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        start: int,
        end: int,
        close_len: int,
        pred_len: int,
        steps_per_day: int,
    ) -> None:
        self.values = values.astype(np.float32)
        self.start = start
        self.end = end
        self.close_len = close_len
        self.pred_len = pred_len
        self.steps_per_day = steps_per_day
        self.first_idx = max(start, close_len)
        self.last_idx = end - pred_len

        if self.last_idx <= self.first_idx:
            raise ValueError(
                f"Not enough data for windows: start={start}, end={end}, "
                f"close_len={close_len}, pred_len={pred_len}"
            )

    def __len__(self) -> int:
        return self.last_idx - self.first_idx

    def __getitem__(self, item: int):
        out_start = self.first_idx + item
        input_start = out_start - self.close_len
        input_end = out_start
        output_end = out_start + self.pred_len

        x = self.values[input_start:input_end].T
        y = self.values[out_start:output_end].T
        mask = np.ones_like(x, dtype=np.float32)
        time_fea = make_time_features(np.arange(input_start, input_end), self.steps_per_day)
        return (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(time_fea),
            torch.from_numpy(y),
        )


def make_time_features(indices: np.ndarray, steps_per_day: int) -> np.ndarray:
    day_phase = (indices % steps_per_day) / steps_per_day
    week_phase = ((indices // steps_per_day) % 7) / 7.0
    month_phase = ((indices // (steps_per_day * 30)) % 12) / 12.0
    day_index = (indices // steps_per_day) % 7
    hour = (indices % steps_per_day) / steps_per_day * 24.0
    is_midnight = ((hour >= 1.0) & (hour <= 6.0)).astype(np.float32)
    weekend = (day_index >= 5).astype(np.float32)

    features = np.stack(
        [
            np.sin(2 * np.pi * month_phase),
            np.cos(2 * np.pi * month_phase),
            np.sin(2 * np.pi * week_phase),
            np.cos(2 * np.pi * week_phase),
            np.sin(2 * np.pi * day_phase),
            np.cos(2 * np.pi * day_phase),
            is_midnight,
            weekend,
            np.zeros_like(day_phase, dtype=np.float32),
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def load_matrix(path: str | Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.astype(np.float32)


def build_dataloaders(config: dict, project_dir: Path):
    data_path = project_dir / config["data_csv"]
    traffic = load_matrix(data_path)
    expected_nodes = int(config["num_nodes"])
    if traffic.shape[1] != expected_nodes:
        raise ValueError(
            f"{data_path} has {traffic.shape[1]} nodes, config expects {expected_nodes}."
        )

    time_range = float(config.get("time_range", 1.0))
    if time_range < 1.0:
        traffic = traffic[: round(len(traffic) * time_range)]

    train_ratio = float(config.get("train_ratio", 0.8))
    val_ratio = float(config.get("val_ratio", 0.1))
    train_end = int(len(traffic) * train_ratio)
    val_end = int(len(traffic) * (train_ratio + val_ratio))
    scaler = MinMaxScaler(float(traffic[:train_end].min()), float(traffic[:train_end].max()))
    traffic_scaled = scaler.transform(traffic)

    close_len = int(config["close_len"])
    pred_len = int(config["pred_len"])
    steps_per_day = int(config["steps_per_day"])
    batch_size = int(config.get("batch_size", 32))

    train_set = TrafficWindowDataset(
        traffic_scaled, 0, train_end, close_len, pred_len, steps_per_day
    )
    val_set = TrafficWindowDataset(
        traffic_scaled,
        max(0, train_end - close_len),
        val_end,
        close_len,
        pred_len,
        steps_per_day,
    )
    test_set = TrafficWindowDataset(
        traffic_scaled,
        max(0, val_end - close_len),
        len(traffic_scaled),
        close_len,
        pred_len,
        steps_per_day,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, scaler, traffic.shape
