"""Benchmark workloads.

Each workload has a PyTorch implementation (preferred) and a NumPy fallback
with comparable FLOP profile, so the pipeline runs in any environment
(e.g. lightweight CI runners without torch installed).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

WORKLOADS = ["mlp", "tiny_cnn", "mini_transformer"]
BATCH_SIZES = [1, 8]


# --------------------------------------------------------------------------
# Torch backend
# --------------------------------------------------------------------------
def build_torch_model(name: str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed")
    if name == "mlp":
        return nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 256),
        )
    if name == "tiny_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 10),
        )
    if name == "mini_transformer":
        return nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, batch_first=True
        )
    raise ValueError(f"unknown workload: {name}")


def torch_input(name: str, batch_size: int):
    if name == "mlp":
        return torch.randn(batch_size, 512)
    if name == "tiny_cnn":
        return torch.randn(batch_size, 3, 64, 64)
    if name == "mini_transformer":
        return torch.randn(batch_size, 32, 128)
    raise ValueError(f"unknown workload: {name}")


# --------------------------------------------------------------------------
# NumPy fallback backend (matmul chains sized to roughly match torch FLOPs)
# --------------------------------------------------------------------------
_NUMPY_SHAPES = {
    "mlp": [(512, 1024), (1024, 1024), (1024, 256)],
    "tiny_cnn": [(4096, 288), (288, 512), (512, 64)],
    "mini_transformer": [(128, 384), (384, 128), (128, 256), (256, 128)],
}


def numpy_forward(name: str, batch_size: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal((batch_size * 8, _NUMPY_SHAPES[name][0][0])).astype(np.float32)
    for shape in _NUMPY_SHAPES[name]:
        w = rng.standard_normal(shape).astype(np.float32)
        x = np.maximum(x @ w, 0.0)
