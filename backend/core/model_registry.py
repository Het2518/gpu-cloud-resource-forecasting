"""
core/model_registry.py

Loads all saved model artifacts from disk into memory at application startup.
Every module that needs a model imports the singleton `registry` object.

Usage:
    from core.model_registry import registry
    rf_model = registry.get("rf_gpu_15min")
    scaler_X = registry.get("scaler_X")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import torch
import torch.nn as nn

from config import MODEL_DIR, MODEL_FILES, PATCHTST_PARAMS

logger = logging.getLogger(__name__)


# ─── PatchTST Architecture (must match NB05 / NB06 exactly) ──────────────────

class PatchTST(nn.Module):
    """
    Patch-based Time Series Transformer with quantile output (P10, P50, P90).
    Architecture is identical to what was trained in notebook 05.
    """

    def __init__(
        self,
        n_feat: int,
        seq_len: int,
        patch_len: int = 4,
        stride: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        pad = max(0, patch_len - stride)
        padded_len = seq_len + pad
        self.num_patches = (padded_len - patch_len) // stride + 1
        self.pad = pad

        self.patch_embed = nn.Linear(patch_len * n_feat, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 3 outputs: P10, P50, P90
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.pad > 0:
            x = torch.cat(
                [torch.zeros(B, self.pad, x.size(2), device=x.device), x], dim=1
            )
        patches = x.unfold(1, self.patch_len, self.stride).contiguous()
        patches = patches.view(B, self.num_patches, -1)
        emb = self.dropout(self.patch_embed(patches) + self.pos_embed)
        enc = self.encoder(emb)
        return self.head(enc)


# ─── Registry ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Singleton that holds all loaded models and scalers in memory.
    Call `load_all()` once at application startup.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._loaded: bool = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Load every model file listed in config.MODEL_FILES."""
        logger.info("ModelRegistry: loading models from %s", MODEL_DIR)
        logger.info("ModelRegistry: torch device = %s", self.device)

        for key, filename in MODEL_FILES.items():
            path = MODEL_DIR / filename
            if not path.exists():
                logger.warning("  [SKIP] %s — file not found: %s", key, path)
                continue
            try:
                if filename.endswith(".pt"):
                    model = self._load_torch(key, path)
                else:
                    model = joblib.load(path)
                self._models[key] = model
                logger.info("  [OK]   %s  <- %s", key, filename)
            except Exception as exc:
                logger.error("  [ERR]  %s — %s. If loading joblib, check for numpy version mismatch.", key, exc)

        self._loaded = True
        logger.info("ModelRegistry: %d / %d models loaded.", len(self._models), len(MODEL_FILES))

    def get(self, key: str) -> Any:
        """Return a loaded model by key. Raises KeyError if not found."""
        if key not in self._models:
            raise KeyError(f"Model '{key}' not in registry. Available: {self.available}")
        return self._models[key]

    def has(self, key: str) -> bool:
        return key in self._models

    @property
    def available(self) -> List[str]:
        return sorted(self._models.keys())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_torch(self, key: str, path: Path) -> nn.Module:
        """Instantiate the correct PyTorch architecture and load weights."""
        if key == "patchtst":
            model = PatchTST(**PATCHTST_PARAMS)
        else:
            raise ValueError(f"Unknown PyTorch model key: '{key}'")

        state = torch.load(str(path), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model


# ─── Singleton ────────────────────────────────────────────────────────────────

registry = ModelRegistry()
