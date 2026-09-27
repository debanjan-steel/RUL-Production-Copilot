"""Multi-Scale CNN-Transformer Architecture with Monte Carlo Dropout and Physics Loss.

Combines parallel multi-scale 1D convolutions for local temporal feature extraction
with multi-head self-attention for global degradation trajectory modeling.
Supports Bayesian uncertainty quantification via active MC Dropout at inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence representation."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiScaleConvBlock(nn.Module):
    """Parallel 1D convolutions extracting multi-frequency degradation signals."""

    def __init__(self, in_features: int, out_channels: int = 32):
        super().__init__()
        # Branch 1: High-frequency transient spikes (kernel=3)
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(in_features, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # Branch 2: Intermediate degradation patterns (kernel=5)
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(in_features, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # Branch 3: Broad baseline shifts (kernel=7)
        self.branch_k7 = nn.Sequential(
            nn.Conv1d(in_features, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # Fusion projection back to out_channels
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, in_features, seq_len)
        k3 = self.branch_k3(x)
        k5 = self.branch_k5(x)
        k7 = self.branch_k7(x)
        concat = torch.cat([k3, k5, k7], dim=1)
        return self.fusion(concat)


class CNNTransformer(nn.Module):
    """State-of-the-art CNN-Transformer with MC Dropout for Turbofan RUL."""

    def __init__(
        self,
        in_features: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.dropout_rate = dropout

        # 1. Multi-scale feature extraction
        self.conv_block = MultiScaleConvBlock(in_features=in_features, out_channels=d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Regression & MC Dropout Head
        self.dropout_layer = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.GELU()
        self.regressor = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, in_features)
        # Transpose for Conv1d: (batch_size, in_features, seq_len)
        x_conv = x.transpose(1, 2)
        feat = self.conv_block(x_conv)  # (batch_size, d_model, seq_len)

        # Transpose back for Transformer: (batch_size, seq_len, d_model)
        feat = feat.transpose(1, 2)
        feat = self.pos_encoder(feat)
        trans_out = self.transformer_encoder(feat)  # (batch_size, seq_len, d_model)

        # Global average pooling across time dimension
        pooled = torch.mean(trans_out, dim=1)  # (batch_size, d_model)

        # Regressor with dropout
        h = self.dropout_layer(pooled)
        h = self.relu(self.fc1(h))
        h = self.dropout_layer(h)
        out = self.regressor(h)  # (batch_size, 1)
        return out.squeeze(-1)

    def enable_mc_dropout(self) -> None:
        """Forces dropout layers to stay active during evaluation for UQ."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict_mc_dropout(
        self, x: torch.Tensor, mc_samples: int = 50
    ) -> tuple[float, float, list[float]]:
        """Runs stochastic MC Dropout forward passes to estimate Bayesian uncertainty.
        
        Returns:
            mean_rul: Expected predicted RUL.
            uncertainty_std: Epistemic uncertainty standard deviation.
            all_preds: Raw predictions across all stochastic passes.
        """
        self.eval()
        self.enable_mc_dropout()

        preds = []
        for _ in range(mc_samples):
            pred = self.forward(x)
            preds.append(pred.item() if pred.numel() == 1 else pred.cpu().numpy())

        preds_arr = torch.tensor(preds, dtype=torch.float32)
        mean_rul = float(torch.mean(preds_arr).item())
        uncertainty_std = float(torch.std(preds_arr).item())
        
        return mean_rul, uncertainty_std, [float(p) for p in preds]


class PhysicsInformedRULLoss(nn.Module):
    """Custom loss enforcing physical wear monotonicity (non-reversible degradation).
    
    Penalizes instances where the model predicts increasing RUL over continuous cycles.
    """

    def __init__(self, lambda_monotonicity: float = 0.1):
        super().__init__()
        self.mse_loss = nn.SmoothL1Loss()
        self.lambda_monotonicity = lambda_monotonicity

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.mse_loss(pred, target)
        
        # Monotonicity penalty along consecutive sequence predictions
        if pred.size(0) > 1:
            diffs = pred[1:] - pred[:-1]
            # Physical machinery wear cannot decrease (RUL cannot increase: delta_RUL <= 0)
            monotonic_violation = F.relu(diffs)
            mono_loss = torch.mean(monotonic_violation ** 2)
        else:
            mono_loss = torch.tensor(0.0, device=pred.device)

        return base_loss + self.lambda_monotonicity * mono_loss
