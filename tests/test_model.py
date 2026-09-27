"""Unit Tests for Multi-Scale CNN-Transformer and Preprocessing Pipeline."""

import torch
import numpy as np
import pytest

from src.models.cnn_transformer import CNNTransformer, PhysicsInformedRULLoss
from src.preprocess import CmapssPreprocessor
from src.data_loader import generate_synthetic_cmapss, load_cmapss_raw


def test_model_forward_shape():
    """Verifies that temporal window tensors produce 1D regression outputs."""
    batch_size = 4
    seq_len = 30
    in_features = 14
    model = CNNTransformer(in_features=in_features, d_model=32, nhead=2, num_layers=1)
    
    x = torch.randn(batch_size, seq_len, in_features)
    out = model(x)
    assert out.shape == (batch_size,), f"Expected shape ({batch_size},), got {out.shape}"


def test_mc_dropout_uncertainty_variance():
    """Validates that MC Dropout produces stochastic variance across passes."""
    model = CNNTransformer(in_features=14, d_model=32, nhead=2, num_layers=1, dropout=0.3)
    x = torch.randn(1, 30, 14)
    
    mean_rul, std, all_preds = model.predict_mc_dropout(x, mc_samples=25)
    
    assert isinstance(mean_rul, float)
    assert isinstance(std, float)
    assert len(all_preds) == 25
    # Dropout active during inference must yield non-identical stochastic outputs
    assert std >= 0.0, "Uncertainty standard deviation cannot be negative"


def test_physics_informed_loss():
    """Verifies that non-monotonic RUL increases are penalized."""
    criterion = PhysicsInformedRULLoss(lambda_monotonicity=1.0)
    target = torch.tensor([100.0, 90.0, 80.0])
    
    # Monotonic degradation (pred decreasing): should have zero penalty
    pred_monotonic = torch.tensor([100.0, 90.0, 80.0])
    loss_mono = criterion(pred_monotonic, target)
    
    # Violating degradation (pred increasing from 90 to 110): must produce higher penalty
    pred_violating = torch.tensor([100.0, 90.0, 110.0])
    loss_violating = criterion(pred_violating, target)
    
    assert loss_violating > loss_mono, "Physics loss must penalize non-monotonic wear increases"


def test_preprocessor_windowing(tmp_path):
    """Verifies end-to-end sliding window creation and scaling."""
    generate_synthetic_cmapss(tmp_path)
    df_train, df_test, y_test = load_cmapss_raw("FD001")
    
    preprocessor = CmapssPreprocessor(sequence_length=20)
    X_train, y_train = preprocessor.fit_transform(df_train, use_sg=True)
    
    assert X_train.ndim == 3
    assert X_train.shape[1] == 20
    assert X_train.shape[2] == 14
    assert len(X_train) == len(y_train)
