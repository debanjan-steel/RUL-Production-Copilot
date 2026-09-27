"""Model definitions for RUL prediction."""
from src.models.cnn_transformer import CNNTransformer, PhysicsInformedRULLoss

__all__ = ["CNNTransformer", "PhysicsInformedRULLoss"]
