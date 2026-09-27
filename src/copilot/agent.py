"""OpsCopilot Agent: Uncertainty-Aware Abstention, Saliency Attribution & RAG Orchestrator.

Governs model predictions, enforces safety abstention thresholds, calculates
feature attribution to isolate degrading components, and queries domain knowledge bases.
"""

from typing import Any
import torch
import numpy as np

from src.data_loader import INFORMATIVE_SENSORS
from src.copilot.retriever import KnowledgeBaseRetriever


def compute_sensor_saliency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    feature_names: list[str] = INFORMATIVE_SENSORS,
) -> list[dict[str, Any]]:
    """Calculates gradient-based feature attribution to identify degradation drivers.
    
    Args:
        model: Trained PyTorch RUL model.
        input_tensor: Sequence tensor of shape (1, seq_len, n_features).
        feature_names: List of sensor names corresponding to the feature columns.
        
    Returns:
        List of dicts with sensor name and normalized attribution percentage.
    """
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)
    out = model(x)
    
    # Backpropagate gradient of predicted RUL w.r.t input telemetry
    out.backward()
    
    if x.grad is None:
        return [{"sensor": name, "attribution": 1.0 / len(feature_names)} for name in feature_names[:3]]
        
    # Average absolute gradients over time steps for each sensor
    grad_per_sensor = torch.mean(torch.abs(x.grad), dim=1).squeeze(0).cpu().numpy()
    total_grad = np.sum(grad_per_sensor) + 1e-8
    normalized_weights = grad_per_sensor / total_grad

    ranked = sorted(
        [{"sensor": name, "importance": float(w)} for name, w in zip(feature_names, normalized_weights)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked


class OpsCopilotAgent:
    """Agentic safety supervisor coordinating predictions, explanations, and RAG."""

    def __init__(
        self,
        uncertainty_threshold: float = 18.0,
        retriever: KnowledgeBaseRetriever | None = None,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.retriever = retriever or KnowledgeBaseRetriever()

    def evaluate(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        mean_rul: float,
        uncertainty_std: float,
    ) -> dict[str, Any]:
        """Evaluates prediction confidence, triggers abstention, and generates advice."""
        
        # 1. Feature Attribution (Saliency)
        saliency = compute_sensor_saliency(model, input_tensor)
        top_sensors = [s["sensor"] for s in saliency[:3]]

        # 2. Safety Abstention Gate
        if uncertainty_std > self.uncertainty_threshold:
            status = "ABSTAIN_ESCALATE"
            query = f"Sensor Fault {' '.join(top_sensors)}"
            retrieved = self.retriever.retrieve(query, top_k=1)
            citation = retrieved[0]["title"] if retrieved else "Sensor Fault / Calibration Drift"
            
            recommendation = (
                f"SAFETY ABSTENTION TRIGGERED: Model uncertainty (σ = {uncertainty_std:.1f} cycles) "
                f"exceeds the maximum allowable safety threshold (τ = {self.uncertainty_threshold:.1f}). "
                f"Automated RUL clearance is withheld. Immediate manual borescope inspection and "
                f"sensor wiring harness verification required. [Ref: {citation}]"
            )
            confidence_level = "LOW_UNCERTAIN"
        else:
            status = "CONFIDENT"
            confidence_level = "HIGH" if uncertainty_std < (self.uncertainty_threshold / 2) else "MODERATE"
            
            # Formulate query for RAG retrieval using top degradation indicators
            query = f"{' '.join(top_sensors)} failure degradation"
            retrieved = self.retriever.retrieve(query, top_k=2)
            citations = [doc["title"] for doc in retrieved]
            
            # Determine maintenance urgency based on estimated RUL
            if mean_rul < 25.0:
                urgency = "CRITICAL (P1)"
                action_window = "Immediate grounding / overhaul within 10 cycles."
            elif mean_rul < 50.0:
                urgency = "WARNING (P2)"
                action_window = f"Schedule maintenance within {int(mean_rul)} operating cycles."
            else:
                urgency = "NORMAL (P3)"
                action_window = f"Continue standard operations. Routine check due in {int(mean_rul)} cycles."

            rec_body = retrieved[0]["content"] if retrieved else "Perform standard engine inspection."
            # Extract first sentence of action
            action_snippet = rec_body.split("\n\n")[-1] if "\n\n" in rec_body else rec_body

            recommendation = (
                f"Urgency: {urgency}. {action_window} "
                f"Degradation signature driven primarily by sensors {', '.join(top_sensors)}. "
                f"Recommended Action: {action_snippet}"
            )

        return {
            "status": status,
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "top_degrading_sensors": saliency[:3],
            "citations": [doc["title"] for doc in self.retriever.retrieve(' '.join(top_sensors), top_k=2)],
        }
