"""FastAPI Serving Layer for Uncertainty-Aware Turbofan RUL Prediction.

Exposes REST endpoints for real-time inference, Monte Carlo Dropout uncertainty
estimation, feature attribution, and grounded Copilot recommendations.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
import torch

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.preprocess import CmapssPreprocessor
from src.models.cnn_transformer import CNNTransformer
from src.copilot.agent import OpsCopilotAgent

app = FastAPI(
    title="Predictive Maintenance RUL OpsCopilot API",
    description="Production-grade API serving a Multi-Scale CNN-Transformer with Bayesian UQ and safety guardrails.",
    version="1.0.0",
)

# Global runtime state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = CmapssPreprocessor(sequence_length=30)
model = CNNTransformer(in_features=14, d_model=64, nhead=4, num_layers=2).to(DEVICE)
agent = OpsCopilotAgent(uncertainty_threshold=18.0)

# Check if pre-trained checkpoint exists
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
if CHECKPOINT_PATH.exists():
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {CHECKPOINT_PATH}: {e}")
else:
    model.eval()


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Turbofan RUL OpsCopilot API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(
        status="HEALTHY",
        model_loaded=True,
        device=str(DEVICE),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rul(data: PredictionRequest):
    """Predicts Remaining Useful Life with Monte Carlo Dropout uncertainty and safety evaluation."""
    try:
        # 1. Format input sequence
        if data.sequence is not None and len(data.sequence) > 0:
            raw_seq = data.sequence
        elif data.sensor_readings is not None and len(data.sensor_readings) > 0:
            # Replicate single cycle reading to create sequence window
            raw_seq = [data.sensor_readings for _ in range(preprocessor.sequence_length)]
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either a 2D 'sequence' or a 1D 'sensor_readings' array.",
            )

        # 2. Preprocess and convert to PyTorch tensor
        processed_window = preprocessor.transform_single_window(raw_seq)
        input_tensor = torch.tensor(processed_window, dtype=torch.float32, device=DEVICE)

        # 3. Monte Carlo Dropout Inference (Epistemic Uncertainty)
        mean_rul, uncertainty_std, _ = model.predict_mc_dropout(
            input_tensor, mc_samples=data.mc_samples
        )

        # 4. Compute 95% Bayesian credible interval
        lower_bound = max(0.0, float(mean_rul - 1.96 * uncertainty_std))
        upper_bound = float(mean_rul + 1.96 * uncertainty_std)

        # 5. OpsCopilot Safety & RAG Evaluation
        evaluation = agent.evaluate(
            model=model,
            input_tensor=input_tensor,
            mean_rul=mean_rul,
            uncertainty_std=uncertainty_std,
        )

        return PredictionResponse(
            engine_id=data.engine_id,
            cycle=data.cycle,
            predicted_rul=round(mean_rul, 2),
            uncertainty_std=round(uncertainty_std, 2),
            confidence_interval_95=[round(lower_bound, 2), round(upper_bound, 2)],
            status=evaluation["status"],
            confidence_level=evaluation["confidence_level"],
            recommendation=evaluation["recommendation"],
            top_degrading_sensors=evaluation["top_degrading_sensors"],
            citations=evaluation["citations"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")