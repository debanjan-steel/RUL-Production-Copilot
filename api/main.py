from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# In a real environment, you will import these from your src folders:
# from src.models.cnn_transformer import CNNTransformer
# from src.preprocess import preprocess_sensor_data
# from src.copilot.agent import evaluate_uncertainty

app = FastAPI(title="RUL Copilot API", version="1.0.0")

# 1. Define the Expected Input Structure using Pydantic
class SensorDataInput(BaseModel):
    engine_id: int
    cycle: int
    sensor_readings: list[float]

# 2. Define the Output Structure
class PredictionOutput(BaseModel):
    engine_id: int
    predicted_rul: float
    uncertainty_std: float
    recommendation: str
    status: str

# 3. Dummy functions to simulate your src logic until we connect them
def mock_preprocess(data):
    return data

def mock_predict(data):
    # Simulating 50 MC Dropout passes returning random variations of RUL
    return np.random.normal(loc=120, scale=15, size=50)

def mock_evaluate(mean_rul, uncertainty):
    if uncertainty > 20:
        return "ESCALATE_TO_HUMAN", "Uncertainty too high. Manual inspection required."
    return "CONFIDENT", f"Run standard maintenance in {int(mean_rul)} cycles."

@app.post("/predict", response_model=PredictionOutput)
async def predict_rul(data: SensorDataInput):
    try:
        # Step 1: Preprocess
        processed_input = mock_preprocess(data.sensor_readings)

        # Step 2: MC Dropout Inference
        predictions = mock_predict(processed_input)
                
        # Step 3: Calculate mean RUL and uncertainty
        mean_rul = float(np.mean(predictions))
        uncertainty = float(np.std(predictions))
        
        # Step 4: Pass through the Copilot Agent rules
        status, recommendation = mock_evaluate(mean_rul, uncertainty)

        return PredictionOutput(
            engine_id=data.engine_id,
            predicted_rul=mean_rul,
            uncertainty_std=uncertainty,
            recommendation=recommendation,
            status=status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))