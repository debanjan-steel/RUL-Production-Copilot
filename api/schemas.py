"""Pydantic V2 Schemas for FastAPI Request/Response Validation.

Defines typed schemas for turbofan telemetry sequences, MC Dropout uncertainty,
feature attribution rankings, and grounded maintenance recommendations.
"""

from pydantic import BaseModel, Field


class SensorImportance(BaseModel):
    sensor: str = Field(..., description="Telemetry sensor identifier (e.g. s2, s3, s4)")
    importance: float = Field(..., description="Normalized gradient attribution score")


class PredictionRequest(BaseModel):
    engine_id: int = Field(..., example=1, description="Turbofan engine unique asset ID")
    cycle: int = Field(..., example=120, description="Current operational cycle index")
    sequence: list[list[float]] | None = Field(
        None,
        description="Temporal sequence of historical cycles, shape [time_steps, num_sensors]",
        example=[[642.0 + j * 0.1 for j in range(14)] for _ in range(30)],
    )
    sensor_readings: list[float] | None = Field(
        None,
        description="Single cycle sensor readings (auto-padded if sequence not provided)",
        example=[642.5, 1588.0, 1405.0, 553.0, 2388.0, 9050.0, 47.5, 521.0, 2388.0, 8130.0, 8.4, 392.0, 38.8, 23.3],
    )
    mc_samples: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of stochastic forward passes for Bayesian uncertainty estimation",
    )


class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float = Field(..., description="Expected Remaining Useful Life in operating cycles")
    uncertainty_std: float = Field(..., description="Epistemic uncertainty standard deviation (sigma)")
    confidence_interval_95: list[float] = Field(
        ..., description="[Lower, Upper] 95% Bayesian credible interval"
    )
    status: str = Field(..., description="'CONFIDENT' or 'ABSTAIN_ESCALATE'")
    confidence_level: str = Field(..., description="'HIGH', 'MODERATE', or 'LOW_UNCERTAIN'")
    recommendation: str = Field(..., description="Actionable root cause guidance grounded in maintenance KB")
    top_degrading_sensors: list[SensorImportance] = Field(
        default_factory=list, description="Top sensors contributing to wear signature"
    )
    citations: list[str] = Field(
        default_factory=list, description="Knowledge base reference manual sections cited"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
