from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Finita = Annotated[float, Field(allow_inf_nan=False)]


class PredictRequest(BaseModel):
    """Últimas `window` horas de consumo (kW) en orden temporal.

    El modelo predice la hora siguiente (t+1) a partir de la ventana.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "series": [
                    0.82,
                    0.85,
                ]
            }
        }
    )

    series: list[Finita]


class PredictResponse(BaseModel):
    """Predicción de la siguiente hora (kW, nunca negativa)."""

    prediction: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    source: str
    window: int
