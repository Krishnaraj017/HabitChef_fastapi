from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_foods: list[str]
