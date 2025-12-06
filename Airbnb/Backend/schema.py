from pydantic import BaseModel

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    borough: str
    room_type: str
    minimum_nights: int = 1
    guests: int = 1
    description: str = ""
