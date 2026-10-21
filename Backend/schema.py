from typing import Literal

from pydantic import BaseModel, Field

Borough = Literal["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
RoomType = Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]


class PredictRequest(BaseModel):
    borough: Borough
    room_type: RoomType
    minimum_nights: int = Field(1, ge=1, le=365)
    availability_365: int = Field(180, ge=0, le=365)
    description: str = Field("", max_length=2000)
