from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd

from load import load_artifacts
from predict import prepare_features   
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
BOROUGH_COORDS = {
    "Manhattan":    (40.7831, -73.9712),
    "Brooklyn":     (40.6782, -73.9442),
    "Queens":       (40.7282, -73.7949),
    "Bronx":        (40.8448, -73.8648),
    "Staten Island":(40.5795, -74.1502)
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load all ML artifacts ONCE
pre, rf, lr, km, nn = load_artifacts()


class PredictRequest(BaseModel):
    borough: str
    room_type: str
    minimum_nights: int
    availability_365: int
    guests: int = 1
    description: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_price(req: PredictRequest):

    # get lat/lon automatically from borough
    lat, lon = BOROUGH_COORDS[req.borough]

    # build feature dict for your model
    df = prepare_features({
        "latitude": lat,
        "longitude": lon,
        "borough": req.borough,
        "room_type": req.room_type,
        "minimum_nights": req.minimum_nights,
        "availability_365": req.availability_365,
        "guests": req.guests,
        "description": req.description
    }, km)

    X = pre.transform(df)

    pred_rf = float(rf.predict(X)[0])
    pred_lr = float(lr.predict(X)[0])

    dense = X.toarray() if hasattr(X, "toarray") else X
    pred_nn = float(nn.predict(dense)[0][0])

    final_price = pred_rf
    confidence_low = final_price * 0.9
    confidence_high = final_price * 1.1

    return {
        "random_forest": pred_rf,
        "linear_regression": pred_lr,
        "neural_network": pred_nn,
        "final_price": final_price,
        "confidence_low": confidence_low,
        "confidence_high": confidence_high
    }



if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
