import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from calibration import prediction_interval
from features import build_live_features
from load import load_artifacts
from schema import PredictRequest

app = FastAPI(title="NYC Airbnb Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all ML artifacts ONCE
pre, rf, lr, km, nn, conformal = load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_price(req: PredictRequest):
    df = build_live_features(
        borough=req.borough,
        room_type=req.room_type,
        minimum_nights=req.minimum_nights,
        availability_365=req.availability_365,
        description=req.description,
        km=km,
    )
    X = pre.transform(df)

    pred_rf = float(rf.predict(X)[0])
    pred_lr = float(lr.predict(X)[0])

    dense = X.toarray() if hasattr(X, "toarray") else X
    pred_nn = float(nn.predict(dense, verbose=0)[0][0])

    final_price = pred_rf
    interval_low, interval_high = prediction_interval(final_price, conformal["log_ratio_quantile"])

    return {
        "random_forest": pred_rf,
        "linear_regression": pred_lr,
        "neural_network": pred_nn,
        "final_price": final_price,
        "interval_low": float(interval_low),
        "interval_high": float(interval_high),
        "interval_coverage": conformal["target_coverage"],
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
