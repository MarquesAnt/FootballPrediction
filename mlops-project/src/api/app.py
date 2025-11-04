from fastapi import FastAPI, Response
from pydantic import BaseModel
import pandas as pd
import joblib
from prometheus_client import Counter, generate_latest
from pathlib import Path

app = FastAPI(title="MLOps Demo API", version="0.1.0")

PREDICTS = Counter("predict_requests_total", "Prediction calls")

class InputRow(BaseModel):
    load: float
    temp: float

class Batch(BaseModel):
    rows: list[InputRow]

# lazy-load model if present
_model = None
def get_model():
    global _model
    if _model is None:
        mp = Path("models/model.pkl")
        if mp.exists():
            _model = joblib.load(mp)
        else:
            # fallback: mean baseline
            class MeanModel:
                def predict(self, X):
                    return [float(X["load"].mean())] * len(X)
            _model = MeanModel()
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
def predict(batch: Batch):
    PREDICTS.inc()
    df = pd.DataFrame([r.dict() for r in batch.rows])
    yhat = get_model().predict(df)
    return {"predictions": list(map(float, yhat))}
