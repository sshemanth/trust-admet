from fastapi import FastAPI
from pydantic import BaseModel

from trust_admet.trust.trust_engine import predict_with_trust


app = FastAPI(
    title="TRUST-ADMET API",
    description="Trustworthy ADMET prediction API",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    smiles: str
    dataset: str = "BBBP"
    split: str = "scaffold"
    model: str = "random_forest"
    seed: str = "42"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    return predict_with_trust(
        smiles=req.smiles,
        dataset=req.dataset,
        split=req.split,
        model_name=req.model,
        seed=req.seed,
    )
