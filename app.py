import os
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
from joblib import load
from modma_eeg import extract_features

# Paths
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "exported_model")

# Load config + artifacts once at startup
with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

selector = load(os.path.join(MODEL_DIR, "selector.joblib"))
ensemble = load(os.path.join(MODEL_DIR, "ensemble_model.joblib"))

app = FastAPI(title="EEG Anxiety Detection API")

# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
async def root():
    return {"message": "EEG Anxiety Detection API is live!"}

# -------------------------
# Prediction logic
# -------------------------
def predict_from_bytes(file_bytes: bytes, filename: str):
    # Save temp file because extract_features expects a path
    temp_path = os.path.join(ROOT, "temp_input.mat")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    feats = extract_features(temp_path)
    os.remove(temp_path)

    if feats is None or getattr(feats, "ndim", 0) != 1:
        raise RuntimeError("Invalid features extracted")

    Xsel = selector.transform(np.asarray(feats, float).reshape(1, -1))
    probs = ensemble.predict_proba(Xsel)[0]
    idx = int(np.argmax(probs))

    labels = CONFIG.get("labels", {"0": "not_anxious", "1": "anxious"})
    label = labels.get(str(idx), str(idx))

    return {
        "label": label,
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": {
            labels.get("0","0"): round(float(probs[0]) * 100, 2),
            labels.get("1","1"): round(float(probs[1]) * 100, 2),
        },
    }

# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_from_bytes(contents, file.filename)
    return result
