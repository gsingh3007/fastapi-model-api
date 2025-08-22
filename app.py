import os
import io
import gc
import json
import numpy as np
from functools import lru_cache
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from modma_eeg import extract_features

# -------------------------
# Paths & config
# -------------------------
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "exported_model")

with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="EEG Anxiety Detection API")

# CORS
origins = [
    "http://localhost:9002",              # local dev
    "https://anxiocheck-ypipu.web.app",   # Firebase frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Lazy, memory-mapped model loading
# (Loads once, keeps arrays memory-mapped from disk)
# -------------------------
def _pick_existing(*candidates: str) -> str:
    for p in candidates:
        fp = os.path.join(MODEL_DIR, p)
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"None of the model files exist: {candidates}")

@lru_cache(maxsize=1)
def get_models():
    # Prefer compressed artifacts if present (you can create them offline with joblib.dump(..., compress=3))
    selector_path = _pick_existing("selector_compressed.joblib", "selector.joblib")
    ensemble_path = _pick_existing("ensemble_model_compressed.joblib", "ensemble_model.joblib")

    # mmap_mode='r' keeps large numpy arrays on disk instead of copying into RAM
    selector = load(selector_path, mmap_mode="r")
    ensemble = load(ensemble_path, mmap_mode="r")
    return selector, ensemble

# -------------------------
# Root / health
# -------------------------
@app.get("/")
async def root():
    return {"message": "EEG Anxiety Detection API is live!"}

@app.get("/health")
async def health():
    try:
        get_models()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Prediction logic
# -------------------------
def predict_from_bytes(file_bytes: bytes):
    # If your extract_features requires a file path, write a temp file, then remove it ASAP.
    # Using delete=False + finally to ensure cleanup on Windows too.
    tmp_path = None
    try:
        with NamedTemporaryFile(suffix=".mat", delete=False, dir=ROOT) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        feats = extract_features(tmp_path)

        if feats is None or getattr(feats, "ndim", 0) != 1:
            raise RuntimeError("Invalid features extracted")

        selector, ensemble = get_models()

        # NOTE: keeping dtype default (user asked for 1,2,3,5; downcasting is step 4)
        Xsel = selector.transform(np.asarray(feats, dtype=np.float64).reshape(1, -1))
        probs = ensemble.predict_proba(Xsel)[0]
        idx = int(np.argmax(probs))

        labels = CONFIG.get("labels", {"0": "not_anxious", "1": "anxious"})
        label = labels.get(str(idx), str(idx))

        return {
            "label": label,
            "confidence": round(float(probs[idx]) * 100, 2),
            "probabilities": {
                labels.get("0", "0"): round(float(probs[0]) * 100, 2),
                labels.get("1", "1"): round(float(probs[1]) * 100, 2),
            },
        }
    finally:
        # Aggressive cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        # Help the 512MB instance by freeing intermediates promptly
        gc.collect()

# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_from_bytes(contents)
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Avoid exposing internals to client; logs on host will have full trace
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Ensure the uploaded buffer can be GC'd
        del contents
        gc.collect()
