import os
import io
import gc
import json
import numpy as np
from functools import lru_cache
from tempfile import NamedTemporaryFile
import logging
import psutil

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
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log_memory(stage=""):
    """Logs current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    logging.info(f"[MEMORY] {stage}: {mem:.2f} MB")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="EEG Anxiety Detection API")

# -------------------------
# CORS middleware
# -------------------------
origins = [
    "http://localhost:9002",               # local dev
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
# -------------------------
def _pick_existing(*candidates: str) -> str:
    for p in candidates:
        fp = os.path.join(MODEL_DIR, p)
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"None of the model files exist: {candidates}")

@lru_cache(maxsize=1)
def get_models():
    """Load models lazily with memory-mapped arrays."""
    selector_path = _pick_existing("selector_compressed.joblib", "selector.joblib")
    ensemble_path = _pick_existing("ensemble_model_compressed.joblib", "ensemble_model.joblib")

    selector = load(selector_path, mmap_mode="r")
    ensemble = load(ensemble_path, mmap_mode="r")

    log_memory("After loading models")
    return selector, ensemble

# -------------------------
# Root / health endpoints
# -------------------------
@app.get("/")
async def root():
    return {"message": "EEG Anxiety Detection API is live!"}

@app.get("/health")
async def health():
    try:
        get_models()
        log_memory("Health check")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Prediction logic
# -------------------------
def predict_from_bytes(file_bytes: bytes):
    tmp_path = None
    try:
        # Save uploaded bytes to a temporary file
        with NamedTemporaryFile(suffix=".mat", delete=False, dir=ROOT) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        log_memory("Before feature extraction")
        feats = extract_features(tmp_path)
        log_memory("After feature extraction")

        if feats is None or getattr(feats, "ndim", 0) != 1:
            raise RuntimeError("Invalid features extracted")

        selector, ensemble = get_models()

        # Transform features
        Xsel = selector.transform(np.asarray(feats, dtype=np.float64).reshape(1, -1))
        log_memory("After selector transform")

        # Make prediction
        probs = ensemble.predict_proba(Xsel)[0]
        log_memory("After prediction")

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
        gc.collect()

# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = None
    try:
        contents = await file.read()
        log_memory("After file upload")
        result = predict_from_bytes(contents)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if contents:
            del contents
        gc.collect()
