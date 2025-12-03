import os
import gc
import json
import numpy as np
from functools import lru_cache
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from modma_eeg import extract_features
import psutil
import logging

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("anxiocheck-api")


def log_memory(stage=""):
    """Utility to monitor memory usage at key steps."""
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        logger.info(f"[MEMORY] {stage}: {mem:.2f} MB")
    except:
        pass


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "exported_model")

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    CONFIG = json.load(f)


# ---------------------------------------------------------
# FastAPI + CORS
# ---------------------------------------------------------
app = FastAPI(title="EEG Anxiety Detection API")

origins = [
    "http://localhost:9002",
    "http://localhost:3000",
    "https://anxiocheck-ypipu.web.app",
    "https://anxiocheck-ypipu.firebaseapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load models (lazy, cached, memory-mapped)
# ---------------------------------------------------------
def _pick_existing(*names):
    for n in names:
        p = os.path.join(MODEL_DIR, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Model files missing.")


@lru_cache(maxsize=1)
def get_models():
    selector_path = _pick_existing("selector_compressed.joblib", "selector.joblib")
    ensemble_path = _pick_existing("ensemble_model_compressed.joblib", "ensemble_model.joblib")

    selector = load(selector_path, mmap_mode="r")
    ensemble = load(ensemble_path, mmap_mode="r")

    log_memory("Models loaded")
    return selector, ensemble


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/health")
async def health():
    try:
        get_models()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_from_bytes(data_bytes: bytes):
    tmp_path = None
    try:
        # Save temp .mat file
        with NamedTemporaryFile(suffix=".mat", delete=False, dir=ROOT) as tmp:
            tmp.write(data_bytes)
            tmp.flush()
            tmp_path = tmp.name

        log_memory("Temp file saved")

        feats = extract_features(tmp_path)
        log_memory("Features extracted")

        if feats is None or getattr(feats, "ndim", 0) != 1:
            raise RuntimeError("Invalid features extracted")

        feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)

        selector, ensemble = get_models()
        Xsel = selector.transform(feats)
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
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

        gc.collect()
        log_memory("Cleanup complete")


# ---------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        return predict_from_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        gc.collect()
        log_memory("Request cleanup complete")
