import os
import gc
import json
import numpy as np
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from modma_eeg import extract_features
import psutil
import logging

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("anxiocheck-api")


def log_memory(stage=""):
    try:
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"[MEMORY] {stage}: {mem:.2f} MB")
    except:
        pass


# ---------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "exported_model")

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    CONFIG = json.load(f)

SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.joblib")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_model.joblib")

selector = None
ensemble = None


# ---------------------------------------------------------
# FastAPI App + CORS
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
# Load Models ONCE at startup
# ---------------------------------------------------------
@app.on_event("startup")
def load_models_once():
    global selector, ensemble

    logger.info("Loading models from disk...")
    selector = load(SELECTOR_PATH)  # NO mmap
    ensemble = load(ENSEMBLE_PATH)  # NO mmap

    log_memory("Models loaded")
    logger.info("Models loaded successfully.")


# ---------------------------------------------------------
# Health Endpoint (NO model loading!)
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True}


# ---------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------
def predict_from_bytes(data: bytes):
    tmp_path = None

    try:
        # Save temp .mat file
        with NamedTemporaryFile(suffix=".mat", delete=False, dir=ROOT) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        log_memory("Temp file saved")

        # Extract features
        feats = extract_features(tmp_path)
        if feats is None or getattr(feats, "ndim", 0) != 1:
            raise RuntimeError("Invalid features extracted")

        feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)

        # Transform + Predict
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
        # Remove temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

        gc.collect()
        log_memory("Cleanup complete")


# ---------------------------------------------------------
# Predict Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        return predict_from_bytes(data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()
        log_memory("Request cleanup done")
