import os
import gc
import json
import numpy as np
import sys
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import psutil
import logging

# ---------------------------------------------------------
# PATH FIX (important for app/ structure)
# ---------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from modma_eeg import extract_features               # MODMA
from app.dasps_predict import predict_dasps          # DASPS

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
# Paths & Config (MODMA)
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
app = FastAPI(title="AnxioCheck EEG API")

origins = [
    "https://anxiocheck-ypipu.web.app",
    "https://anxiocheck-ypipu.firebaseapp.com",
    "https://api.anxiocheck.online",
    "https://anxiocheck.online",
    "https://www.anxiocheck.online",
    "*",  # ⚠ remove later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load MODMA models ONCE at startup
# ---------------------------------------------------------
@app.on_event("startup")
def load_models_once():
    global selector, ensemble

    logger.info("Loading MODMA models from disk...")
    selector = load(SELECTOR_PATH)
    ensemble = load(ENSEMBLE_PATH)

    log_memory("MODMA models loaded")
    logger.info("MODMA models ready.")


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True}


# ---------------------------------------------------------
# MODMA prediction logic (128-channel)
# ---------------------------------------------------------
def predict_modma_from_bytes(data: bytes):
    tmp_path = None

    try:
        with NamedTemporaryFile(suffix=".mat", delete=False, dir=ROOT) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        log_memory("MODMA temp file saved")

        feats = extract_features(tmp_path)
        if feats is None or getattr(feats, "ndim", 0) != 1:
            raise RuntimeError("Invalid MODMA features")

        feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)

        Xsel = selector.transform(feats)
        probs = ensemble.predict_proba(Xsel)[0]

        idx = int(np.argmax(probs))
        labels = CONFIG.get("labels", {"0": "not_anxious", "1": "anxious"})
        label = labels.get(str(idx), str(idx))

        return {
            "model": "MODMA_128_Ensemble_v1",
            "prediction": label,
            "confidence": round(float(probs[idx]) * 100, 2),
            "probabilities": {
                labels.get("0", "0"): round(float(probs[0]) * 100, 2),
                labels.get("1", "1"): round(float(probs[1]) * 100, 2),
            },
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()
        log_memory("MODMA cleanup complete")


# ---------------------------------------------------------
# Unified Predict Endpoint (14 + 128)
# ---------------------------------------------------------
@app.options("/predict")
async def predict_options():
    return {}


@app.post("/predict")
async def predict(
    channel_type: str = Form(...),   # "14" or "128"
    file: UploadFile = File(...)
):
    try:
        data = await file.read()

        # -------------------------------
        # MODMA (128-channel, .mat)
        # -------------------------------
        if channel_type == "128":
            return predict_modma_from_bytes(data)

        # -------------------------------
        # DASPS (14-channel, .edf)
        # -------------------------------
        elif channel_type == "14":
            tmp_path = None
            try:
                with NamedTemporaryFile(suffix=".edf", delete=False, dir=ROOT) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    tmp_path = tmp.name

                return predict_dasps(tmp_path)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid channel_type. Use '14' or '128'."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        gc.collect()
        log_memory("Request cleanup done")
