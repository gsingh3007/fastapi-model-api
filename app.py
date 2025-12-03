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


# -------------------------------------------------------------
#                    LOGGING + MEMORY DEBUG
# -------------------------------------------------------------
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
    except Exception:
        pass


# -------------------------------------------------------------
#                    PATHS & MODEL CONFIG
# -------------------------------------------------------------
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "exported_model")

# Load labels & metadata
with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# -------------------------------------------------------------
#                     FASTAPI APP + CORS
# -------------------------------------------------------------
app = FastAPI(title="EEG Anxiety Detection API")

origins = [
    "http://localhost:9002",
    "http://localhost:3000",
    "http://127.0.0.1:9002",
    "http://127.0.0.1:3000",

    # Firebase Hosting (production)
    "https://anxiocheck-ypipu.web.app",
    "https://anxiocheck-ypipu.firebaseapp.com",

    # Firebase preview URLs
    "https://*.web.app",
    "https://*.firebaseapp.com",

    # Firebase Studio editor
    "https://studio.firebase.google.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
#               MEMORY-MAPPED MODEL LOADING (LAZY)
# -------------------------------------------------------------
def _pick_existing(*candidates: str) -> str:
    """Returns the first filename that exists inside MODEL_DIR."""
    for name in candidates:
        fp = os.path.join(MODEL_DIR, name)
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"Missing model files: {candidates}")


@lru_cache(maxsize=1)
def get_models():
    """Load selector + ensemble model once (cached + memory-mapped)."""
    selector_path = _pick_existing("selector_compressed.joblib", "selector.joblib")
    ensemble_path = _pick_existing("ensemble_model_compressed.joblib",
                                   "ensemble_model.joblib")

    selector = load(selector_path, mmap_mode="r")
    ensemble = load(ensemble_path, mmap_mode="r")

    log_memory("Models loaded (selector + ensemble)")
    return selector, ensemble


# -------------------------------------------------------------
#                 BASIC ROOT + HEALTH ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "EEG Anxiety Detection API is live!"}


@app.get("/health")
async def health():
    try:
        get_models()
        log_memory("HealthCheck OK")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
#                        PREDICTION CORE
# -------------------------------------------------------------
def predict_from_bytes(file_bytes: bytes):
    tmp_path = None
    feats = None
    Xsel = None
    probs = None

    try:
        # 1. Save temp file
        with NamedTemporaryFile(suffix=".mat", delete=False, dir=ROOT) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        log_memory("File saved for extraction")

        # 2. Extract features
        feats = extract_features(tmp_path)
        log_memory("Features extracted")

        if feats is None or getattr(feats, "ndim", 0) != 1:
            raise RuntimeError("Invalid features extracted")

        # Convert to float32 to reduce RAM footprint
        feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        log_memory("Converted to float32")

        # 3. Load models
        selector, ensemble = get_models()

        # 4. Transform
        Xsel = selector.transform(feats)
        log_memory("Selector transform done")

        # 5. Predict
        probs = ensemble.predict_proba(Xsel)[0]
        log_memory("Prediction done")

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
        # Cleanup temporary file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        # Cleanup arrays
        del feats, Xsel, probs
        gc.collect()
        log_memory("Cleanup complete")


# -------------------------------------------------------------
#                     PREDICT ENDPOINT
# -------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = None
    try:
        contents = await file.read()
        log_memory("File received")
        return predict_from_bytes(contents)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
    # Safely delete temporary file
    try:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    # Safely delete variables
    for var in ["feats", "Xsel", "probs"]:
        if var in locals():
            try:
                del locals()[var]
            except:
                pass

    gc.collect()
    log_memory("After cleanup")
