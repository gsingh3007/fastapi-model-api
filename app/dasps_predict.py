import sys
import os
import joblib
import numpy as np
import mne

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

from app.dasps_runtime import preprocess_raw, extract_subject_features

MODEL_DIR = os.path.join(APP_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "dasps_svc_model.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "dasps_selector.pkl"))

def predict_dasps(edf_path):
    raw = preprocess_raw(edf_path)

    feats = extract_subject_features(raw)
    feats = np.array(feats).reshape(1, -1)

    feats_sel = selector.transform(feats)

    pred = model.predict(feats_sel)[0]
    prob = model.predict_proba(feats_sel)[0][1]

    return {
        "model": "DASPS_14_SVC_Robust_v1",
        "prediction": "Anxious" if pred == 1 else "Not Anxious",
        "confidence": round(float(prob), 3),
    }
