import os
import sys
import joblib
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

from app.dasps_runtime import preprocess_raw, extract_subject_features

MODEL_DIR = os.path.join(APP_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "dasps_svc_model.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "dasps_selector.pkl"))

def predict_dasps(edf_path: str):
    # 1. Read + preprocess EDF
    raw = preprocess_raw(edf_path)

    # 2. Extract features
    feats = extract_subject_features(raw)
    feats = np.array(feats).reshape(1, -1)

    # 3. Feature selection
    feats_sel = selector.transform(feats)

    # 4. Predict
    pred = int(model.predict(feats_sel)[0])
    prob = float(model.predict_proba(feats_sel)[0][pred])

    return {
        "model": "DASPS_14_SVC_Robust_v1",
        "prediction": "Anxious" if pred == 1 else "Not Anxious",
        "confidence": round(prob * 100, 2),
        "probabilities": {
            "not_anxious": round(model.predict_proba(feats_sel)[0][0] * 100, 2),
            "anxious": round(model.predict_proba(feats_sel)[0][1] * 100, 2),
        },
    }
