import os, json, argparse
import numpy as np
from joblib import load
from modma_eeg import extract_features  # make sure modma_eeg.py is in same folder

# Paths
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "exported_model")

# Load config + artifacts
with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

selector = load(os.path.join(MODEL_DIR, "selector.joblib"))
ensemble = load(os.path.join(MODEL_DIR, "ensemble_model.joblib"))

def predict_from_file(file_path: str):
    feats = extract_features(file_path)
    if feats is None or getattr(feats, "ndim", 0) != 1:
        raise RuntimeError(f"Unexpected feature shape: {None if feats is None else getattr(feats,'shape',None)}")

    # Apply feature selector
    Xsel = selector.transform(np.asarray(feats, float).reshape(1, -1))

    # Predict with ensemble
    probs = ensemble.predict_proba(Xsel)[0]
    idx = int(np.argmax(probs))

    labels = CONFIG.get("labels", {"0": "not_anxious", "1": "anxious"})
    label = labels.get(str(idx), str(idx))

    return {
    "label": label,
    "confidence": round(float(probs[idx]) * 100, 2),
    "probabilities": {
        labels.get("0","0"): round(float(probs[0]) * 100, 6),
        labels.get("1","1"): round(float(probs[1]) * 100, 6),
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="Path to one sample EEG .mat file")
    args = parser.parse_args()
    result = predict_from_file(args.sample)
    result = predict_from_file(args.sample)

print("\n================= PREDICTION RESULT =================")
print(f" Prediction : {result['label'].upper()} ({result['confidence']}% confidence)")
print(" Probabilities:")
for cls, prob in result["probabilities"].items():
    print(f"   - {cls}: {prob}%")
print("=====================================================\n")