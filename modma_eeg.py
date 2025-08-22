import os
import json
import hashlib
import time
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
EEG_DIR = r"D:\CAPSTONE\MODMA\EEG_128channels_resting_lanzhou_2015"
SAMPLE_RATE = 250
EPOCH_LENGTH_SEC = 2
N_EPOCHS = 35
TOP_K_FEATURES = 300
CACHE_FILE = "cached_features_modma.npz"
CACHE_META = "cached_features_modma.meta.json"
label = {
    "02010002":1, "02010004":1, "02010005":1, "02010006":1, "02010008":1,
    "02010010":1, "02010011":1, "02010012":1, "02010013":0, "02010015":1,
    "02010016":1, "02010018":0, "02010019":1, "02010021":1, "02010022":1,
    "02010023":1, "02010024":1, "02010025":1, "02010026":1, "02010028":0,
    "02010030":1, "02010033":1, "02010034":0, "02010036":1, "02020008":0,
    "02020010":0, "02020013":0, "02020014":0, "02020015":0, "02020016":0,
    "02020018":0, "02020019":0, "02020020":0, "02020021":0, "02020022":0,
    "02020023":0, "02020025":0, "02020026":0, "02020027":0, "02020029":0,
    "02030002":0, "02030003":0, "02030004":0, "02030005":0, "02030006":0,
    "02030007":0, "02030009":0, "02030014":0, "02030017":0, "02030018":0,
    "02030019":0, "02030020":0, "02030021":0
}

FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)}

# ---------- UTILS ----------
def _file_inventory(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".mat")])
    inv = []
    for f in files:
        p = os.path.join(path, f)
        try:
            st = os.stat(p)
            inv.append({"name": f, "size": st.st_size, "mtime": int(st.st_mtime)})
        except FileNotFoundError:
            pass
    return inv

def _config_signature():
    payload = {
        "SAMPLE_RATE": SAMPLE_RATE,
        "EPOCH_LENGTH_SEC": EPOCH_LENGTH_SEC,
        "N_EPOCHS": N_EPOCHS,
        "TOP_K_FEATURES": TOP_K_FEATURES,
        "FREQ_BANDS": FREQ_BANDS,
        "EEG_DIR": EEG_DIR,
        "files": _file_inventory(EEG_DIR),}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest(), payload

def _save_cache(X, y):
    sig, meta = _config_signature()
    np.savez_compressed(CACHE_FILE, X=X, y=y, signature=sig)
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump({"signature": sig, "meta": meta}, f, indent=2)

def _load_cache():
    if not (os.path.exists(CACHE_FILE) and os.path.exists(CACHE_META)):
        return None, None
    try:
        data = np.load(CACHE_FILE, allow_pickle=False)
        with open(CACHE_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sig_current, _ = _config_signature()
        if str(data.get("signature", "")) != sig_current or meta.get("signature") != sig_current:
            print("⚠️ Cache signature mismatch (config/files changed). Will rebuild.", flush=True)
            return None, None
        return data["X"], data["y"]
    except Exception as e:
        print(f"⚠️ Cache load failed ({e}). Will rebuild.", flush=True)
        return None, None

# ---------- HELPERS ----------
def bandpass_filter(data, lowcut, highcut, fs=250.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def epoch_split(data, fs=250, epoch_sec=2, n_epochs=35):
    epoch_len = epoch_sec * fs
    epochs = []
    for start in range(0, data.shape[1] - epoch_len, epoch_len):
        if len(epochs) >= n_epochs:
            break
        epochs.append(data[:, start:start + epoch_len])
    return np.array(epochs)

def compute_pli(epoch):
    n_ch = epoch.shape[0]
    analytic = hilbert(epoch, axis=1)
    phase = np.angle(analytic)
    pli = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dp = phase[i] - phase[j]
            pli[i, j] = pli[j, i] = np.abs(np.mean(np.sign(np.sin(dp))))
    return pli

def flatten_upper(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]

def graph_features(pli):
    G = nx.from_numpy_array(pli)
    T = nx.minimum_spanning_tree(G)
    strengths = [val for _, val in T.degree(weight='weight')]
    ecc = list(nx.eccentricity(T).values())
    return [np.mean(strengths), np.var(strengths),
        nx.global_efficiency(T), np.mean(ecc)]

def hjorth_params(sig):
    from numpy import diff, var, sqrt
    d1, d2 = diff(sig), diff(diff(sig))
    var0, var1, var2 = var(sig), var(d1), var(d2)
    mobility = sqrt(var1 / var0) if var0 > 0 else 0
    complexity = sqrt(var2 / var1) if var1 > 0 else 0
    return [var0, mobility, complexity]

def statistical_features(epoch):
    feats = []
    for ch in epoch:
        feats += [np.mean(ch), np.std(ch), np.min(ch), np.max(ch),
                  np.median(ch), np.percentile(ch, 25), np.percentile(ch, 75)]
        feats += hjorth_params(ch)
    return feats

def extract_features(file_path):
    mat = sio.loadmat(file_path)
    key = [k for k in mat if not k.startswith("__") and isinstance(mat[k], np.ndarray)][0]
    raw = mat[key]
    data = raw[1:] if raw.shape[0] == 129 else raw

    features = []
    for band, (low, high) in FREQ_BANDS.items():
        filtered = bandpass_filter(data, low, high, fs=SAMPLE_RATE)
        epochs = epoch_split(filtered, fs=SAMPLE_RATE, epoch_sec=EPOCH_LENGTH_SEC, n_epochs=N_EPOCHS)
        pli_matrices = np.array([compute_pli(ep) for ep in epochs])
        pli_avg = np.mean(pli_matrices, axis=0)
        features += list(flatten_upper(pli_avg))
        features += graph_features(pli_avg)

    epochs = epoch_split(data, fs=SAMPLE_RATE, epoch_sec=EPOCH_LENGTH_SEC, n_epochs=N_EPOCHS)
    stat_feats = np.mean([statistical_features(ep) for ep in epochs], axis=0)
    features += list(stat_feats)

    return np.array(features)

# ---------- DATA LOADING WITH CACHING ----------
def load_or_extract_features():
    X, Y = _load_cache()
    if X is not None and Y is not None:
        print(f"🔄 Using cached features: X={X.shape}, y={Y.shape}", flush=True)
        return X, Y

    print("⚡ Cache not found or invalid. Extracting features (this will take a while)...", flush=True)
    files = sorted(f for f in os.listdir(EEG_DIR) if f.endswith('.mat'))
    X, y = [], []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        fp = os.path.join(EEG_DIR, f)
        subj_id = os.path.splitext(f)[0]   # remove .mat extension
        subj_id = subj_id.split("_")[0]    # clean suffix if present

        if subj_id not in label:
            print(f"⚠️ Skipping {f}, subject_id {subj_id} not in labels dict", flush=True)
            continue

        feat = extract_features(fp)
        label = label[subj_id]  # <-- use explicit mapping
        X.append(feat)
        y.append(label)

    X, y = np.array(X), np.array(y)
    print(f"✅ Feature extraction done in {time.time()-t0:.2f}s. Shape: X={X.shape}, y={y.shape}", flush=True)

    print("💾 Writing cache...", flush=True)
    _save_cache(X, y)
    print(f"✅ Cache saved to {CACHE_FILE}", flush=True)
    return X, y

# ---------- PIPELINE ----------
def run_pipeline_once(X, y):
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K_FEATURES, X.shape[1]))
    X_sel = selector.fit_transform(X, y)

    scalers = {"StandardScaler": StandardScaler(), "RobustScaler": RobustScaler()}
    results = []

    for folds in range(4, 8):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        # SVC & LogisticRegression
        # for name, clf in [
        #     ("SVC", SVC(kernel='rbf', probability=True)),
        #     ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight='balanced'))
        # ]:
        #     for scaler_name, scaler in scalers.items():
        #         model_name = f"{name}_{scaler_name}_CV{folds}"
        #         model = make_pipeline(scaler, clf) if scaler else clf
        #         start = time.time()
        #         scores = cross_validate(model, X_sel, y, cv=cv,
        #                                 scoring=["accuracy", "precision", "recall", "f1"])
        #         elapsed = time.time() - start
        #         results.append({
        #             "model": model_name,
        #             "accuracy": float(np.mean(scores["test_accuracy"]) * 100),  # %
        #             "precision": float(np.mean(scores["test_precision"]) * 100), # %
        #             "recall": float(np.mean(scores["test_recall"])),
        #             "f1": float(np.mean(scores["test_f1"])),
        #             "time(s)": float(elapsed)
        #         })

        # RandomForest & XGBoost
        for est in range(200, 501, 50):
            for name, clf in [
                ("RandomForest", RandomForestClassifier(n_estimators=est, random_state=42)),
                # ("XGBoost", XGBClassifier(
                #     n_estimators=est, learning_rate=0.05,
                #     eval_metric='logloss', use_label_encoder=False,
                #     random_state=42
                # ))
            ]:
                model_name = f"{name}_{est}_CV{folds}"
                start = time.time()
                scores = cross_validate(clf, X_sel, y, cv=cv,
                                        scoring=["accuracy", "precision", "recall", "f1"])
                elapsed = time.time() - start
                results.append({
                    "model": model_name,
                    "accuracy": float(np.mean(scores["test_accuracy"]) * 100),  # %
                    "precision": float(np.mean(scores["test_precision"]) * 100), # %
                    "recall": float(np.mean(scores["test_recall"])),
                    "f1": float(np.mean(scores["test_f1"])),
                    "time(s)": float(elapsed)
                })

    return pd.DataFrame(results)

# ---------- PLOTTING ----------
def plot_metric(df, metric, ylabel, n_runs):
    plt.figure(figsize=(14, 6))
    plt.bar(df["model"], df[f"{metric}_mean"], yerr=df[f"{metric}_std"], capsize=5)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Mean ± Std over {n_runs} runs)")
    plt.tight_layout()
    plt.savefig(f"{metric}_mean_std.png", dpi=300)
    plt.close()

# ---------- MAIN ----------
if __name__ == "__main__":
    N_RUNS = 30
    all_runs = []

    # Load or extract features ONCE
    X, y = load_or_extract_features()

    for run in range(N_RUNS):
        print(f"\n========== Cycle {run+1}/{N_RUNS} started ==========", flush=True)
        t_run = time.time()
        run_df = run_pipeline_once(X, y)
        run_df["run"] = run + 1
        all_runs.append(run_df)

        # Top-5 by F1
        top5 = run_df.sort_values("accuracy", ascending=False).head(5)
        print("\nTop-5 models this cycle (by F1):", flush=True)
        print(top5[["model", "accuracy", "precision", "recall", "f1", "time(s)"]]
              .round({"accuracy":1, "precision":1, "recall":3, "f1":3, "time(s)":2})
              .to_string(index=False), flush=True)

        print(f"✅ Cycle {run+1}/{N_RUNS} done in {time.time()-t_run:.2f}s", flush=True)

    all_results = pd.concat(all_runs, ignore_index=True)

    # Save per-run raw results
    all_results.round({"accuracy":1, "precision":1, "recall":3, "f1":3, "time(s)":2})\
               .to_csv("all_runs_results.csv", index=False)
    print("\n💾 Saved per-run results to all_runs_results.csv", flush=True)

    # Aggregate Mean ± Std
    agg = all_results.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        time_mean=("time(s)", "mean"),
        time_std=("time(s)", "std")
    ).reset_index()

    # Save aggregated results
    agg.round({"accuracy_mean":1, "accuracy_std":1,
               "precision_mean":1, "precision_std":1,
               "recall_mean":3, "recall_std":3,
               "f1_mean":3, "f1_std":3,
               "time_mean":2, "time_std":2})\
       .to_csv("aggregated_mean_std.csv", index=False)
    print("💾 Saved aggregated results to aggregated_mean_std.csv", flush=True)

    # Final Aggregated Display
    print("\n📊 Final Aggregated Results (Mean ± Std):", flush=True)
    with pd.option_context('display.max_rows', None):
        print(agg.round({"accuracy_mean":1, "accuracy_std":1,
                         "precision_mean":1, "precision_std":1,
                         "recall_mean":3, "recall_std":3,
                         "f1_mean":3, "f1_std":3,
                         "time_mean":2, "time_std":2}).to_string(index=False), flush=True)

    # Plots
    for m in ["accuracy", "precision", "recall", "f1", "time"]:
        metric = m if m != "time" else "time"
        plot_metric(agg, metric, m.capitalize(), N_RUNS)

    print("\n✅ All plots saved (accuracy_mean_std.png, precision_mean_std.png, recall_mean_std.png, f1_mean_std.png, time_mean_std.png).", flush=True)