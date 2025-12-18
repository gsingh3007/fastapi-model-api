import os, json, hashlib, time, warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt, hilbert, welch
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import networkx as nx
import antropy as ant

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
EEG_DIR = r"D:\CAPSTONE\DASPS_Database\Rawdataedf"
SAMPLE_RATE = 128                      # corrected for DASPS
EPOCH_LENGTH_SEC = 4                   # 4s epochs (512 samples)
N_EPOCHS = 30
TOP_K_FEATURES = 100
CACHE_FILE = "dasps_cached_features.npz"
CACHE_META = "dasps_cached_features.meta.json"

LABELS = {  "S01.edf":1,"S02.edf":1,"S03.edf":1,"S04.edf":1,"S05.edf":1,"S06.edf":1,
            "S07.edf":1,"S08.edf":0,"S09.edf":0,"S10.edf":0,"S11.edf":1,"S12.edf":1,
            "S13.edf":1,"S14.edf":0,"S15.edf":0,"S16.edf":1,"S17.edf":1,"S18.edf":0,
            "S19.edf":1,"S20.edf":1,"S21.edf":1,"S22.edf":1,"S23.edf":0 }


FREQ_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# ---------- CACHE UTILITIES ----------
def _file_inventory(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".edf")])
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
        "files": _file_inventory(EEG_DIR),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest(), payload

def _save_cache(X, y):
    sig, meta = _config_signature()
    np.savez_compressed(CACHE_FILE, X=X, y=y, signature=sig)
    with open(CACHE_META, "w") as f:
        json.dump({"signature": sig, "meta": meta}, f, indent=2)

def _load_cache():
    if not (os.path.exists(CACHE_FILE) and os.path.exists(CACHE_META)):
        return None, None
    try:
        data = np.load(CACHE_FILE, allow_pickle=False)
        with open(CACHE_META, "r") as f:
            meta = json.load(f)
        sig_current, _ = _config_signature()
        if str(data.get("signature", "")) != sig_current or meta.get("signature") != sig_current:
            return None, None
        return data["X"], data["y"]
    except:
        return None, None

# ---------- PREPROCESSING ----------
def preprocess_raw(fp):
    raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
    raw.set_eeg_reference("average", projection=False)
    raw.filter(1, 45, fir_design="firwin", verbose=False)  # upper cutoff lowered
    raw.notch_filter([50], verbose=False)                  # only 50 Hz notch

    # ICA to remove EOG/eye artifacts
    ica = mne.preprocessing.ICA(n_components=0.99, random_state=42, max_iter=800)
    ica.fit(raw)
    raw = ica.apply(raw)
    return raw

# ---------- FEATURE FUNCTIONS ----------
def compute_stats(sig):
    from scipy.stats import skew, kurtosis
    return [np.mean(sig), np.std(sig), skew(sig), kurtosis(sig)]

def hjorth(sig):
    d1, d2 = np.diff(sig), np.diff(np.diff(sig))
    v0, v1, v2 = np.var(sig), np.var(d1), np.var(d2)
    mobility = np.sqrt(v1/v0) if v0>0 else 0
    complexity = np.sqrt(v2/v1)/mobility if v1>0 and mobility>0 else 0
    return [v0, mobility, complexity]

def compute_entropy(sig):
    return [ant.sample_entropy(sig), ant.perm_entropy(sig, normalize=True), ant.svd_entropy(sig, normalize=True)]

def bandpass_filter(data, lo, hi, sfreq):
    b, a = butter(4, [lo/(0.5*sfreq), hi/(0.5*sfreq)], btype="band")
    return filtfilt(b, a, data, axis=1)

def pli_graph_features(epoch):
    ph = np.angle(hilbert(epoch, axis=1))
    n = ph.shape[0]
    pli = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            pli[i,j] = pli[j,i] = abs(np.mean(np.sign(np.sin(ph[i]-ph[j]))))
    G = nx.from_numpy_array(pli)
    deg = [d for _,d in G.degree(weight="weight")]
    clustering = np.mean(list(nx.clustering(G).values()))
    try:
        path_len = nx.average_shortest_path_length(G)
    except:
        path_len = 0
    return [pli[np.triu_indices(n,1)].mean(), np.mean(deg), nx.global_efficiency(G), clustering, path_len]

def epoch_signal(data, sfreq, sec=EPOCH_LENGTH_SEC, n=N_EPOCHS):
    sfreq = int(sfreq)
    step = int(sec * sfreq)
    max_samples = int(sec * sfreq * n)
    epochs = []
    for i in range(0, min(data.shape[1] - step, max_samples), step):
        epochs.append(data[:, i:i + step])
    return epochs

def extract_subject_features(raw):
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    epochs = epoch_signal(data, sfreq)

    all_epoch_feats = []
    for ep in epochs:
        feat = []
        for ch in ep:
            sig = ch
            feat += compute_stats(sig)
            feat += hjorth(sig)
            feat += compute_entropy(sig)
            f, P = welch(sig, sfreq, nperseg=int(sfreq*4))   # 4s windows
            for lo, hi in FREQ_BANDS.values():
                feat.append(np.mean(P[(f>=lo)&(f<=hi)]))
        for lo, hi in FREQ_BANDS.values():
            filtered = bandpass_filter(ep, lo, hi, sfreq)
            feat += pli_graph_features(filtered)
        all_epoch_feats.append(feat)

    all_epoch_feats = np.array(all_epoch_feats)
    return np.concatenate([
        np.mean(all_epoch_feats, axis=0),
        np.median(all_epoch_feats, axis=0),
        np.std(all_epoch_feats, axis=0),
    ])

# ---------- LOAD OR BUILD FEATURES ----------
def load_or_extract_features():
    X, y = _load_cache()
    if X is not None and y is not None:
        print(f"🔄 Using cached features: X={X.shape}, y={y.shape}")
        return X, y

    X, y = [], []
    files = sorted(f for f in os.listdir(EEG_DIR) if f.endswith(".edf"))
    for f in files:
        if f not in LABELS:
            continue
        raw = preprocess_raw(os.path.join(EEG_DIR, f))
        feats = extract_subject_features(raw)
        X.append(feats)
        y.append(LABELS[f])
        print(f"✅ {f} → {len(feats)} features")

    X, y = np.array(X), np.array(y)
    _save_cache(X, y)
    return X, y



def run_pipeline(X, y):
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K_FEATURES, X.shape[1]))
    X_sel = selector.fit_transform(X, y)

    base_models = {
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    }

    scalable_models = {
        "SVC": SVC(kernel="rbf", probability=True),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
    }

    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler()
    }

    results = []

    gb_params = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3],
    }

    xgb_params = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3],
        "subsample": [0.8],
        "colsample_bytree": [0.8, 1.0],
    }

    # enable sklearn caching
    from tempfile import mkdtemp
    from shutil import rmtree
    cache_dir = mkdtemp()

    try:
        # ================= Tune GB & XGB once before the loop =================
        gb_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params, cv=5, n_jobs=-1, scoring="accuracy"
        )
        gb_search.fit(X_sel, y)
        best_gb_params = gb_search.best_params_
        print(f"🔍 Best GB params: {best_gb_params} | Acc={gb_search.best_score_:.2f}%")

        xgb_search = GridSearchCV(
            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
            xgb_params, cv=5, n_jobs=-1, scoring="accuracy"
        )
        xgb_search.fit(X_sel, y)
        best_xgb_params = xgb_search.best_params_
        print(f"🔍 Best XGB params: {best_xgb_params} | Acc={xgb_search.best_score_:.2f}%")

        # ================= CV Loop for folds 3–6 =================
        for folds in range(3, 7):  # 3–6 folds
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            print(f"\n StratifiedKFold (n_splits={folds})\n")

            # ================= RandomForest =================
            for name, clf in base_models.items():
                pipe = Pipeline([("clf", clf)], memory=cache_dir)
                t0 = time.time()
                scores = cross_validate(
                    pipe, X_sel, y, cv=cv,
                    scoring = ["accuracy", "precision", "recall", "f1"],
                    n_jobs=-1
                )
                elapsed = time.time() - t0
                res = {
                    "folds": folds,
                    "model": name,
                    "accuracy": np.mean(scores["test_accuracy"]) * 100,
                    "precision": np.mean(scores["test_precision"]) * 100,
                    "recall": np.mean(scores["test_recall"]) * 100,
                    "f1": np.mean(scores["test_f1"]) * 100,
                    "time(s)": elapsed,
                }
                results.append(res)
                print(
                    f" {name:17} | Acc={res['accuracy']:.2f}% | Prec={res['precision']:.2f}% | "
                    f"Rec={res['recall']:.2f}% | F1={res['f1']:.2f}% | Time={elapsed:.1f}s"
                )

            # ================= GradientBoost (best params) =================
            gb_best = GradientBoostingClassifier(random_state=42, **best_gb_params)
            pipe = Pipeline([("clf", gb_best)], memory=cache_dir)
            t0 = time.time()
            scores = cross_validate(
                pipe, X_sel, y, cv=cv,
                scoring = ["accuracy", "precision", "recall", "f1"],
                n_jobs=-1
            )
            elapsed = time.time() - t0
            res = {
                "folds": folds,
                "model": "GradientBoost",
                "accuracy": np.mean(scores["test_accuracy"]) * 100,
                "precision": np.mean(scores["test_precision"]) * 100,
                "recall": np.mean(scores["test_recall"]) * 100,
                "f1": np.mean(scores["test_f1"]) * 100,
                "time(s)": elapsed
            }
            results.append(res)
            print(
                f" GradientBoost      | Acc={res['accuracy']:.2f}% | Prec={res['precision']:.2f}% "
                f"| Rec={res['recall']:.2f}% | F1={res['f1']:.2f}% | Time={elapsed:.1f}s"
            )

            # ================= XGBoost (best params) =================
            xgb_best = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                                     **best_xgb_params)
            pipe = Pipeline([("clf", xgb_best)], memory=cache_dir)
            t0 = time.time()
            scores = cross_validate(
                pipe, X_sel, y, cv=cv,
                scoring = ["accuracy", "precision", "recall", "f1"],
                n_jobs=-1
            )
            elapsed = time.time() - t0
            res = {
                "folds": folds,
                "model": "XGBoost",
                "accuracy": np.mean(scores["test_accuracy"]) * 100,
                "precision": np.mean(scores["test_precision"]) * 100,
                "recall": np.mean(scores["test_recall"]) * 100,
                "f1": np.mean(scores["test_f1"]) * 100,
                "time(s)": elapsed
            }
            results.append(res)
            print(
                f" XGBoost           | Acc={res['accuracy']:.2f}% | Prec={res['precision']:.2f}% "
                f"| Rec={res['recall']:.2f}% | F1={res['f1']:.2f}% | Time={elapsed:.1f}s"
            )

            # ================= Scaled Models (SVC + LR) =================
            for scaler_name, scaler in scalers.items():
                for name, clf in scalable_models.items():
                    pipe = Pipeline([(scaler_name, scaler), ("clf", clf)], memory=cache_dir)
                    model_name = f"{name}_{scaler_name}"
                    t0 = time.time()
                    scores = cross_validate(
                        pipe, X_sel, y, cv=cv,
                        scoring = ["accuracy", "precision", "recall", "f1"],
                        n_jobs=-1
                    )
                    elapsed = time.time() - t0
                    res = {
                        "folds": folds,
                        "model": model_name,
                        "accuracy": np.mean(scores["test_accuracy"]) * 100,
                        "precision": np.mean(scores["test_precision"]) * 100,
                        "recall": np.mean(scores["test_recall"]) * 100,
                        "f1": np.mean(scores["test_f1"]) * 100,
                        "time(s)": elapsed,
                    }
                    results.append(res)
                    print(
                        f" {model_name:17} | Acc={res['accuracy']:.2f}% | Prec={res['precision']:.2f}% | "
                        f"Rec={res['recall']:.2f}% | F1={res['f1']:.2f}% | Time={elapsed:.1f}s"
                    )

            # ================= Voting Classifiers =================
            for vote_type in ["hard", "soft"]:
                voting_clf = VotingClassifier(
                    estimators=[
                        ("lr", Pipeline([("scaler", StandardScaler()), ("lr", scalable_models["LogReg"])])),
                        ("svcs", Pipeline([("scaler", StandardScaler()), ("svcs", scalable_models["SVC"])])),
                        ("svcr", Pipeline([("scaler", RobustScaler()), ("svcr", scalable_models["SVC"])])),
                        ("rf", base_models["RandomForest"]),
                    ],
                    voting=vote_type,
                    weights = [2,2,2,1],
                    n_jobs=-1
                )
                t0 = time.time()
                scores = cross_validate(
                    voting_clf, X_sel, y, cv=cv,
                    scoring = ["accuracy", "precision", "recall", "f1"],
                    n_jobs=-1
                )
                elapsed = time.time() - t0
                res = {
                    "folds": folds,
                    "model": f"Voting_{vote_type}",
                    "accuracy": np.mean(scores["test_accuracy"]) * 100,
                    "precision": np.mean(scores["test_precision"]) * 100,
                    "recall": np.mean(scores["test_recall"]) * 100,
                    "f1": np.mean(scores["test_f1"]) * 100,
                    "time(s)": elapsed,
                }
                results.append(res)
                print(
                    f" Ensemble_{vote_type:9} | Acc={res['accuracy']:.2f}% | Prec={res['precision']:.2f}% | "
                    f"Rec={res['recall']:.2f}% | F1={res['f1']:.2f}% | Time={elapsed:.1f}s"
                )
    finally:
        rmtree(cache_dir)

    return pd.DataFrame(results)



# ---------- MAIN ----------
if __name__ == "__main__":
    X, y = load_or_extract_features()
    print(f"\n✅ Dataset built: X={X.shape}, y={y.shape}")
    results_df = run_pipeline(X, y)

    print("\n===== FULL RESULTS (all folds) =====")
    print(
        results_df.sort_values(["model", "folds"])
        .round({"accuracy": 2, "precision": 2, "recall": 2, "f1": 2, "time(s)": 2})
        .to_string(index=False)
    )

    print("\n===== BEST PER MODEL (by Accuracy) =====")
    print(
        results_df.loc[results_df.groupby("model")["accuracy"].idxmax()]
        .sort_values("accuracy", ascending=False)
        .round({"accuracy": 2, "precision": 2, "recall": 2, "f1": 2, "time(s)": 2})
        .to_string(index=False)
    )
