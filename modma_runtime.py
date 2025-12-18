import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert
import networkx as nx

# ---------- CONFIG ----------
SAMPLE_RATE = 250
EPOCH_LENGTH_SEC = 2
N_EPOCHS = 35

FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# ---------- HELPERS ----------
def bandpass_filter(data, lowcut, highcut, fs=250.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
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
    strengths = [val for _, val in T.degree(weight="weight")]
    ecc = list(nx.eccentricity(T).values())
    return [
        np.mean(strengths),
        np.var(strengths),
        nx.global_efficiency(T),
        np.mean(ecc),
    ]

def hjorth_params(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var0 = np.var(sig)
    var1 = np.var(d1)
    var2 = np.var(d2)
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(var2 / var1) if var1 > 0 else 0
    return [var0, mobility, complexity]

def statistical_features(epoch):
    feats = []
    for ch in epoch:
        feats += [
            np.mean(ch),
            np.std(ch),
            np.min(ch),
            np.max(ch),
            np.median(ch),
            np.percentile(ch, 25),
            np.percentile(ch, 75),
        ]
        feats += hjorth_params(ch)
    return feats

# ---------- PUBLIC RUNTIME FUNCTION ----------
def extract_features(file_path):
    mat = sio.loadmat(file_path)
    key = [k for k in mat if not k.startswith("__") and isinstance(mat[k], np.ndarray)][0]
    raw = mat[key]
    data = raw[1:] if raw.shape[0] == 129 else raw

    features = []

    for low, high in FREQ_BANDS.values():
        filtered = bandpass_filter(data, low, high, fs=SAMPLE_RATE)
        epochs = epoch_split(filtered)
        pli_avg = np.mean([compute_pli(ep) for ep in epochs], axis=0)
        features += list(flatten_upper(pli_avg))
        features += graph_features(pli_avg)

    epochs = epoch_split(data)
    stat_feats = np.mean([statistical_features(ep) for ep in epochs], axis=0)
    features += list(stat_feats)

    return np.array(features, dtype=np.float32)
