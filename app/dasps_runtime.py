# app/dasps_runtime.py
import numpy as np
import mne
from scipy.signal import butter, filtfilt, hilbert, welch
import networkx as nx
import antropy as ant
import warnings

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
SAMPLE_RATE = 128
EPOCH_LENGTH_SEC = 4
N_EPOCHS = 30

FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# ---------- PREPROCESS ----------
def preprocess_raw(fp):
    raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
    raw.set_eeg_reference("average", projection=False)
    raw.filter(1, 45, fir_design="firwin", verbose=False)
    raw.notch_filter([50], verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=0.99, random_state=42, max_iter=800
    )
    ica.fit(raw)
    raw = ica.apply(raw)

    return raw

# ---------- FEATURES ----------
def compute_stats(sig):
    from scipy.stats import skew, kurtosis
    return [np.mean(sig), np.std(sig), skew(sig), kurtosis(sig)]

def hjorth(sig):
    d1, d2 = np.diff(sig), np.diff(np.diff(sig))
    v0, v1, v2 = np.var(sig), np.var(d1), np.var(d2)
    mobility = np.sqrt(v1 / v0) if v0 > 0 else 0
    complexity = np.sqrt(v2 / v1) / mobility if v1 > 0 and mobility > 0 else 0
    return [v0, mobility, complexity]

def compute_entropy(sig):
    return [
        ant.sample_entropy(sig),
        ant.perm_entropy(sig, normalize=True),
        ant.svd_entropy(sig, normalize=True),
    ]

def bandpass_filter(data, lo, hi, sfreq):
    b, a = butter(4, [lo / (0.5 * sfreq), hi / (0.5 * sfreq)], btype="band")
    return filtfilt(b, a, data, axis=1)

def pli_graph_features(epoch):
    ph = np.angle(hilbert(epoch, axis=1))
    n = ph.shape[0]
    pli = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            pli[i, j] = pli[j, i] = abs(
                np.mean(np.sign(np.sin(ph[i] - ph[j])))
            )

    G = nx.from_numpy_array(pli)
    deg = [d for _, d in G.degree(weight="weight")]

    clustering = np.mean(list(nx.clustering(G).values()))
    try:
        path_len = nx.average_shortest_path_length(G)
    except:
        path_len = 0

    return [
        pli[np.triu_indices(n, 1)].mean(),
        np.mean(deg),
        nx.global_efficiency(G),
        clustering,
        path_len,
    ]

def epoch_signal(data, sfreq):
    step = int(EPOCH_LENGTH_SEC * sfreq)
    epochs = []

    for i in range(0, data.shape[1] - step, step):
        epochs.append(data[:, i : i + step])
        if len(epochs) >= N_EPOCHS:
            break

    return epochs

def extract_subject_features(raw):
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    epochs = epoch_signal(data, sfreq)

    all_epoch_feats = []

    for ep in epochs:
        feat = []

        for ch in ep:
            feat += compute_stats(ch)
            feat += hjorth(ch)
            feat += compute_entropy(ch)

            f, P = welch(ch, sfreq, nperseg=int(sfreq * 4))
            for lo, hi in FREQ_BANDS.values():
                feat.append(np.me
