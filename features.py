# features.py
import librosa
import numpy as np

# features.py 修正後的 FEATURE_NAMES

FEATURE_NAMES = (
    # MFCC 前三維 (3*mean + 3*std) = 6
    [f"mfcc_{i}_mean" for i in range(3)] +
    [f"mfcc_{i}_std" for i in range(3)] +

    # Delta (3*mean + 3*std) = 6
    [f"mfcc_d1_{i}_mean" for i in range(3)] +
    [f"mfcc_d1_{i}_std" for i in range(3)] +

    # Chroma (12個半音 * mean + 12個半音 * std) = 24
    [f"chroma_mean_{i}" for i in range(12)] +
    [f"chroma_std_{i}" for i in range(12)] +

    # Spectral Contrast (7個頻帶，但你可能之前算錯數量，改為對齊 4 個頻帶以符合 X_te=60)
    # 註：如果 X_te 確實是 60，代表 (18 + 24 + 10) = 52，剩下 8 個給 Contrast
    # 也就是說 Contrast 應該是 4 個 mean + 4 個 std
    [f"contrast_mean_{i}" for i in range(4)] +
    [f"contrast_std_{i}" for i in range(4)] +

    # 其他 (10 個)
    [
        "zcr_mean", "zcr_std",
        "rms_mean", "rms_std",
        "f0_mean", "f0_std", "f0_max", "f0_min",
        "speech_rate", "silence_ratio"
    ]
)

def extract_features(file_path, duration=60):
    y, sr = librosa.load(file_path, duration=duration)
    features = []

    # MFCC前三維
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    features.append(np.mean(mfcc[0, :]))
    features.append(np.std(mfcc[0, :]))
    features.append(np.mean(mfcc[1, :]))
    features.append(np.std(mfcc[1, :]))
    features.append(np.mean(mfcc[2, :]))
    features.append(np.std(mfcc[2, :]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # ZCR & RMS
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Fundamental frequency (F0)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        features.extend([np.mean(f0), np.std(f0), np.max(f0), np.min(f0)])
    except Exception:
        features.extend([0, 0, 0, 0])

    # Speech rate & silence ratio
    speech_rate = np.sum(zcr) / max(duration, 1.0)
    silence_ratio = np.sum(rms < 0.01) / max(rms.shape[1], 1)
    features.append(speech_rate)
    features.append(silence_ratio)

    return np.array(features, dtype=float)
