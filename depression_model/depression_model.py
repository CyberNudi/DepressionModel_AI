from fastapi import FastAPI
app = FastAPI()
# train_weighted.py
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# --------- (1) 特徵函式 ----------
def extract_features(file_path, duration=60):
    y, sr = librosa.load(file_path, duration=duration)
    features = []

    # MFCC mean/std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Δ and ΔΔ
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))
    features.extend(np.mean(mfcc_delta2, axis=1))
    features.extend(np.std(mfcc_delta2, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # RMS
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Pitch (yin)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        features.append(np.mean(f0))
        features.append(np.std(f0))
        features.append(np.max(f0))
        features.append(np.min(f0))
    except Exception:
        features.extend([0, 0, 0, 0])

    # speech rate (簡易 proxy)
    speech_rate = np.sum(zcr) / max(duration, 1.0)
    features.append(speech_rate)

    # silence ratio (RMS < threshold)
    silence = np.sum(rms < 0.01) / max(rms.shape[1], 1)
    features.append(silence)

    return np.array(features)


# --------- (2) 權重函式與過採樣 ----------
def compute_weight(score, mode='exp', base=1.5, max_weight=100, linear_alpha=2.0):
    if score <= 4:
        w = 1.0
    else:
        if mode == 'exp':
            w = base ** (score - 4)
        elif mode == 'linear':
            w = 1.0 + linear_alpha * (score - 4)
        elif mode == 'log':
            w = 1.0 + np.log1p(score - 4)
        else:
            w = 1.0
    return min(w, max_weight)


def oversample_high(X, y, weights, threshold=10, factor=2):
    if factor <= 1:
        return X, y, weights
    mask = y >= threshold
    if np.sum(mask) == 0:
        return X, y, weights
    X_high, y_high, w_high = X[mask], y[mask], weights[mask]
    X_rep = np.vstack([X] + [X_high for _ in range(factor - 1)])
    y_rep = np.hstack([y] + [y_high for _ in range(factor - 1)])
    w_rep = np.hstack([weights] + [w_high for _ in range(factor - 1)])
    return X_rep, y_rep, w_rep


def oversample_low(X, y, weights, threshold=2, factor=3):
    if factor <= 1:
        return X, y, weights
    mask = y <= threshold
    if np.sum(mask) == 0:
        return X, y, weights
    X_low, y_low, w_low = X[mask], y[mask], weights[mask]
    X_rep = np.vstack([X] + [X_low for _ in range(factor - 1)])
    y_rep = np.hstack([y] + [y_low for _ in range(factor - 1)])
    w_rep = np.hstack([weights] + [w_low for _ in range(factor - 1)])
    return X_rep, y_rep, w_rep


# --------- (3) Spearman ----------
def spearman_corr(y_true, y_pred):
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(y_true, y_pred)
        return rho, p
    except Exception:
        r1 = np.argsort(np.argsort(y_true))
        r2 = np.argsort(np.argsort(y_pred))
        rho = np.corrcoef(r1, r2)[0, 1]
        return rho, None


# --------- (4) 訓練模型 ----------
def train_model(train_folder, label_file, model_out, weight_mode='exp', exp_base=1.2, max_weight=120,
                oversample=True, oversample_factor=5, oversample_threshold_high=8, oversample_threshold_low=3,
                model_type='rf'):
    df = pd.read_csv(label_file).set_index("Participant_ID")
    X_list, y_list = [], []

    for fname in os.listdir(train_folder):
        if not fname.lower().endswith(".wav"):
            continue
        try:
            pid = int(os.path.splitext(fname)[0].split("_")[0])
            if pid not in df.index:
                continue
            label = df.loc[pid, "PHQ_8Total"]
            feat = extract_features(os.path.join(train_folder, fname))
            X_list.append(feat)
            y_list.append(label)
        except Exception as e:
            print("⚠️ 處理失敗:", fname, e)

    if not X_list:
        print("⚠️ 沒有有效資料")
        return

    X, y = np.vstack(X_list), np.array(y_list)
    weights = np.array([compute_weight(v, mode=weight_mode, base=exp_base, max_weight=max_weight) for v in y])
    weights = weights / np.mean(weights)

    if oversample:
        X, y, weights = oversample_high(X, y, weights, oversample_threshold_high, oversample_factor)
        X, y, weights = oversample_low(X, y, weights, oversample_threshold_low, oversample_factor)

    stratify_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.3, random_state=42, stratify=stratify_bins
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42) if model_type == 'rf' else \
            GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)

    joblib.dump({
        "model": model,
        "weight_mode": weight_mode,
        "exp_base": exp_base,
        "max_weight": max_weight
    }, model_out)
    print(f"✅ 模型已存檔: {model_out}")

    y_pred = np.clip(model.predict(X_test), 0, 24)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rho, _ = spearman_corr(y_test, y_pred)
    print(f"MAE: {mae:.3f}, R2: {r2:.3f}, Spearman: {rho:.3f}")



    # --- 顯示散點圖 ---
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, y_pred)
    try:
        m, b = np.polyfit(y_test, y_pred, 1)
        xs = np.linspace(0, 24, 100)
        plt.plot(xs, m*xs + b, 'g--', label=f"trend: y={m:.2f}x+{b:.2f}")
    except Exception:
        pass
    plt.plot([0,24],[0,24],'r--', label="ideal")
    plt.xlabel("True PHQ_8Total")
    plt.ylabel("Predicted PHQ_8Total")
    plt.title(f"Model ({model_type}) True vs Pred")
    plt.legend()
    plt.show()   # ✅ 本地端顯示

# --------- (5) 預測分段函式 ----------
def phq8_stage(score: float) -> str:
    """依據 PHQ-8 分數分級"""
    if score < 5:
        return "輕微無憂"
    elif score < 10:
        return "輕度憂鬱"
    elif score < 15:
        return "中度憂鬱"
    else:
        return "中重／重度憂鬱"


# --------- (6) 單檔預測 ----------
def predict_audio(file_path, model_file="model_weighted.pkl"):
    data = joblib.load(model_file)
    model = data["model"]
    features = extract_features(file_path).reshape(1, -1)
    pred = np.clip(model.predict(features), 0, 24)
    score = float(pred[0])
    stage = phq8_stage(score)
    return score, stage



# --------- (7) FastAPI 部署區 ----------

from fastapi import UploadFile, File
import uvicorn

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    接收上傳音訊並輸出模型預測值與分級
    """
    # 暫存檔案
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 預測
    try:
        score, stage = predict_audio(temp_path, model_file="model_weighted.pkl")
        os.remove(temp_path)
        return {
            "predicted_PHQ8": round(score, 3),
            "stage": stage
        }
    except Exception as e:
        return {"error": str(e)}


# --- Render 啟動 ---
if __name__ == "__main__":
    mode = "predict"  # 訓練請在本地執行
    if mode == "train":
        train_model(
            train_folder="train_data",
            label_file="Detailed_PHQ8_Labels.csv",
            model_out="model_weighted.pkl",
            weight_mode='exp',
            exp_base=1.2,
            max_weight=120,
            oversample=True,
            oversample_factor=5,
            oversample_threshold_high=8,
            oversample_threshold_low=3,
            model_type='rf'
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=10000)


