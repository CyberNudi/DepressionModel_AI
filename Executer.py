import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from features import extract_features

app = Flask(__name__)

MODEL_PATH = "model_weighted.pkl"

# 啟動時載入模型（避免每次請求都load）
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")


@app.route("/")
def home():
    return "Audio Depression Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # 暫存檔案
    temp_path = "temp.wav"
    file.save(temp_path)

    try:
        # 特徵萃取
        features = extract_features(temp_path)
        features = features.reshape(1, -1)

        # 預測
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)