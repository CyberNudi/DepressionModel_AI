import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from features import extract_features  # 確認這個模組能 import

app = Flask(__name__)
CORS(app)  # 允許跨域請求

MODEL_PATH = "model_weighted.pkl"

# 啟動時載入模型
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
    temp_path = "temp.wav"
    file.save(temp_path)

    try:
        # 特徵萃取
        features = extract_features(temp_path)
        features = features.reshape(1, -1)

        # 預測
        prediction = model.predict(features)[0]

        # 回傳 JSON，前端直接拿 prediction 當 PHQ-8 分數
        return jsonify({"phq8_score": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    # Render 預設 PORT 用環境變數
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
