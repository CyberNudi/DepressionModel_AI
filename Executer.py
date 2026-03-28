import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from features import extract_features  # 確認這個模組能 import

app = Flask(__name__)

# 完整允許跨域，這裡允許所有網域
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = "model_weighted.pkl"

# 啟動時載入模型
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")


@app.route("/")
def home():
    return "Audio Depression Prediction API Running"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_api():
    # 處理預檢請求
    if request.method == "OPTIONS":
        response = jsonify({"message": "Preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

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

        # 回傳 JSON
        response = jsonify({"phq8_score": float(prediction)})
        response.headers.add("Access-Control-Allow-Origin", "*")  # 確保回應有 header
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
