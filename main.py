from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Tambahin ini
import tensorflow as tf
import numpy as np
from PIL import Image  # ✅ Tambahin ini juga

app = Flask(__name__)
CORS(app)  # ✅ Aktifin CORS untuk semua route

# Muat model
MODEL_PATH = "my_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def index():
    return jsonify({"message": "Model API aktif!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img = Image.open(request.files["file"].stream).convert("RGB").resize((224, 224))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    result = "Positive" if prediction[0][0] > 0.5 else "Negative"
    confidence = float(prediction[0][0])

    return jsonify({"result": result, "confidence": confidence})

application = app

if __name__ == "__main__":
    app.run(debug=True)
