from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import requests
from io import BytesIO

# hugging face model & scaler URLs
model_url = "https://huggingface.co/mahaqj/ml_assignment_3/resolve/main/best_model.joblib"
scaler_url = "https://huggingface.co/mahaqj/ml_assignment_3/resolve/main/scaler.joblib"

# download and load the model and scaler
model_bytes = BytesIO(requests.get(model_url).content)
model = joblib.load(model_bytes)

scaler_bytes = BytesIO(requests.get(scaler_url).content)
scaler = joblib.load(scaler_bytes)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        predicted_price = round(prediction * 100000, 2)  # scale back
        return jsonify({'predicted_price': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
