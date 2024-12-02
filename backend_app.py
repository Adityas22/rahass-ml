from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load('svm_model.pkl')

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Data dikirim dalam format JSON
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
