import time
import psutil
import joblib
import pandas as pd
import threading
from flask import Flask, request, jsonify
from prometheus_client import start_http_server

from prometheus_exporter import (
    APP_REQUEST_COUNT, APP_LATENCY, APP_EXCEPTION, 
    MODEL_PREDICTION, SYSTEM_CPU_USAGE, SYSTEM_RAM_USAGE
)

app = Flask(__name__)

try:
    model = joblib.load("model/model.pkl")
    print("Model berhasil di-load!")
except Exception as e:
    print(f"Gagal load model: {e}")
    model = None

def update_system_metrics():
    while True:
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
        SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
        time.sleep(5)

threading.Thread(target=update_system_metrics, daemon=True).start()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    APP_REQUEST_COUNT.inc()
    
    if model is None:
        return jsonify({'error': 'Model belum diload.'}), 500

    try:
        data = request.json['inputs']
        
        column_names = [
            'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
            'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
            'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption',
            'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
            'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
        ]
        
        df = pd.DataFrame(data, columns=column_names) 
        
        prediction = model.predict(df)
        result = int(prediction[0])
        
        MODEL_PREDICTION.labels(output_class=str(result)).inc()
        APP_LATENCY.observe(time.time() - start_time)
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        APP_EXCEPTION.inc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    start_http_server(8000) 
    app.run(host='0.0.0.0', port=5000)