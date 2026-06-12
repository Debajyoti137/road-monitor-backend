from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import warnings

# Mute the Scikit-Learn warning so Render logs stay clean
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)
CORS(app)

# Load the NEW 5-feature model
model = joblib.load('road_model_v2.pkl')

@app.route('/predict', methods=['POST'])
def predict_road_condition():
    try:
        # 1. Get the payload from the mobile app
        payload = request.json
        data = payload.get('window', []) # The 10 hardware readings
        
        if not data:
             return jsonify({'status': 'error', 'message': 'No sensor data received'}), 400
             
        # Safely get speed, defaulting to 0.0 if the vehicle is stopped
        current_speed = float(payload.get('speed', 0.0)) 
        
        # 2. Separate the axes
        x_vals = [point['x'] for point in data]
        y_vals = [point['y'] for point in data]
        z_vals = [point['z'] for point in data]
        
        # 3. Calculate your 4 hardware features
        accZ_std = np.std(z_vals)
        accZ_p2p = np.ptp(z_vals)
        accX_std = np.std(x_vals)
        accY_std = np.std(y_vals)
        
        # 4. Combine into exactly 5 features: [4 Hardware + 1 GPS Speed]
        input_data = np.array([accZ_std, accZ_p2p, accX_std, accY_std, current_speed]).reshape(1, -1)
        
        # 5. Get Prediction
        prediction = model.predict(input_data)[0]
        
        # Print to Render logs so you can monitor real-time physics
        print(f"Speed: {current_speed}m/s | Z-Spike: {accZ_p2p:.2f} | AI: {prediction}", flush=True)
        
        return jsonify({
            'status': 'success',
            'label': str(prediction) 
        })
        
    except Exception as e:
        print(f"CRITICAL AI ERROR: {str(e)}", flush=True) 
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Gunicorn handles the port on Render, but this is here for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
