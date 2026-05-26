from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app) # This allows your mobile app to talk to the server

# Load your trained model
model = joblib.load('road_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_road_condition():
    try:
        # 1. Get the 10 data points sent from the mobile app
        data = request.json['window']
        
        # 2. Separate the incoming data into individual X, Y, and Z lists
        x_vals = [point['x'] for point in data]
        y_vals = [point['y'] for point in data]
        z_vals = [point['z'] for point in data]
        
        # 3. Calculate the exact 4 features the Random Forest model needs
        accZ_std = np.std(z_vals)
        accZ_p2p = np.ptp(z_vals)  # np.ptp calculates Peak-to-Peak (Max - Min)
        accX_std = np.std(x_vals)
        accY_std = np.std(y_vals)
        
        # 4. Package them into a 2D array in the EXACT order the model expects
        input_data = np.array([accZ_std, accZ_p2p, accX_std, accY_std]).reshape(1, -1)
        
        # 5. Ask the Random Forest model for a prediction
        prediction = model.predict(input_data)[0]
        
        # 6. Send the result back to the phone
        return jsonify({
            'status': 'success',
            'label': str(prediction) 
        })
        
    except Exception as e:
        # If anything crashes, print it to the Render logs
        print(f"CRITICAL AI ERROR: {str(e)}", flush=True) 
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
