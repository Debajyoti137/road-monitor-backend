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
        
        # 2. Extract just the X, Y, Z values for the AI
        # (Modify this if your model needs specific features like standard deviation)
        features = []
        for point in data:
            features.extend([point['x'], point['y'], point['z']])
            
        # Convert to a 2D numpy array (1 sample, 30 features)
        input_data = np.array(features).reshape(1, -1)
        
        # 3. Ask the Random Forest model for a prediction
        prediction = model.predict(input_data)[0]
        
        # 4. Send the result back to the phone
        return jsonify({
            'status': 'success',
            'label': str(prediction) # Usually returns 'G', 'B', or 'P'
        })
        
    except Exception as e:
        # THIS IS THE MAGIC LINE:
        print(f"CRITICAL AI ERROR: {str(e)}", flush=True) 
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
