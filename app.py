"""
Flask Web Application for Road Accident Severity Prediction
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import json
import os

app = Flask(__name__)

# Load models and preprocessor
print("Loading models...")
try:
    rf_model = joblib.load('models/random_forest.pkl')
    xgb_model = joblib.load('models/xgboost.pkl')
    nn_model = keras.models.load_model('models/neural_network.h5')
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load models - {e}")
    rf_model = None
    xgb_model = None
    nn_model = None
    preprocessor = None

# Severity mapping
severity_mapping = {
    0: 'Minor',
    1: 'Moderate',
    2: 'Severe',
    3: 'Fatal'
}

# Feature options for form dropdowns
weather_options = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog', 'Storm']
road_surface_options = ['Dry', 'Wet', 'Icy', 'Snowy', 'Muddy']
road_type_options = ['Highway', 'Urban Road', 'Rural Road', 'Intersection', 'Roundabout']
lighting_options = ['Daylight', 'Dusk', 'Dark (Lit)', 'Dark (Unlit)']
vehicle_type_options = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Van', 'Bicycle']
location_type_options = ['Urban', 'Suburban', 'Rural']


@app.route('/')
def home():
    """Home page"""
    return render_template('indax.html')


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html',
                         weather_options=weather_options,
                         road_surface_options=road_surface_options,
                         road_type_options=road_type_options,
                         lighting_options=lighting_options,
                         vehicle_type_options=vehicle_type_options,
                         location_type_options=location_type_options)


@app.route('/analytics')
def analytics_page():
    """Analytics dashboard page"""
    # Load summary report if available
    summary = {}
    if os.path.exists('data/analytics/summary_report.json'):
        with open('data/analytics/summary_report.json', 'r') as f:
            summary = json.load(f)
    
    return render_template('analytics.html', summary=summary)


@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """API endpoint for predictions"""
    try:
        # Get form data
        data = request.json
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'datetime': [pd.Timestamp.now()],
            'hour': [int(data['hour'])],
            'day_of_week': [int(data['day_of_week'])],
            'month': [int(data['month'])],
            'weather_condition': [data['weather_condition']],
            'temperature': [float(data['temperature'])],
            'precipitation': [float(data['precipitation'])],
            'visibility': [float(data['visibility'])],
            'road_surface': [data['road_surface']],
            'road_type': [data['road_type']],
            'lighting_condition': [data['lighting_condition']],
            'speed_limit': [int(data['speed_limit'])],
            'vehicle_speed': [float(data['vehicle_speed'])],
            'vehicle_type': [data['vehicle_type']],
            'num_vehicles': [int(data['num_vehicles'])],
            'location_type': [data['location_type']],
            'traffic_density': [float(data['traffic_density'])],
            'driver_age': [float(data['driver_age'])],
            'driver_experience': [float(data['driver_experience'])],
            'alcohol_involved': [int(data['alcohol_involved'])],
            'accident_id': [0]
        })
        
        # Preprocess using the same pipeline
        from src.preprocessing import AccidentDataPreprocessor
        prep = AccidentDataPreprocessor()
        
        # Load preprocessor components
        if preprocessor:
            prep.label_encoders = preprocessor['label_encoders']
            prep.scaler = preprocessor['scaler']
            prep.feature_columns = preprocessor['feature_columns']
            prep.categorical_features = preprocessor['categorical_features']
        
        # Engineer features
        input_eng = prep.engineer_features(input_data)
        
        # Encode categorical
        input_encoded = prep.encode_categorical(input_eng, fit=False)
        
        # Prepare features
        X, _ = prep.prepare_features(input_encoded)
        
        # Scale
        X_scaled = prep.scale_features(X, fit=False)
        
        # Make predictions with all models
        predictions = {}
        
        if rf_model:
            rf_pred = rf_model.predict(X_scaled)[0]
            rf_proba = rf_model.predict_proba(X_scaled)[0]
            predictions['random_forest'] = {
                'severity': int(rf_pred),
                'severity_label': severity_mapping[rf_pred],
                'probabilities': {severity_mapping[i]: float(prob) for i, prob in enumerate(rf_proba)}
            }
        
        if xgb_model:
            xgb_pred = xgb_model.predict(X_scaled)[0]
            xgb_proba = xgb_model.predict_proba(X_scaled)[0]
            predictions['xgboost'] = {
                'severity': int(xgb_pred),
                'severity_label': severity_mapping[xgb_pred],
                'probabilities': {severity_mapping[i]: float(prob) for i, prob in enumerate(xgb_proba)}
            }
        
        if nn_model:
            nn_proba = nn_model.predict(X_scaled, verbose=0)[0]
            nn_pred = np.argmax(nn_proba)
            predictions['neural_network'] = {
                'severity': int(nn_pred),
                'severity_label': severity_mapping[nn_pred],
                'probabilities': {severity_mapping[i]: float(prob) for i, prob in enumerate(nn_proba)}
            }
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/stats')
def get_stats():
    """API endpoint for statistics"""
    try:
        # Load model comparison
        comparison = {}
        if os.path.exists('models/model_comparison.csv'):
            df = pd.read_csv('models/model_comparison.csv')
            comparison = df.to_dict('records')
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Road Accident Prediction System")
    print("="*50)
    print("\nAccess the application at: http://localhost:5000")
    print("\nEndpoints:")
    print("  - Home: /")
    print("  - Prediction: /predict")
    print("  - Analytics: /analytics")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
