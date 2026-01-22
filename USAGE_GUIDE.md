# Usage Guide - Road Accident Severity Prediction System

## Quick Start

### Option 1: Automated Setup (Recommended)
Run the quickstart script to set everything up automatically:

```bash
python quickstart.py
```

This will:
1. Generate synthetic accident data (10,000 records)
2. Preprocess and engineer features
3. Train all three ML models (Random Forest, XGBoost, Neural Network)
4. Generate environmental analytics

**Time Required:** 10-15 minutes

### Option 2: Manual Setup
Run each step individually:

```bash
# Step 1: Generate data
python src/data_generator.py

# Step 2: Preprocess data
python src/preprocessing.py

# Step 3: Train models
python src/model_training.py

# Step 4: Run analytics
python src/environmental_analytics.py
```

## Running the Web Application

After completing setup, start the Flask web server:

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

The web interface provides:
- **Home**: Overview and system features
- **Predict**: Interactive form to predict accident severity
- **Analytics**: Dashboard with statistics and visualizations

## Project Components

### 1. Data Generation (`src/data_generator.py`)

Generates synthetic accident data with realistic distributions.

**Features generated:**
- Time: hour, day of week, month
- Weather: condition, temperature, precipitation, visibility
- Road: surface, type, lighting
- Vehicle: type, speed, number of vehicles
- Location: urban/suburban/rural, traffic density
- Driver: age, experience, alcohol involvement
- Severity: Minor, Moderate, Severe, Fatal

**Usage:**
```python
from src.data_generator import AccidentDataGenerator

generator = AccidentDataGenerator(n_samples=10000)
df = generator.generate_dataset()
generator.save_dataset(df, 'accidents_data.csv')
```

### 2. Data Preprocessing (`src/preprocessing.py`)

Comprehensive preprocessing pipeline with:
- Data cleaning and validation
- Feature engineering (20+ new features)
- Categorical encoding
- Feature scaling
- Train/validation/test splitting

**Usage:**
```python
from src.preprocessing import AccidentDataPreprocessor

preprocessor = AccidentDataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline()
```

**Engineered Features:**
- Speed-related: excess speed, speed ratio, is_speeding
- Time-based: is_rush_hour, is_night, is_weekend, season
- Weather: weather_severity, has_precipitation
- Road: road_risk, visibility_category
- Driver: young_driver, senior_driver, inexperienced_driver
- Vehicle: high_risk_vehicle, multi_vehicle

### 3. Model Training (`src/model_training.py`)

Trains three ML models:

#### Random Forest
- 200 trees, max depth 20
- Class balancing for imbalanced data
- Feature importance analysis

#### XGBoost
- Gradient boosting with 200 estimators
- Learning rate 0.1
- Early stopping support

#### Neural Network
- 4-layer architecture (256→128→64→32 neurons)
- Batch normalization and dropout
- Adam optimizer with learning rate scheduling

**Usage:**
```python
from src.model_training import AccidentSeverityModels

trainer = AccidentSeverityModels()
X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train, y_train)

trainer.train_random_forest(X_train_balanced, y_train_balanced, X_val, y_val)
trainer.train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
trainer.train_neural_network(X_train_balanced, y_train_balanced, X_val, y_val)

trainer.evaluate_all_models(X_test, y_test)
trainer.save_models()
```

**Outputs:**
- Trained models: `models/*.pkl`, `models/*.h5`
- Confusion matrices: `models/plots/confusion_matrix_*.png`
- Model comparison: `models/model_comparison.csv`
- Training results: `models/training_results.json`

### 4. Environmental Analytics (`src/environmental_analytics.py`)

Comprehensive analysis of environmental factors:

**Analyses:**
1. **Weather Impact**: Severity distribution by weather conditions
2. **Time Patterns**: Accidents by hour, day, month; severity trends
3. **Road Conditions**: Surface, lighting, road type analysis
4. **Vehicle Factors**: Vehicle type, speed, number of vehicles
5. **Interactive Dashboard**: Plotly visualizations

**Usage:**
```python
from src.environmental_analytics import EnvironmentalAnalytics

analytics = EnvironmentalAnalytics()
analytics.load_data()

analytics.analyze_weather_impact()
analytics.analyze_time_patterns()
analytics.analyze_road_conditions()
analytics.analyze_vehicle_factors()
analytics.create_interactive_dashboard()
analytics.generate_summary_report()
```

**Outputs:**
- Static plots: `data/analytics/*.png`
- Interactive dashboard: `data/analytics/interactive_dashboard.html`
- Summary report: `data/analytics/summary_report.json`

### 5. Web Application (`app.py`)

Flask-based web interface with three main pages:

#### Home Page (`/`)
- System overview
- Feature highlights
- Severity level descriptions

#### Prediction Page (`/predict`)
- Interactive form with all input features
- Real-time predictions from all three models
- Probability distributions for each severity level

#### Analytics Page (`/analytics`)
- Summary statistics
- Severity distribution
- High-risk conditions

**API Endpoints:**
- `POST /api/predict`: Make predictions
- `GET /api/stats`: Get model statistics

## Making Predictions

### Via Web Interface
1. Navigate to `http://localhost:5000/predict`
2. Fill in the form with accident details
3. Click "Predict Severity"
4. View predictions from all three models

### Via Python Code
```python
import pandas as pd
import joblib
from tensorflow import keras
from src.preprocessing import AccidentDataPreprocessor

# Load models
rf_model = joblib.load('models/random_forest.pkl')
xgb_model = joblib.load('models/xgboost.pkl')
nn_model = keras.models.load_model('models/neural_network.h5')
preprocessor_data = joblib.load('models/preprocessor.pkl')

# Create sample data
sample = pd.DataFrame({
    'hour': [22], 'day_of_week': [5], 'month': [12],
    'weather_condition': ['Rain'], 'temperature': [5.0],
    'precipitation': [10.0], 'visibility': [200.0],
    'road_surface': ['Wet'], 'road_type': ['Highway'],
    'lighting_condition': ['Dark (Unlit)'],
    'speed_limit': [100], 'vehicle_speed': [120.0],
    'vehicle_type': ['Car'], 'num_vehicles': [2],
    'location_type': ['Rural'], 'traffic_density': [50.0],
    'driver_age': [22.0], 'driver_experience': [2.0],
    'alcohol_involved': [1], 'accident_id': [0],
    'datetime': [pd.Timestamp.now()]
})

# Preprocess
prep = AccidentDataPreprocessor()
prep.label_encoders = preprocessor_data['label_encoders']
prep.scaler = preprocessor_data['scaler']
prep.feature_columns = preprocessor_data['feature_columns']

sample_eng = prep.engineer_features(sample)
sample_enc = prep.encode_categorical(sample_eng, fit=False)
X, _ = prep.prepare_features(sample_enc)
X_scaled = prep.scale_features(X, fit=False)

# Predict
rf_pred = rf_model.predict(X_scaled)[0]
xgb_pred = xgb_model.predict(X_scaled)[0]
nn_pred = np.argmax(nn_model.predict(X_scaled), axis=1)[0]

print(f"Random Forest: {rf_pred}")
print(f"XGBoost: {xgb_pred}")
print(f"Neural Network: {nn_pred}")
```

## Jupyter Notebook

For interactive exploration, use the provided notebook:

```bash
jupyter notebook notebooks/example_usage.ipynb
```

The notebook includes:
- Data generation and exploration
- Visualization examples
- Model loading and prediction
- Feature importance analysis

## Customization

### Changing Dataset Size
Edit `n_samples` in data generator:
```python
generator = AccidentDataGenerator(n_samples=50000)  # Generate 50k records
```

### Adjusting Model Parameters
Edit hyperparameters in `src/model_training.py`:
```python
# Random Forest
model = RandomForestClassifier(
    n_estimators=300,  # More trees
    max_depth=30,      # Deeper trees
    ...
)

# XGBoost
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,  # Lower learning rate
    ...
)
```

### Adding New Features
Edit `engineer_features()` in `src/preprocessing.py`:
```python
def engineer_features(self, df):
    df_eng = df.copy()
    
    # Add your custom feature
    df_eng['my_feature'] = df_eng['col1'] * df_eng['col2']
    
    return df_eng
```

## Output Files

```
pro 1/
├── data/
│   ├── accidents_data.csv           # Generated dataset
│   ├── X_train.npy, y_train.npy     # Training data
│   ├── X_val.npy, y_val.npy         # Validation data
│   ├── X_test.npy, y_test.npy       # Test data
│   ├── feature_names.txt            # Feature list
│   └── analytics/
│       ├── weather_impact.png
│       ├── time_patterns.png
│       ├── road_conditions.png
│       ├── vehicle_factors.png
│       ├── interactive_dashboard.html
│       └── summary_report.json
├── models/
│   ├── random_forest.pkl            # RF model
│   ├── xgboost.pkl                  # XGB model
│   ├── neural_network.h5            # NN model
│   ├── preprocessor.pkl             # Preprocessor
│   ├── model_comparison.csv         # Performance comparison
│   ├── training_results.json        # Training metrics
│   └── plots/
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_xgboost.png
│       ├── confusion_matrix_neural_network.png
│       └── model_comparison.png
```

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### TensorFlow Warnings
These are usually harmless. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

### Memory Issues
Reduce dataset size or batch size:
```python
# Smaller dataset
generator = AccidentDataGenerator(n_samples=5000)

# Smaller batch size for NN
model.fit(..., batch_size=64)  # Instead of 128
```

### Port Already in Use
Change Flask port:
```python
app.run(debug=True, port=5001)  # Use different port
```

## Performance Tips

1. **GPU Acceleration** (for Neural Network):
   - Install `tensorflow-gpu` if you have CUDA-capable GPU
   - Training will be 5-10x faster

2. **Parallel Processing**:
   - Random Forest and XGBoost use `n_jobs=-1` for all CPU cores

3. **Caching**:
   - Preprocessed data is saved to `.npy` files
   - Models are saved after training
   - Reuse without retraining

## Additional Resources

- Documentation: This file
- Example notebook: `notebooks/example_usage.ipynb`
- Source code: `src/` directory
- Web interface: `app.py`, `templates/`, `static/`

## Contact & Support

For issues or questions about this project, refer to the README.md or inspect the source code in the `src/` directory.
