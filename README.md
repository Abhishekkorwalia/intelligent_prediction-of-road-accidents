# Intelligent Prediction of Road Accident Severity and Environmental Analytics

A comprehensive machine learning system for predicting road accident severity and analyzing environmental factors contributing to accidents.

## Features

- **Accident Severity Prediction**: Multiple ML models (Random Forest, XGBoost, Neural Network)
- **Environmental Analytics**: Analysis of weather, road conditions, time patterns
- **Interactive Dashboard**: Web-based interface for predictions and visualizations
- **Data Generation**: Synthetic dataset generator for testing
- **Model Evaluation**: Comprehensive metrics and comparison

## Project Structure

```
pro 1/
├── data/               # Generated datasets
├── models/             # Trained model files
├── src/                # Source code
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── environmental_analytics.py
│   └── visualization.py
├── notebooks/          # Jupyter notebooks
├── static/             # Web assets (CSS, JS)
├── templates/          # HTML templates
├── tests/              # Unit tests
├── app.py              # Flask web application
└── requirements.txt    # Dependencies

```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
 
### 1. Generate Dataset
```bash
python src/data_generator.py
```

### 2. Train Models
```bash
python src/model_training.py
```

### 3. Run Analytics
```bash
python src/environmental_analytics.py
```

### 4. Launch Web Application
```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## Models

- **Random Forest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high accuracy
- **Neural Network**: Deep learning for complex patterns

## Dataset Features

- Time features (hour, day of week, month)
- Weather conditions (temperature, precipitation, visibility)
- Road conditions (surface, lighting)
- Vehicle details (type, speed)
- Location data (urban/rural, traffic density)
- Driver factors (age, experience)

## Severity Levels

1. Minor: Property damage only
2. Moderate: Minor injuries
3. Severe: Serious injuries
4. Fatal: Fatal injuries

## License

MIT License
