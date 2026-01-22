# intelligent_prediction-of-road-accidents
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
