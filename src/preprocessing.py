"""
Data Preprocessing Pipeline
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class AccidentDataPreprocessor:
    """Preprocess accident data for ML models"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_features = [
            'weather_condition', 'road_surface', 'road_type',
            'lighting_condition', 'vehicle_type', 'location_type'
        ]
        
    def load_data(self, filepath='data/accidents_data.csv'):
        """Load dataset from CSV"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def clean_data(self, df):
        """Clean and validate data"""
        print("Cleaning data...")
        
        # Create a copy
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Validate numerical ranges
        df_clean = df_clean[df_clean['temperature'] >= -50]
        df_clean = df_clean[df_clean['temperature'] <= 50]
        df_clean = df_clean[df_clean['driver_age'] >= 16]
        df_clean = df_clean[df_clean['driver_age'] <= 100]
        df_clean = df_clean[df_clean['vehicle_speed'] >= 0]
        df_clean = df_clean[df_clean['vehicle_speed'] <= 200]
        
        print(f"After cleaning: {len(df_clean)} records remaining")
        return df_clean
    
    def engineer_features(self, df):
        """Create new features from existing ones"""
        print("Engineering features...")
        
        df_eng = df.copy()
        
        # Speed-related features
        df_eng['speed_excess'] = df_eng['vehicle_speed'] - df_eng['speed_limit']
        df_eng['speed_ratio'] = df_eng['vehicle_speed'] / (df_eng['speed_limit'] + 1)
        df_eng['is_speeding'] = (df_eng['speed_excess'] > 0).astype(int)
        
        # Time-based features
        df_eng['is_rush_hour'] = df_eng['hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
        )
        df_eng['is_night'] = df_eng['hour'].apply(
            lambda x: 1 if x >= 22 or x <= 5 else 0
        )
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0
        )
        
        # Season from month
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df_eng['season'] = df_eng['month'].apply(get_season)
        
        # Weather severity
        weather_severity = {
            'Clear': 0, 'Cloudy': 1, 'Rain': 2, 
            'Fog': 3, 'Snow': 3, 'Storm': 4
        }
        df_eng['weather_severity'] = df_eng['weather_condition'].map(weather_severity)
        
        # Road risk score
        road_risk = {
            'Dry': 0, 'Wet': 1, 'Muddy': 2, 'Snowy': 3, 'Icy': 4
        }
        df_eng['road_risk'] = df_eng['road_surface'].map(road_risk)
        
        # Visibility category
        def visibility_category(vis):
            if vis < 100:
                return 'Very Poor'
            elif vis < 500:
                return 'Poor'
            elif vis < 1000:
                return 'Moderate'
            else:
                return 'Good'
        
        df_eng['visibility_category'] = df_eng['visibility'].apply(visibility_category)
        
        # Driver risk factors
        df_eng['young_driver'] = (df_eng['driver_age'] < 25).astype(int)
        df_eng['senior_driver'] = (df_eng['driver_age'] > 65).astype(int)
        df_eng['inexperienced_driver'] = (df_eng['driver_experience'] < 3).astype(int)
        
        # Vehicle risk
        high_risk_vehicles = ['Motorcycle', 'Bicycle']
        df_eng['high_risk_vehicle'] = df_eng['vehicle_type'].apply(
            lambda x: 1 if x in high_risk_vehicles else 0
        )
        
        # Multi-vehicle collision
        df_eng['multi_vehicle'] = (df_eng['num_vehicles'] > 1).astype(int)
        
        # Precipitation presence
        df_eng['has_precipitation'] = (df_eng['precipitation'] > 0).astype(int)
        
        print(f"Added {len(df_eng.columns) - len(df.columns)} new features")
        return df_eng
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        print("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Update categorical features to include new ones
        all_categorical = self.categorical_features + ['season', 'visibility_category']
        
        for col in all_categorical:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    if col in self.label_encoders:
                        # Handle unknown categories
                        le = self.label_encoders[col]
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare final feature set for modeling"""
        print("Preparing features...")
        
        # Features to drop
        drop_cols = ['accident_id', 'datetime', 'severity']
        
        # Keep only relevant features
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols].copy()
        y = df['severity'].copy() if 'severity' in df.columns else None
        
        # Store feature columns
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        return X, y
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        print("Scaling features...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def full_pipeline(self, filepath='data/accidents_data.csv', save_preprocessor=True):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*50)
        print("Running Full Preprocessing Pipeline")
        print("="*50 + "\n")
        
        # Load data
        df = self.load_data(filepath)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_eng = self.engineer_features(df_clean)
        
        # Encode categorical
        df_encoded = self.encode_categorical(df_eng, fit=True)
        
        # Prepare features
        X, y = self.prepare_features(df_encoded)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_scaled, y)
        
        # Save preprocessor
        if save_preprocessor:
            self.save_preprocessor()
        
        print("\n" + "="*50)
        print("Preprocessing Complete!")
        print("="*50)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessor"""
        # Clean
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_eng = self.engineer_features(df_clean)
        
        # Encode
        df_encoded = self.encode_categorical(df_eng, fit=False)
        
        # Prepare features
        X, y = self.prepare_features(df_encoded)
        
        # Scale
        X_scaled = self.scale_features(X, fit=False)
        
        return X_scaled, y
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor to file"""
        os.makedirs('models', exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load preprocessor from file"""
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        self.categorical_features = preprocessor_data['categorical_features']
        
        print(f"Preprocessor loaded from {filepath}")


def main():
    """Main execution"""
    preprocessor = AccidentDataPreprocessor()
    
    # Run full pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline()
    
    # Save processed data
    print("\nSaving processed data...")
    os.makedirs('data', exist_ok=True)
    
    np.save('data/X_train.npy', X_train.values)
    np.save('data/X_val.npy', X_val.values)
    np.save('data/X_test.npy', X_test.values)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_val.npy', y_val.values)
    np.save('data/y_test.npy', y_test.values)
    
    # Save feature names
    with open('data/feature_names.txt', 'w') as f:
        f.write('\n'.join(X_train.columns))
    
    print("Processed data saved to data/ directory")
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
