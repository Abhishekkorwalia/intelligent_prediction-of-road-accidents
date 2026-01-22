"""
Synthetic Road Accident Data Generator
Generates realistic accident data with severity levels and environmental factors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


class AccidentDataGenerator:
    """Generate synthetic road accident data"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.severity_mapping = {
            0: 'Minor',
            1: 'Moderate', 
            2: 'Severe',
            3: 'Fatal'
        }
        
    def generate_dataset(self):
        """Generate complete accident dataset"""
        print(f"Generating {self.n_samples} accident records...")
        
        data = {
            'accident_id': range(1, self.n_samples + 1),
            'datetime': self._generate_datetime(),
            'hour': None,
            'day_of_week': None,
            'month': None,
            'weather_condition': self._generate_weather(),
            'temperature': self._generate_temperature(),
            'precipitation': self._generate_precipitation(),
            'visibility': self._generate_visibility(),
            'road_surface': self._generate_road_surface(),
            'road_type': self._generate_road_type(),
            'lighting_condition': self._generate_lighting(),
            'speed_limit': self._generate_speed_limit(),
            'vehicle_speed': None,
            'vehicle_type': self._generate_vehicle_type(),
            'num_vehicles': self._generate_num_vehicles(),
            'location_type': self._generate_location_type(),
            'traffic_density': self._generate_traffic_density(),
            'driver_age': self._generate_driver_age(),
            'driver_experience': None,
            'alcohol_involved': self._generate_alcohol(),
            'severity': None
        }
        
        df = pd.DataFrame(data)
        
        # Extract time features
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # Calculate vehicle speed based on speed limit
        df['vehicle_speed'] = df['speed_limit'] + np.random.normal(0, 15, self.n_samples)
        df['vehicle_speed'] = df['vehicle_speed'].clip(0, 150)
        
        # Calculate driver experience
        df['driver_experience'] = (df['driver_age'] - 18).clip(0, 50)
        df['driver_experience'] = df['driver_experience'] + np.random.randint(-2, 5, self.n_samples)
        df['driver_experience'] = df['driver_experience'].clip(0, 60)
        
        # Generate severity based on features
        df['severity'] = self._generate_severity(df)
        
        print(f"Dataset generated successfully!")
        print(f"Severity distribution:\n{df['severity'].value_counts().sort_index()}")
        
        return df
    
    def _generate_datetime(self):
        """Generate random datetime over past 3 years"""
        start_date = datetime.now() - timedelta(days=3*365)
        dates = []
        for _ in range(self.n_samples):
            random_days = random.randint(0, 3*365)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            dt = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
            dates.append(dt)
        return dates
    
    def _generate_weather(self):
        """Generate weather conditions"""
        weather_types = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog', 'Storm']
        weights = [0.4, 0.25, 0.15, 0.08, 0.07, 0.05]
        return np.random.choice(weather_types, size=self.n_samples, p=weights)
    
    def _generate_temperature(self):
        """Generate temperature in Celsius"""
        return np.random.normal(20, 15, self.n_samples).clip(-20, 45)
    
    def _generate_precipitation(self):
        """Generate precipitation in mm"""
        # 70% chance of no precipitation
        precip = np.random.exponential(2, self.n_samples)
        precip[np.random.random(self.n_samples) > 0.3] = 0
        return precip.clip(0, 50)
    
    def _generate_visibility(self):
        """Generate visibility in meters"""
        visibility = np.random.gamma(20, 50, self.n_samples)
        return visibility.clip(10, 10000)
    
    def _generate_road_surface(self):
        """Generate road surface conditions"""
        surfaces = ['Dry', 'Wet', 'Icy', 'Snowy', 'Muddy']
        weights = [0.50, 0.25, 0.10, 0.08, 0.07]
        return np.random.choice(surfaces, size=self.n_samples, p=weights)
    
    def _generate_road_type(self):
        """Generate road types"""
        road_types = ['Highway', 'Urban Road', 'Rural Road', 'Intersection', 'Roundabout']
        weights = [0.25, 0.30, 0.20, 0.15, 0.10]
        return np.random.choice(road_types, size=self.n_samples, p=weights)
    
    def _generate_lighting(self):
        """Generate lighting conditions"""
        lighting = ['Daylight', 'Dusk', 'Dark (Lit)', 'Dark (Unlit)']
        weights = [0.50, 0.15, 0.25, 0.10]
        return np.random.choice(lighting, size=self.n_samples, p=weights)
    
    def _generate_speed_limit(self):
        """Generate speed limits in km/h"""
        speed_limits = [30, 50, 60, 80, 100, 120]
        weights = [0.15, 0.25, 0.20, 0.20, 0.15, 0.05]
        return np.random.choice(speed_limits, size=self.n_samples, p=weights)
    
    def _generate_vehicle_type(self):
        """Generate vehicle types"""
        vehicles = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Van', 'Bicycle']
        weights = [0.50, 0.15, 0.12, 0.08, 0.10, 0.05]
        return np.random.choice(vehicles, size=self.n_samples, p=weights)
    
    def _generate_num_vehicles(self):
        """Generate number of vehicles involved"""
        return np.random.choice([1, 2, 3, 4, 5], size=self.n_samples, p=[0.35, 0.40, 0.15, 0.07, 0.03])
    
    def _generate_location_type(self):
        """Generate location types"""
        locations = ['Urban', 'Suburban', 'Rural']
        weights = [0.45, 0.30, 0.25]
        return np.random.choice(locations, size=self.n_samples, p=weights)
    
    def _generate_traffic_density(self):
        """Generate traffic density (vehicles per hour)"""
        return np.random.gamma(5, 50, self.n_samples).clip(10, 2000)
    
    def _generate_driver_age(self):
        """Generate driver age"""
        # Bimodal distribution: young and middle-aged drivers
        young = np.random.normal(25, 5, self.n_samples // 2)
        middle = np.random.normal(45, 15, self.n_samples - self.n_samples // 2)
        ages = np.concatenate([young, middle])
        np.random.shuffle(ages)
        return ages.clip(16, 90)
    
    def _generate_alcohol(self):
        """Generate alcohol involvement"""
        return np.random.choice([0, 1], size=self.n_samples, p=[0.85, 0.15])
    
    def _generate_severity(self, df):
        """Generate severity based on features (rule-based with randomness)"""
        severity = np.zeros(self.n_samples, dtype=int)
        
        for i in range(self.n_samples):
            risk_score = 0
            
            # Weather risk
            if df.loc[i, 'weather_condition'] in ['Storm', 'Fog']:
                risk_score += 3
            elif df.loc[i, 'weather_condition'] in ['Rain', 'Snow']:
                risk_score += 2
            
            # Visibility risk
            if df.loc[i, 'visibility'] < 100:
                risk_score += 3
            elif df.loc[i, 'visibility'] < 500:
                risk_score += 1
            
            # Road surface risk
            if df.loc[i, 'road_surface'] in ['Icy', 'Snowy']:
                risk_score += 3
            elif df.loc[i, 'road_surface'] == 'Wet':
                risk_score += 1
            
            # Speed risk
            speed_diff = df.loc[i, 'vehicle_speed'] - df.loc[i, 'speed_limit']
            if speed_diff > 30:
                risk_score += 3
            elif speed_diff > 15:
                risk_score += 2
            elif speed_diff > 5:
                risk_score += 1
            
            # Time risk (night time)
            if df.loc[i, 'hour'] >= 22 or df.loc[i, 'hour'] <= 5:
                risk_score += 1
            
            # Lighting risk
            if df.loc[i, 'lighting_condition'] == 'Dark (Unlit)':
                risk_score += 2
            
            # Vehicle type risk
            if df.loc[i, 'vehicle_type'] in ['Motorcycle', 'Bicycle']:
                risk_score += 2
            
            # Multiple vehicles
            if df.loc[i, 'num_vehicles'] >= 3:
                risk_score += 2
            
            # Alcohol risk
            if df.loc[i, 'alcohol_involved'] == 1:
                risk_score += 4
            
            # Driver experience risk
            if df.loc[i, 'driver_experience'] < 2:
                risk_score += 2
            elif df.loc[i, 'driver_experience'] < 5:
                risk_score += 1
            
            # Map risk score to severity with randomness
            risk_score += np.random.randint(-2, 3)
            
            if risk_score <= 3:
                severity[i] = 0  # Minor
            elif risk_score <= 7:
                severity[i] = 1  # Moderate
            elif risk_score <= 12:
                severity[i] = 2  # Severe
            else:
                severity[i] = 3  # Fatal
        
        return severity
    
    def save_dataset(self, df, filename='accidents_data.csv'):
        """Save dataset to CSV"""
        filepath = f'data/{filename}'
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return filepath


def main():
    """Main execution"""
    # Generate dataset
    generator = AccidentDataGenerator(n_samples=10000)
    df = generator.generate_dataset()
    
    # Save to CSV
    generator.save_dataset(df)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"\nSeverity distribution:")
    severity_counts = df['severity'].value_counts().sort_index()
    for sev, count in severity_counts.items():
        print(f"  {generator.severity_mapping[sev]}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nFeature summary:")
    print(df.describe())
    
    print(f"\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()
