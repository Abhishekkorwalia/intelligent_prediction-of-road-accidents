"""
Environmental Analytics for Road Accidents
Analyzes weather, road conditions, time patterns and their impact on accidents
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class EnvironmentalAnalytics:
    """Analyze environmental factors in accidents"""
    
    def __init__(self):
        self.df = None
        self.severity_mapping = {
            0: 'Minor',
            1: 'Moderate',
            2: 'Severe',
            3: 'Fatal'
        }
    
    def load_data(self, filepath='data/accidents_data.csv'):
        """Load accident data"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['severity_label'] = self.df['severity'].map(self.severity_mapping)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def analyze_weather_impact(self):
        """Analyze weather conditions impact on severity"""
        print("\nAnalyzing weather impact...")
        
        # Create output directory
        os.makedirs('data/analytics', exist_ok=True)
        
        # Weather distribution by severity
        weather_severity = pd.crosstab(
            self.df['weather_condition'],
            self.df['severity_label'],
            normalize='index'
        ) * 100
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Stacked bar chart
        weather_severity.plot(kind='bar', stacked=True, ax=ax1, colormap='RdYlGn_r')
        ax1.set_title('Accident Severity Distribution by Weather Condition')
        ax1.set_xlabel('Weather Condition')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend(title='Severity', bbox_to_anchor=(1.05, 1))
        ax1.grid(axis='y', alpha=0.3)
        
        # Average severity by weather
        severity_score = self.df.groupby('weather_condition')['severity'].mean()
        severity_score.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Average Severity Score by Weather')
        ax2.set_xlabel('Weather Condition')
        ax2.set_ylabel('Average Severity')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/analytics/weather_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Weather impact analysis saved")
        return weather_severity
    
    def analyze_time_patterns(self):
        """Analyze temporal patterns"""
        print("\nAnalyzing time patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Hourly distribution
        hourly = self.df.groupby('hour')['accident_id'].count()
        hourly.plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Accidents by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Accidents')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Day of week
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly = self.df.groupby('day_of_week')['accident_id'].count()
        weekly.plot(kind='bar', ax=axes[0, 1], color='darkgreen')
        axes[0, 1].set_xticklabels(days, rotation=45)
        axes[0, 1].set_title('Accidents by Day of Week')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Number of Accidents')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Monthly distribution
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly = self.df.groupby('month')['accident_id'].count()
        monthly.plot(kind='bar', ax=axes[1, 0], color='purple')
        axes[1, 0].set_xticklabels(months, rotation=45)
        axes[1, 0].set_title('Accidents by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Accidents')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Severity by hour
        hourly_severity = self.df.groupby('hour')['severity'].mean()
        hourly_severity.plot(ax=axes[1, 1], color='red', linewidth=2, marker='o')
        axes[1, 1].set_title('Average Severity by Hour')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Average Severity')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/analytics/time_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Time patterns analysis saved")
    
    def analyze_road_conditions(self):
        """Analyze road surface and lighting impact"""
        print("\nAnalyzing road conditions...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Road surface impact
        road_severity = self.df.groupby('road_surface')['severity'].mean().sort_values(ascending=False)
        road_severity.plot(kind='barh', ax=ax1, color='orange')
        ax1.set_title('Average Severity by Road Surface')
        ax1.set_xlabel('Average Severity')
        ax1.set_ylabel('Road Surface')
        ax1.grid(axis='x', alpha=0.3)
        
        # Lighting conditions
        lighting_severity = self.df.groupby('lighting_condition')['severity'].mean().sort_values(ascending=False)
        lighting_severity.plot(kind='barh', ax=ax2, color='navy')
        ax2.set_title('Average Severity by Lighting Condition')
        ax2.set_xlabel('Average Severity')
        ax2.set_ylabel('Lighting Condition')
        ax2.grid(axis='x', alpha=0.3)
        
        # Road type distribution
        road_type_counts = self.df['road_type'].value_counts()
        ax3.pie(road_type_counts.values, labels=road_type_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('Set3'))
        ax3.set_title('Accident Distribution by Road Type')
        
        # Location type vs severity
        location_severity = pd.crosstab(
            self.df['location_type'],
            self.df['severity_label'],
            normalize='index'
        ) * 100
        location_severity.plot(kind='bar', ax=ax4, colormap='viridis')
        ax4.set_title('Severity Distribution by Location Type')
        ax4.set_xlabel('Location Type')
        ax4.set_ylabel('Percentage (%)')
        ax4.legend(title='Severity')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/analytics/road_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Road conditions analysis saved")
    
    def analyze_vehicle_factors(self):
        """Analyze vehicle-related factors"""
        print("\nAnalyzing vehicle factors...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Vehicle type impact
        vehicle_severity = self.df.groupby('vehicle_type')['severity'].mean().sort_values(ascending=False)
        vehicle_severity.plot(kind='barh', ax=axes[0, 0], color='teal')
        axes[0, 0].set_title('Average Severity by Vehicle Type')
        axes[0, 0].set_xlabel('Average Severity')
        axes[0, 0].set_ylabel('Vehicle Type')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Number of vehicles
        num_vehicles_severity = self.df.groupby('num_vehicles')['severity'].mean()
        num_vehicles_severity.plot(kind='bar', ax=axes[0, 1], color='maroon')
        axes[0, 1].set_title('Average Severity by Number of Vehicles')
        axes[0, 1].set_xlabel('Number of Vehicles')
        axes[0, 1].set_ylabel('Average Severity')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Speed analysis
        speed_bins = [0, 40, 60, 80, 100, 200]
        speed_labels = ['0-40', '40-60', '60-80', '80-100', '100+']
        self.df['speed_bin'] = pd.cut(self.df['vehicle_speed'], bins=speed_bins, labels=speed_labels)
        speed_severity = self.df.groupby('speed_bin')['severity'].mean()
        speed_severity.plot(kind='bar', ax=axes[1, 0], color='darkred')
        axes[1, 0].set_title('Average Severity by Vehicle Speed (km/h)')
        axes[1, 0].set_xlabel('Speed Range')
        axes[1, 0].set_ylabel('Average Severity')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Driver age
        age_bins = [16, 25, 35, 45, 55, 65, 100]
        age_labels = ['16-25', '25-35', '35-45', '45-55', '55-65', '65+']
        self.df['age_bin'] = pd.cut(self.df['driver_age'], bins=age_bins, labels=age_labels)
        age_severity = self.df.groupby('age_bin')['severity'].mean()
        age_severity.plot(kind='bar', ax=axes[1, 1], color='indigo')
        axes[1, 1].set_title('Average Severity by Driver Age')
        axes[1, 1].set_xlabel('Age Group')
        axes[1, 1].set_ylabel('Average Severity')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/analytics/vehicle_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Vehicle factors analysis saved")
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("\nCreating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accidents by Weather', 'Severity Heatmap (Hour vs Day)',
                           'Temperature vs Severity', 'Visibility Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'box'}]]
        )
        
        # 1. Weather distribution
        weather_counts = self.df['weather_condition'].value_counts()
        fig.add_trace(
            go.Bar(x=weather_counts.index, y=weather_counts.values,
                   marker_color='lightblue', name='Count'),
            row=1, col=1
        )
        
        # 2. Hour vs Day heatmap
        heatmap_data = pd.crosstab(self.df['hour'], self.df['day_of_week'])
        fig.add_trace(
            go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns,
                      y=heatmap_data.index, colorscale='YlOrRd'),
            row=1, col=2
        )
        
        # 3. Temperature vs Severity scatter
        sample = self.df.sample(min(1000, len(self.df)))
        for sev in range(4):
            sev_data = sample[sample['severity'] == sev]
            fig.add_trace(
                go.Scatter(x=sev_data['temperature'], y=sev_data['severity'],
                          mode='markers', name=self.severity_mapping[sev],
                          marker=dict(size=5, opacity=0.6)),
                row=2, col=1
            )
        
        # 4. Visibility by severity
        for sev in range(4):
            sev_data = self.df[self.df['severity'] == sev]
            fig.add_trace(
                go.Box(y=sev_data['visibility'], name=self.severity_mapping[sev]),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Road Accident Analytics Dashboard")
        fig.write_html('data/analytics/interactive_dashboard.html')
        
        print("Interactive dashboard saved to data/analytics/interactive_dashboard.html")
    
    def generate_summary_report(self):
        """Generate summary statistics report"""
        print("\nGenerating summary report...")
        
        report = {
            'Total Accidents': len(self.df),
            'Severity Distribution': self.df['severity_label'].value_counts().to_dict(),
            'Most Dangerous Weather': self.df.groupby('weather_condition')['severity'].mean().idxmax(),
            'Most Dangerous Hour': int(self.df.groupby('hour')['severity'].mean().idxmax()),
            'Most Dangerous Road Surface': self.df.groupby('road_surface')['severity'].mean().idxmax(),
            'Average Driver Age': round(self.df['driver_age'].mean(), 1),
            'Alcohol Involvement Rate': f"{(self.df['alcohol_involved'].sum() / len(self.df) * 100):.1f}%",
            'High-Risk Conditions': {
                'Weather': self.df.groupby('weather_condition')['severity'].mean().nlargest(3).to_dict(),
                'Road Surface': self.df.groupby('road_surface')['severity'].mean().nlargest(3).to_dict(),
                'Vehicle Type': self.df.groupby('vehicle_type')['severity'].mean().nlargest(3).to_dict()
            }
        }
        
        # Save report
        import json
        with open('data/analytics/summary_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("\nFull report saved to data/analytics/summary_report.json")
        
        return report


def main():
    """Main execution"""
    analytics = EnvironmentalAnalytics()
    
    # Load data
    analytics.load_data()
    
    # Run all analyses
    print("\n" + "="*50)
    print("ENVIRONMENTAL ANALYTICS")
    print("="*50)
    
    analytics.analyze_weather_impact()
    analytics.analyze_time_patterns()
    analytics.analyze_road_conditions()
    analytics.analyze_vehicle_factors()
    analytics.create_interactive_dashboard()
    analytics.generate_summary_report()
    
    print("\n" + "="*50)
    print("Analytics Complete!")
    print("="*50)
    print("Results saved to data/analytics/")


if __name__ == "__main__":
    main()
