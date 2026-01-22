"""
Quickstart Script for Road Accident Prediction System
Runs all necessary steps in sequence
"""

import os
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def run_step(step_name, command):
    """Run a step and handle errors"""
    print_header(f"Step: {step_name}")
    print(f"Running: {command}\n")
    
    result = os.system(command)
    
    if result != 0:
        print(f"\n❌ Error in {step_name}")
        return False
    else:
        print(f"\n✅ {step_name} completed successfully")
        return True

def main():
    """Main quickstart execution"""
    print_header("Road Accident Prediction System - Quickstart")
    print("This script will:")
    print("1. Generate synthetic accident data")
    print("2. Preprocess the data")
    print("3. Train ML models")
    print("4. Run environmental analytics")
    print("\nThis may take 10-15 minutes depending on your system.\n")
    
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Step 1: Generate data
    if not run_step("Data Generation", "python src/data_generator.py"):
        return
    
    # Step 2: Preprocess data
    if not run_step("Data Preprocessing", "python src/preprocessing.py"):
        return
    
    # Step 3: Train models
    if not run_step("Model Training", "python src/model_training.py"):
        return
    
    # Step 4: Run analytics
    if not run_step("Environmental Analytics", "python src/environmental_analytics.py"):
        return
    
    # All done
    print_header("✅ QUICKSTART COMPLETE!")
    print("All components are ready!")
    print("\nNext steps:")
    print("1. Start the web application:")
    print("   python app.py")
    print("\n2. Open your browser to:")
    print("   http://localhost:5000")
    print("\n3. Explore the analytics:")
    print("   - Static plots: data/analytics/")
    print("   - Interactive: data/analytics/interactive_dashboard.html")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
