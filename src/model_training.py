"""
Machine Learning Models for Accident Severity Prediction
Implements Random Forest, XGBoost, and Neural Network models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


class AccidentSeverityModels:
    """Train and evaluate multiple ML models for severity prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.severity_mapping = {
            0: 'Minor',
            1: 'Moderate',
            2: 'Severe',
            3: 'Fatal'
        }
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance"""
        print("Applying SMOTE for class imbalance...")
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Before SMOTE: {X_train.shape}")
        print(f"After SMOTE: {X_train_balanced.shape}")
        print(f"Class distribution: {np.bincount(y_train_balanced.astype(int))}")
        
        return X_train_balanced, y_train_balanced
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'feature_importance': feature_importance.tolist()
        }
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost")
        print("="*50)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=4,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'feature_importance': feature_importance.tolist()
        }
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        print("\n" + "="*50)
        print("Training Neural Network")
        print("="*50)
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=4)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=4)
        
        # Build model
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=50,
            batch_size=128,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        train_pred = np.argmax(model.predict(X_train), axis=1)
        val_pred = np.argmax(model.predict(X_val), axis=1)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")
        
        self.models['neural_network'] = model
        self.results['neural_network'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        
        return model
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n" + "="*50)
        print("Evaluating All Models on Test Set")
        print("="*50)
        
        for model_name in self.models.keys():
            print(f"\n{model_name.upper()}:")
            print("-" * 50)
            
            model = self.models[model_name]
            
            if model_name == 'neural_network':
                y_pred = np.argmax(model.predict(X_test), axis=1)
            else:
                y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            print(f"Accuracy: {acc:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=[self.severity_mapping[i] for i in range(4)]
            ))
            
            # Update results
            self.results[model_name].update({
                'test_accuracy': acc,
                'test_f1': f1,
                'test_precision': precision,
                'test_recall': recall
            })
            
            # Confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, model_name)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[self.severity_mapping[i] for i in range(4)],
            yticklabels=[self.severity_mapping[i] for i in range(4)]
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        os.makedirs('models/plots', exist_ok=True)
        plt.savefig(f'models/plots/confusion_matrix_{model_name}.png', dpi=300)
        plt.close()
        
        print(f"Confusion matrix saved to models/plots/confusion_matrix_{model_name}.png")
    
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        comparison = pd.DataFrame({
            'Model': [],
            'Test Accuracy': [],
            'Test F1-Score': [],
            'Test Precision': [],
            'Test Recall': []
        })
        
        for model_name, metrics in self.results.items():
            comparison = pd.concat([comparison, pd.DataFrame({
                'Model': [model_name],
                'Test Accuracy': [metrics['test_accuracy']],
                'Test F1-Score': [metrics['test_f1']],
                'Test Precision': [metrics['test_precision']],
                'Test Recall': [metrics['test_recall']]
            })], ignore_index=True)
        
        print(comparison.to_string(index=False))
        
        # Save comparison
        comparison.to_csv('models/model_comparison.csv', index=False)
        print("\nComparison saved to models/model_comparison.csv")
        
        # Visualization
        self.plot_model_comparison(comparison)
        
        return comparison
    
    def plot_model_comparison(self, comparison):
        """Plot model comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison))
        width = 0.2
        
        metrics = ['Test Accuracy', 'Test F1-Score', 'Test Precision', 'Test Recall']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, comparison[metric], width, label=metric, color=colors[i])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison['Model'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/plots/model_comparison.png', dpi=300)
        plt.close()
        
        print("Comparison plot saved to models/plots/model_comparison.png")
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        os.makedirs('models', exist_ok=True)
        
        # Save sklearn models
        if 'random_forest' in self.models:
            joblib.dump(self.models['random_forest'], 'models/random_forest.pkl')
            print("Random Forest saved")
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], 'models/xgboost.pkl')
            print("XGBoost saved")
        
        # Save Keras model
        if 'neural_network' in self.models:
            self.models['neural_network'].save('models/neural_network.h5')
            print("Neural Network saved")
        
        # Save results
        with open('models/training_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        print("Training results saved")


def main():
    """Main execution"""
    # Initialize
    trainer = AccidentSeverityModels()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train, y_train)
    
    # Train models
    trainer.train_random_forest(X_train_balanced, y_train_balanced, X_val, y_val)
    trainer.train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
    trainer.train_neural_network(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Evaluate on test set
    trainer.evaluate_all_models(X_test, y_test)
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*50)
    print("Model Training Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
