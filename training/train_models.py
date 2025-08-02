
"""
AI Risk Prediction Models Training Script

This script trains machine learning models for predicting:
1. Critical Thinking Risk Score
2. AI Overdependence Risk
3. Incorrect Information Risk Score
4. Privacy Concern Level
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AIRiskPredictor:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.model_performance = {}
        
    def load_data(self, filepath='data/clean/ai_cleaned.csv'):
        """Load cleaned data"""
        print("Loading cleaned data...")
        self.df = pd.read_csv(filepath)
        
      
        self.target_cols = {
            'critical_thinking': 'critical_thinking_risk_score',
            'ai_overdependence': 'ai_overdependence_risk',
            'incorrect_info': 'incorrect_info_risk_score',
            'privacy_concern': 'privacy_concern_level'
        }
        
     
        exclude_cols = list(self.target_cols.values()) + ['user_id']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        print(f"Features: {len(self.feature_cols)}")
        return self.df
    
    def prepare_data(self, target_name):
        """Prepare features and target for a specific prediction task"""
        X = self.df[self.feature_cols]
        y = self.df[self.target_cols[target_name]]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, target_name):
        """Train multiple models for a specific target"""
        print(f"\n{'='*50}")
        print(f"Training models for {target_name.replace('_', ' ').title()}")
        print('='*50)
        
        X_train, X_test, y_train, y_test = self.prepare_data(target_name)
        
        # Define models to try
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Ridge Regression': Ridge(),
            'Linear Regression': LinearRegression(),
            'Support Vector Regression': SVR()
        }
        
       
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'Linear Regression': {},
            'Support Vector Regression': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if param_grids[name]:
               
                grid_search = GridSearchCV(
                    model, param_grids[name], 
                    cv=5, scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                # No hyperparameters to tune
                best_model = model
                best_model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
           
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': best_model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
            print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
      
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model for {target_name}: {best_model_name}")
        print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
        
        self.models[target_name] = results
        self.best_models[target_name] = best_model
        self.model_performance[target_name] = results[best_model_name]
        
        return results, best_model
    
    def save_models(self, models_dir='models'):#../
        """Save trained models"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        for target_name, model in self.best_models.items():
            model_path = os.path.join(models_dir, f'{target_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Saved {target_name} model to {model_path}")
        
        # Save performance metrics
        performance_path = os.path.join(models_dir, 'model_performance.pkl')
        with open(performance_path, 'wb') as f:
            pickle.dump(self.model_performance, f)
        print(f"✅ Saved performance metrics to {performance_path}")
    
    def generate_model_comparison_report(self):
        """Generate a comprehensive model comparison report"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        for target_name in self.target_cols.keys():
            print(f"\n📊 {target_name.replace('_', ' ').title()} Prediction Models:")
            print("-" * 60)
            
            results = self.models[target_name]
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Test R²': f"{metrics['test_r2']:.4f}",
                    'Test MSE': f"{metrics['test_mse']:.4f}",
                    'Test MAE': f"{metrics['test_mae']:.4f}",
                    'CV R² Mean': f"{metrics['cv_r2_mean']:.4f}",
                    'CV R² Std': f"{metrics['cv_r2_std']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
            print(f"\n🏆 Winner: {best_model_name}")
    
    def plot_model_performance(self):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (target_name, results) in enumerate(self.models.items()):
            if i < len(axes):
                models = list(results.keys())
                r2_scores = [results[model]['test_r2'] for model in models]
                
                axes[i].bar(models, r2_scores, color=f'C{i}', alpha=0.7)
                axes[i].set_title(f'{target_name.replace("_", " ").title()} - Test R² Scores')
                axes[i].set_ylabel('R² Score')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, score in enumerate(r2_scores):
                    axes[i].text(j, score + 0.01, f'{score:.3f}', 
                               ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('models/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Performance comparison plot saved to '../models/model_performance_comparison.png'")

def main():
    """Main training pipeline"""
    print("🚀 Starting AI Risk Prediction Model Training...")
    
    # Initialize predictor
    predictor = AIRiskPredictor()
    
    # Load data
    try:
        predictor.load_data()
    except FileNotFoundError:
        print("❌ Error: Cleaned data not found. Please run data_cleaning.ipynb first!")
        return
    
    target_names = ['critical_thinking', 'ai_overdependence', 'incorrect_info', 'privacy_concern']
    
    for target in target_names:
        predictor.train_models(target)
    
    # Generate reports
    predictor.generate_model_comparison_report()
    predictor.plot_model_performance()
    
    # Save models
    predictor.save_models()
    
    print("\n🎉 Model training completed successfully!")
    print("📁 Models saved in '../models/' directory")
    print("📊 Check model_performance_comparison.png for visual comparison")

if __name__ == "__main__":
    main()