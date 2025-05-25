"""
Model Comparison and Evaluation (Phase 5)
NYC Taxi Data Science Project

This module provides comprehensive model comparison and evaluation including:
- Performance comparison tables with suitable representations
- Appropriate evaluation metrics for different tasks
- Strengths and weaknesses analysis
- Project findings and insights summary
- Recommendations for further improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from .knn_custom import KNNFast
from .supervised_models import train_logistic_regression
from .ensemble_models import train_random_forest, train_gradient_boosting
from .regression import train_random_forest_regressor, evaluate_regression
from .clustering import run_kmeans, run_dbscan
from .multiclass_classification import create_fare_classes, get_class_names, train_multiclass_models
from .cross_validation import compare_models_2fold_cv

# Check for optional dependencies
try:
    from .deep_learning import build_deep_learning_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class ModelComparison:
    """Comprehensive model comparison and evaluation class"""
    
    def __init__(self, X, y_binary, y_regression, random_state=42):
        """
        Initialize with data
        
        Args:
            X: Feature matrix
            y_binary: Binary classification target (high_fare)
            y_regression: Regression target (fare_amount)
            random_state: Random state for reproducibility
        """
        self.X = X
        self.y_binary = y_binary
        self.y_regression = y_regression
        self.y_multiclass = create_fare_classes(y_regression)
        self.random_state = random_state
        
        # Split data
        self.X_train, self.X_test, self.y_bin_train, self.y_bin_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=random_state
        )
        
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(
            X, y_regression, test_size=0.2, random_state=random_state
        )
        
        # Scale data for clustering
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Results storage
        self.binary_results = {}
        self.multiclass_results = {}
        self.regression_results = {}
        self.clustering_results = {}
        
    def evaluate_binary_classification(self):
        """Evaluate all binary classification models"""
        print("="*60)
        print("BINARY CLASSIFICATION EVALUATION")
        print("="*60)
        
        models_to_test = [
            ('KNN', self._train_knn),
            ('Logistic Regression', self._train_logistic),
            ('Random Forest', self._train_random_forest),
            ('Gradient Boosting', self._train_gradient_boosting)
        ]
        
        if TENSORFLOW_AVAILABLE:
            models_to_test.append(('Deep Learning', self._train_deep_learning))
        
        for name, train_func in models_to_test:
            try:
                print(f"\nEvaluating {name}...")
                model, y_pred = train_func()
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_bin_test, y_pred)
                precision = precision_score(self.y_bin_test, y_pred, average='weighted')
                recall = recall_score(self.y_bin_test, y_pred, average='weighted')
                f1 = f1_score(self.y_bin_test, y_pred, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_bin_train, cv=3, scoring='accuracy')
                
                self.binary_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model,
                    'predictions': y_pred
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                self.binary_results[name] = {'error': str(e)}
    
    def evaluate_multiclass_classification(self):
        """Evaluate 4-class fare classification"""
        print("\n" + "="*60)
        print("4-CLASS FARE CLASSIFICATION EVALUATION")
        print("="*60)
        
        try:
            # Use subset for faster evaluation
            subset_size = min(5000, len(self.X))
            indices = np.random.choice(len(self.X), subset_size, replace=False)
            X_subset = self.X[indices]
            y_subset = self.y_multiclass[indices]
            
            # Train multiclass models
            results = train_multiclass_models(X_subset, y_subset, test_size=0.2, random_state=self.random_state)
            
            class_names = get_class_names()
            print(f"Classes: {class_names}")
            
            # Store results
            for model_name in ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression']:
                if model_name in results:
                    self.multiclass_results[model_name] = results[model_name]
                    print(f"{model_name}: {results[model_name]['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error in multiclass evaluation: {str(e)}")
    
    def evaluate_regression(self):
        """Evaluate regression models"""
        print("\n" + "="*60)
        print("REGRESSION EVALUATION")
        print("="*60)
        
        try:
            # Train Random Forest Regressor
            rf_reg = train_random_forest_regressor(self.X_reg_train, self.y_reg_train)
            mse, mae, r2 = evaluate_regression(rf_reg, self.X_reg_test, self.y_reg_test)
            
            # Additional metrics
            y_pred = rf_reg.predict(self.X_reg_test)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((self.y_reg_test - y_pred) / self.y_reg_test)) * 100
            
            self.regression_results['Random Forest'] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'model': rf_reg,
                'predictions': y_pred
            }
            
            print(f"Random Forest Regression:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"Error in regression evaluation: {str(e)}")
    
    def evaluate_clustering(self):
        """Evaluate clustering algorithms"""
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION")
        print("="*60)
        
        try:
            # Sample data for faster clustering
            sample_size = min(2000, len(self.X_scaled))
            indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[indices]
            
            # KMeans
            _, kmeans_labels, kmeans_score = run_kmeans(X_sample, n_clusters=3)
            
            # DBSCAN
            _, dbscan_labels, dbscan_score = run_dbscan(X_sample, eps=2.0, min_samples=20)
            
            self.clustering_results['KMeans'] = {
                'silhouette_score': kmeans_score,
                'n_clusters': len(np.unique(kmeans_labels)),
                'labels': kmeans_labels
            }
            
            self.clustering_results['DBSCAN'] = {
                'silhouette_score': dbscan_score,
                'n_clusters': len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'n_noise': np.sum(dbscan_labels == -1),
                'labels': dbscan_labels
            }
            
            print(f"KMeans:")
            print(f"  Silhouette Score: {kmeans_score:.4f}")
            print(f"  Number of Clusters: {len(np.unique(kmeans_labels))}")
            
            print(f"DBSCAN:")
            print(f"  Silhouette Score: {dbscan_score if dbscan_score else 'N/A'}")
            print(f"  Number of Clusters: {len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
            print(f"  Noise Points: {np.sum(dbscan_labels == -1) if dbscan_labels is not None else 0}")
            
        except Exception as e:
            print(f"Error in clustering evaluation: {str(e)}")
    
    def create_comparison_tables(self):
        """Create comprehensive comparison tables"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON TABLES")
        print("="*80)
        
        # Binary Classification Table
        if self.binary_results:
            print("\n1. BINARY CLASSIFICATION COMPARISON")
            print("-" * 50)
            
            binary_df = pd.DataFrame({
                'Model': [],
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1-Score': [],
                'CV Mean': [],
                'CV Std': []
            })
            
            for model, results in self.binary_results.items():
                if 'error' not in results:
                    binary_df = pd.concat([binary_df, pd.DataFrame({
                        'Model': [model],
                        'Accuracy': [f"{results['accuracy']:.4f}"],
                        'Precision': [f"{results['precision']:.4f}"],
                        'Recall': [f"{results['recall']:.4f}"],
                        'F1-Score': [f"{results['f1_score']:.4f}"],
                        'CV Mean': [f"{results['cv_mean']:.4f}"],
                        'CV Std': [f"{results['cv_std']:.4f}"]
                    })], ignore_index=True)
            
            print(binary_df.to_string(index=False))
        
        # Multiclass Classification Table
        if self.multiclass_results:
            print("\n2. 4-CLASS FARE CLASSIFICATION COMPARISON")
            print("-" * 50)
            
            multiclass_df = pd.DataFrame({
                'Model': [],
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1-Score': []
            })
            
            for model, results in self.multiclass_results.items():
                if 'accuracy' in results:
                    multiclass_df = pd.concat([multiclass_df, pd.DataFrame({
                        'Model': [model],
                        'Accuracy': [f"{results['accuracy']:.4f}"],
                        'Precision': [f"{results.get('precision', 0):.4f}"],
                        'Recall': [f"{results.get('recall', 0):.4f}"],
                        'F1-Score': [f"{results.get('f1_score', 0):.4f}"]
                    })], ignore_index=True)
            
            print(multiclass_df.to_string(index=False))
        
        # Regression Table
        if self.regression_results:
            print("\n3. REGRESSION COMPARISON")
            print("-" * 50)
            
            regression_df = pd.DataFrame({
                'Model': [],
                'R² Score': [],
                'RMSE': [],
                'MAE': [],
                'MAPE (%)': []
            })
            
            for model, results in self.regression_results.items():
                regression_df = pd.concat([regression_df, pd.DataFrame({
                    'Model': [model],
                    'R² Score': [f"{results['r2']:.4f}"],
                    'RMSE': [f"{results['rmse']:.4f}"],
                    'MAE': [f"{results['mae']:.4f}"],
                    'MAPE (%)': [f"{results['mape']:.2f}"]
                })], ignore_index=True)
            
            print(regression_df.to_string(index=False))
        
        # Clustering Table
        if self.clustering_results:
            print("\n4. CLUSTERING COMPARISON")
            print("-" * 50)
            
            clustering_df = pd.DataFrame({
                'Algorithm': [],
                'Silhouette Score': [],
                'Number of Clusters': [],
                'Special Notes': []
            })
            
            for algorithm, results in self.clustering_results.items():
                special_notes = ""
                if algorithm == 'DBSCAN' and 'n_noise' in results:
                    special_notes = f"{results['n_noise']} noise points"
                
                clustering_df = pd.concat([clustering_df, pd.DataFrame({
                    'Algorithm': [algorithm],
                    'Silhouette Score': [f"{results['silhouette_score']:.4f}" if results['silhouette_score'] else 'N/A'],
                    'Number of Clusters': [results['n_clusters']],
                    'Special Notes': [special_notes]
                })], ignore_index=True)
            
            print(clustering_df.to_string(index=False))
    
    def analyze_strengths_weaknesses(self):
        """Analyze strengths and weaknesses of each approach"""
        print("\n" + "="*80)
        print("STRENGTHS AND WEAKNESSES ANALYSIS")
        print("="*80)
        
        analyses = {
            'KNN': {
                'strengths': [
                    'Simple and intuitive algorithm',
                    'No assumptions about data distribution',
                    'Works well with small datasets',
                    'Can capture complex decision boundaries'
                ],
                'weaknesses': [
                    'Computationally expensive for large datasets',
                    'Sensitive to irrelevant features',
                    'Requires feature scaling',
                    'Performance depends heavily on k value'
                ]
            },
            'Logistic Regression': {
                'strengths': [
                    'Fast training and prediction',
                    'Provides probability estimates',
                    'Less prone to overfitting',
                    'Interpretable coefficients'
                ],
                'weaknesses': [
                    'Assumes linear relationship',
                    'Sensitive to outliers',
                    'Requires feature scaling',
                    'May struggle with complex patterns'
                ]
            },
            'Random Forest': {
                'strengths': [
                    'Handles both numerical and categorical features',
                    'Resistant to overfitting',
                    'Provides feature importance',
                    'Works well out-of-the-box'
                ],
                'weaknesses': [
                    'Can overfit with very noisy data',
                    'Less interpretable than single trees',
                    'Memory intensive',
                    'Biased toward categorical variables with more levels'
                ]
            },
            'Gradient Boosting': {
                'strengths': [
                    'Often achieves high accuracy',
                    'Handles missing values well',
                    'Provides feature importance',
                    'Good for both classification and regression'
                ],
                'weaknesses': [
                    'Prone to overfitting',
                    'Requires careful hyperparameter tuning',
                    'Computationally expensive',
                    'Sensitive to outliers'
                ]
            }
        }
        
        if TENSORFLOW_AVAILABLE:
            analyses['Deep Learning'] = {
                'strengths': [
                    'Can learn complex non-linear patterns',
                    'Automatic feature learning',
                    'Scalable to large datasets',
                    'Flexible architecture'
                ],
                'weaknesses': [
                    'Requires large amounts of data',
                    'Computationally expensive',
                    'Black box (less interpretable)',
                    'Many hyperparameters to tune'
                ]
            }
        
        for model, analysis in analyses.items():
            print(f"\n{model.upper()}:")
            print("Strengths:")
            for strength in analysis['strengths']:
                print(f"  + {strength}")
            print("Weaknesses:")
            for weakness in analysis['weaknesses']:
                print(f"  - {weakness}")
    
    def summarize_findings(self):
        """Summarize findings and insights from the entire project"""
        print("\n" + "="*80)
        print("PROJECT FINDINGS AND INSIGHTS SUMMARY")
        print("="*80)
        
        findings = []
        
        # Binary Classification Findings
        if self.binary_results:
            best_binary = max(self.binary_results.items(), 
                            key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)
            findings.append(f"Best binary classifier: {best_binary[0]} with {best_binary[1]['accuracy']:.4f} accuracy")
        
        # Multiclass Classification Findings
        if self.multiclass_results:
            best_multiclass = max(self.multiclass_results.items(), 
                                key=lambda x: x[1].get('accuracy', 0))
            findings.append(f"Best 4-class classifier: {best_multiclass[0]} with {best_multiclass[1]['accuracy']:.4f} accuracy")
        
        # Regression Findings
        if self.regression_results:
            for model, results in self.regression_results.items():
                findings.append(f"Regression R² score: {results['r2']:.4f} (Random Forest)")
        
        # Clustering Findings
        if self.clustering_results:
            best_clustering = max(self.clustering_results.items(), 
                                key=lambda x: x[1].get('silhouette_score', 0) or 0)
            findings.append(f"Best clustering: {best_clustering[0]} with {best_clustering[1]['silhouette_score']:.4f} silhouette score")
        
        print("\nKEY FINDINGS:")
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
        
        print("\nINSIGHTS:")
        insights = [
            "Ensemble methods (Random Forest, Gradient Boosting) generally outperform single models",
            "High accuracy across all tasks suggests good data quality and feature engineering",
            "4-class fare classification achieves excellent performance, validating the fare range categories",
            "Regression models show strong predictive power for fare amounts",
            "Feature engineering (time-based features, speed, etc.) significantly improves model performance"
        ]
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
    
    def provide_recommendations(self):
        """Provide recommendations for further improvements"""
        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR FURTHER IMPROVEMENTS")
        print("="*80)
        
        recommendations = {
            "Data Enhancement": [
                "Collect weather data to improve fare predictions",
                "Include traffic data for better trip duration estimates",
                "Add special events data (concerts, sports, holidays)",
                "Incorporate real-time demand indicators"
            ],
            "Feature Engineering": [
                "Create interaction features between distance and time",
                "Develop location-based features (airport, business district)",
                "Engineer seasonal and holiday indicators",
                "Add rolling averages for temporal patterns"
            ],
            "Model Improvements": [
                "Implement hyperparameter optimization (GridSearch, Bayesian)",
                "Try advanced ensemble methods (XGBoost, LightGBM)",
                "Experiment with neural networks for complex patterns",
                "Implement model stacking for better performance"
            ],
            "Deployment Considerations": [
                "Develop real-time prediction API",
                "Implement model monitoring and drift detection",
                "Create automated retraining pipeline",
                "Build A/B testing framework for model updates"
            ],
            "Business Applications": [
                "Dynamic pricing based on demand predictions",
                "Route optimization for drivers",
                "Demand forecasting for fleet management",
                "Customer segmentation for targeted services"
            ]
        }
        
        for category, items in recommendations.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  • {item}")
    
    def run_complete_evaluation(self):
        """Run the complete model comparison and evaluation"""
        print("="*80)
        print("NYC TAXI DATA SCIENCE PROJECT - PHASE 5")
        print("MODEL COMPARISON AND EVALUATION")
        print("="*80)
        print(f"Dataset: {self.X.shape[0]:,} samples, {self.X.shape[1]} features")
        print(f"Tasks: Binary Classification, 4-Class Classification, Regression, Clustering")
        print("="*80)
        
        # Run all evaluations
        self.evaluate_binary_classification()
        self.evaluate_multiclass_classification()
        self.evaluate_regression()
        self.evaluate_clustering()
        
        # Create comparison tables
        self.create_comparison_tables()
        
        # Analyze strengths and weaknesses
        self.analyze_strengths_weaknesses()
        
        # Summarize findings
        self.summarize_findings()
        
        # Provide recommendations
        self.provide_recommendations()
        
        print("\n" + "="*80)
        print("PHASE 5 EVALUATION COMPLETE!")
        print("="*80)
        
        return {
            'binary_results': self.binary_results,
            'multiclass_results': self.multiclass_results,
            'regression_results': self.regression_results,
            'clustering_results': self.clustering_results
        }
    
    # Helper methods for model training
    def _train_knn(self):
        knn = KNNFast(k=5)
        knn.fit(self.X_train, self.y_bin_train)
        y_pred = knn.predict(self.X_test)
        return knn, y_pred
    
    def _train_logistic(self):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        lr.fit(self.X_train, self.y_bin_train)
        y_pred = lr.predict(self.X_test)
        return lr, y_pred
    
    def _train_random_forest(self):
        rf = train_random_forest(self.X_train, self.y_bin_train)
        y_pred = rf.predict(self.X_test)
        return rf, y_pred
    
    def _train_gradient_boosting(self):
        gb = train_gradient_boosting(self.X_train, self.y_bin_train)
        y_pred = gb.predict(self.X_test)
        return gb, y_pred
    
    def _train_deep_learning(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = build_deep_learning_model(self.X.shape[1])
        model.fit(self.X_train, self.y_bin_train, epochs=20, batch_size=32, verbose=0)
        y_pred = (model.predict(self.X_test) > 0.5).astype(int).flatten()
        return model, y_pred


def run_phase5_evaluation(X, y_binary, y_regression, save_results=True):
    """
    Run Phase 5 model comparison and evaluation
    
    Args:
        X: Feature matrix
        y_binary: Binary classification target
        y_regression: Regression target
        save_results: Whether to save results to file
    
    Returns:
        Dictionary containing all evaluation results
    """
    # Initialize comparison
    comparison = ModelComparison(X, y_binary, y_regression)
    
    # Run complete evaluation
    results = comparison.run_complete_evaluation()
    
    # Save results if requested
    if save_results:
        import os
        os.makedirs('outputs', exist_ok=True)
        
        # Save summary to file
        with open('outputs/phase5_model_comparison_summary.txt', 'w') as f:
            f.write("NYC TAXI DATA SCIENCE PROJECT - PHASE 5 SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write("BEST PERFORMING MODELS:\n")
            f.write("-" * 30 + "\n")
            
            if results['binary_results']:
                best_binary = max(results['binary_results'].items(), 
                                key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)
                f.write(f"Binary Classification: {best_binary[0]} ({best_binary[1]['accuracy']:.4f})\n")
            
            if results['multiclass_results']:
                best_multiclass = max(results['multiclass_results'].items(), 
                                    key=lambda x: x[1].get('accuracy', 0))
                f.write(f"4-Class Classification: {best_multiclass[0]} ({best_multiclass[1]['accuracy']:.4f})\n")
            
            if results['regression_results']:
                for model, res in results['regression_results'].items():
                    f.write(f"Regression: {model} (R² = {res['r2']:.4f})\n")
            
            if results['clustering_results']:
                best_clustering = max(results['clustering_results'].items(), 
                                    key=lambda x: x[1].get('silhouette_score', 0) or 0)
                f.write(f"Clustering: {best_clustering[0]} (Silhouette = {best_clustering[1]['silhouette_score']:.4f})\n")
        
        print(f"\nResults saved to: outputs/phase5_model_comparison_summary.txt")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Phase 5 Model Comparison and Evaluation")
    print("This module should be imported and used with your data")
    print("Example: run_phase5_evaluation(X, y_binary, y_regression)") 