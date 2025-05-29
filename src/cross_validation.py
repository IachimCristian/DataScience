"""
2-Fold Cross Validation Implementation for NYC Taxi Analytics

This module provides functions to perform 2-fold cross validation
on various machine learning models used in the project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import time

# Import your custom models
from .knn_custom import KNNFast
from .supervised_models import train_logistic_regression
from .ensemble_models import train_random_forest, train_gradient_boosting
from .regression import train_random_forest_regressor


def perform_2fold_cv(X, y, model, model_name="Model", task="classification", **model_params):
    """
    Perform 2-fold cross validation on a given model.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    model : estimator object
        The model to evaluate
    model_name : str
        Name of the model for display
    task : str
        Either 'classification' or 'regression'
    **model_params : dict
        Additional parameters for the model
        
    Returns:
    --------
    dict : Dictionary containing cross-validation results
    """
    
    # Initialize 2-fold cross validation
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    
    # Store results
    results = {
        'model_name': model_name,
        'fold_scores': [],
        'fold_times': [],
        'detailed_metrics': []
    }
    
    print(f"\n{'='*50}")
    print(f"2-Fold Cross Validation for {model_name}")
    print(f"{'='*50}")
    
    # Perform cross validation
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold_idx}:")
        print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        start_time = time.time()
        
        # Handle different model types
        if model_name == "KNN":
            model_instance = KNNFast(**model_params)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
        elif model_name == "Logistic Regression":
            # Special handling for logistic regression
            from sklearn.linear_model import LogisticRegression
            model_instance = LogisticRegression(max_iter=1000, random_state=42)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
        elif model_name == "Deep Learning":
            if task == "classification":
                from src.deep_learning import build_deep_learning_model
                input_dim = X_train.shape[1]
                model_instance = build_deep_learning_model(input_dim)
                model_instance.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                y_pred_probs = model_instance.predict(X_test).flatten()
                y_pred = (y_pred_probs > 0.5).astype(int)

            elif task == "regression":
                from src.deep_learning import build_deep_learning_regressor
                input_dim = X_train.shape[1]
                model_instance = build_deep_learning_regressor(input_dim)
                model_instance.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                y_pred = model_instance.predict(X_test).flatten()
        else:
            # For sklearn-compatible models
            model_instance = model(**model_params)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
        
        train_time = time.time() - start_time
        
        # Calculate metrics based on task
        if task == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            fold_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_time': train_time
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Training time: {train_time:.2f}s")
            
            results['fold_scores'].append(accuracy)
            
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            fold_metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'train_time': train_time
            }
            
            print(f"  MSE:  {mse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  Training time: {train_time:.2f}s")
            
            results['fold_scores'].append(r2)
        
        results['fold_times'].append(train_time)
        results['detailed_metrics'].append(fold_metrics)
    
    # Calculate summary statistics
    results['mean_score'] = np.mean(results['fold_scores'])
    results['std_score'] = np.std(results['fold_scores'])
    results['mean_time'] = np.mean(results['fold_times'])
    
    print(f"\n{'='*50}")
    print(f"Summary for {model_name}:")
    if task == "classification":
        print(f"Mean Accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    else:
        print(f"Mean R² Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    print(f"Mean Training Time: {results['mean_time']:.2f}s")
    print(f"{'='*50}")
    
    return results


def compare_models_2fold_cv(X, y, task="classification"):
    """
    Compare multiple models using 2-fold cross validation.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target values
    task : str
        Either 'classification' or 'regression'
        
    Returns:
    --------
    pd.DataFrame : Comparison results
    """
    
    results_list = []
    
    if task == "classification":
        # Define models to compare
        models = [
            ("KNN", KNNFast, {'k': 5}),
            ("Logistic Regression", None, {}),
            ("Random Forest", RandomForestClassifier, {}),
            ("Gradient Boosting", GradientBoostingClassifier, {'learning_rate': 0.1}),
            ("Deep Learning", None, {})
        ]
        
        print("\nComparing Classification Models with 2-Fold Cross Validation")
        print("="*60)
        
        for model_name, model_func, params in models:
            if model_func is None and model_name == "Logistic Regression":
                # Special handling for logistic regression
                results = perform_2fold_cv(X, y, None, model_name, task, **params)
            else:
                results = perform_2fold_cv(X, y, model_func, model_name, task, **params)
            results_list.append(results)
    
    else:  # regression
        # For regression tasks
        models = [
            ("Random Forest Regressor", RandomForestRegressor, {'n_estimators': 100}),
            ("Gradient Boosting Regressor", GradientBoostingRegressor, {'n_estimators': 100}),
            ("Linear Regression", LinearRegression, {}),
            ("KNN Regressor", KNeighborsRegressor, {'n_neighbors': 5}),
            ("Deep Learning", None, {})
        ]
        
        print("\nComparing Regression Models with 2-Fold Cross Validation")
        print("="*60)
        
        for model_name, model_func, params in models:
            results = perform_2fold_cv(X, y, model_func, model_name, task, **params)
            results_list.append(results)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Mean Score': f"{r['mean_score']:.4f}",
            'Std Dev': f"{r['std_score']:.4f}",
            'Fold 1 Score': f"{r['fold_scores'][0]:.4f}",
            'Fold 2 Score': f"{r['fold_scores'][1]:.4f}",
            'Mean Time (s)': f"{r['mean_time']:.2f}"
        }
        for r in results_list
    ])
    
    print("\n\nFinal Comparison Table:")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    return comparison_df, results_list


def plot_cv_results(results_list, task="classification"):
    """
    Create visualizations for cross-validation results.
    
    Parameters:
    -----------
    results_list : list
        List of results from compare_models_2fold_cv
    task : str
        Either 'classification' or 'regression'
    """
    import matplotlib.pyplot as plt
    
    # Extract data for plotting
    model_names = [r['model_name'] for r in results_list]
    mean_scores = [r['mean_score'] for r in results_list]
    std_scores = [r['std_score'] for r in results_list]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Model comparison with error bars
    x_pos = np.arange(len(model_names))
    ax1.bar(x_pos, mean_scores, yerr=std_scores, capsize=10, 
            color='skyblue', edgecolor='navy', linewidth=2)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score' if task == 'classification' else 'R² Score')
    ax1.set_title('2-Fold CV Model Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(mean_scores, std_scores)):
        ax1.text(i, mean + std + 0.01, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Fold scores comparison
    fold1_scores = [r['fold_scores'][0] for r in results_list]
    fold2_scores = [r['fold_scores'][1] for r in results_list]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x - width/2, fold1_scores, width, label='Fold 1', color='lightcoral')
    ax2.bar(x + width/2, fold2_scores, width, label='Fold 2', color='lightgreen')
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Score' if task == 'classification' else 'R² Score')
    ax2.set_title('Individual Fold Scores')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Training time comparison
    plt.figure(figsize=(8, 5))
    mean_times = [r['mean_time'] for r in results_list]
    plt.bar(model_names, mean_times, color='orange', edgecolor='darkred', linewidth=2)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Average Training Time per Fold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, time in enumerate(mean_times):
        plt.text(i, time + 0.1, f'{time:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


# Example usage function
def run_2fold_cv_example():
    """
    Example of how to use the 2-fold cross validation functions.
    """
    # Load your data
    from main import load_data
    X, y_class, y_reg = load_data()
    
    # Run classification models comparison
    print("\n" + "="*80)
    print("CLASSIFICATION TASK - 2-FOLD CROSS VALIDATION")
    print("="*80)
    
    classification_results, classification_list = compare_models_2fold_cv(
        X, y_class, task="classification"
    )
    
    # Run regression model evaluation
    print("\n" + "="*80)
    print("REGRESSION TASK - 2-FOLD CROSS VALIDATION")
    print("="*80)
    
    regression_results, regression_list = compare_models_2fold_cv(
        X, y_reg, task="regression"
    )
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    plot_cv_results(classification_list, task="classification")
    plot_cv_results(regression_list, task="regression")
    
    return classification_results, regression_results


if __name__ == "__main__":
    # Run the example when module is executed directly
    run_2fold_cv_example() 