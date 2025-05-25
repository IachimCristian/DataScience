"""
Run 2-Fold Cross Validation on NYC Taxi Data

This script demonstrates how to use the 2-fold cross validation
implementation to evaluate different models on the NYC taxi dataset.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the cross validation module
from src.cross_validation import (
    perform_2fold_cv, 
    compare_models_2fold_cv, 
    plot_cv_results
)

# Import data loading function
from main import load_data


def main():
    """
    Main function to run 2-fold cross validation analysis.
    """
    print("="*80)
    print("NYC TAXI DATA - 2-FOLD CROSS VALIDATION ANALYSIS")
    print("="*80)
    
    # Load the data
    print("\nLoading data...")
    try:
        X, y_class, y_reg = load_data()
        print(f"Data loaded successfully!")
        print(f"Shape of features: {X.shape}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y_class)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Option to use a subset for faster testing
    use_subset = input("\nUse a subset of data for faster testing? (y/n): ").lower() == 'y'
    
    if use_subset:
        subset_size = int(input("Enter subset size (e.g., 5000): "))
        indices = np.random.choice(X.shape[0], min(subset_size, X.shape[0]), replace=False)
        X = X[indices]
        y_class = y_class[indices]
        y_reg = y_reg[indices]
        print(f"\nUsing subset of {len(indices)} samples")
    
    # Classification task
    print("\n" + "="*80)
    print("CLASSIFICATION TASK - 2-FOLD CROSS VALIDATION")
    print("="*80)
    
    classification_df, classification_results = compare_models_2fold_cv(
        X, y_class, task="classification"
    )
    
    # Save classification results
    classification_df.to_csv('outputs/tables/2fold_cv_classification_results.csv', index=False)
    print("\nClassification results saved to: outputs/tables/2fold_cv_classification_results.csv")
    
    # Regression task
    print("\n" + "="*80)
    print("REGRESSION TASK - 2-FOLD CROSS VALIDATION")
    print("="*80)
    
    regression_df, regression_results = compare_models_2fold_cv(
        X, y_reg, task="regression"
    )
    
    # Save regression results
    regression_df.to_csv('outputs/tables/2fold_cv_regression_results.csv', index=False)
    print("\nRegression results saved to: outputs/tables/2fold_cv_regression_results.csv")
    
    # Generate visualizations
    generate_plots = input("\nGenerate visualization plots? (y/n): ").lower() == 'y'
    
    if generate_plots:
        print("\nGenerating plots...")
        try:
            plot_cv_results(classification_results, task="classification")
            plot_cv_results(regression_results, task="regression")
            print("Plots generated successfully!")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    print("\nBest Classification Model:")
    best_class_idx = classification_df['Mean Score'].str.replace('', '').astype(float).idxmax()
    print(f"  Model: {classification_df.iloc[best_class_idx]['Model']}")
    print(f"  Mean Accuracy: {classification_df.iloc[best_class_idx]['Mean Score']}")
    print(f"  Std Dev: {classification_df.iloc[best_class_idx]['Std Dev']}")
    
    print("\nRegression Model Performance:")
    print(f"  Model: {regression_df.iloc[0]['Model']}")
    print(f"  Mean R² Score: {regression_df.iloc[0]['Mean Score']}")
    print(f"  Std Dev: {regression_df.iloc[0]['Std Dev']}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


def run_single_model_cv():
    """
    Run 2-fold CV on a single model with custom parameters.
    """
    print("\n" + "="*80)
    print("SINGLE MODEL 2-FOLD CROSS VALIDATION")
    print("="*80)
    
    # Load data
    X, y_class, y_reg = load_data()
    
    # Example: Run CV on KNN with different k values
    k_values = [3, 5, 7, 9]
    results = []
    
    print("\nTesting KNN with different k values:")
    for k in k_values:
        from src.knn_custom import KNNFast
        result = perform_2fold_cv(
            X, y_class, 
            KNNFast, 
            model_name=f"KNN (k={k})",
            task="classification",
            k=k
        )
        results.append(result)
    
    # Create comparison
    comparison_df = pd.DataFrame([
        {
            'k': k,
            'Mean Accuracy': f"{r['mean_score']:.4f}",
            'Std Dev': f"{r['std_score']:.4f}",
            'Training Time': f"{r['mean_time']:.2f}s"
        }
        for k, r in zip(k_values, results)
    ])
    
    print("\nKNN Parameter Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('outputs/tables', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Run main analysis
    main()
    
    # Optionally run single model analysis
    run_single = input("\n\nRun single model parameter analysis? (y/n): ").lower() == 'y'
    if run_single:
        run_single_model_cv() 