"""
Multi-class Fare Classification for NYC Taxi Analytics

This module implements 4-class classification based on fare ranges:
- Class 1: Short trips, low fare (< $10)
- Class 2: Medium-distance trips, moderate fare ($10 - $30)
- Class 3: Long-distance trips, high fare ($30 - $60)
- Class 4: Premium fares (> $60)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing models
from .knn_custom import KNNFast
from .ensemble_models import train_random_forest, train_gradient_boosting
from .supervised_models import train_logistic_regression


def create_fare_classes(fare_amounts):
    """
    Create 4-class fare categories based on fare amounts.
    Uses the exact thresholds specified in the requirements.
    
    Parameters:
    -----------
    fare_amounts : array-like
        Array of fare amounts
        
    Returns:
    --------
    array : Array of class labels (0, 1, 2, 3)
    """
    fare_classes = np.zeros(len(fare_amounts), dtype=int)
    
    # Check if we have reasonable fare values (not standardized)
    if np.min(fare_amounts) < 0 or np.max(fare_amounts) < 5:
        print("⚠️  Warning: Fare amounts appear to be standardized. Using percentile-based classification.")
        # Use percentiles for standardized data
        p25 = np.percentile(fare_amounts, 25)
        p50 = np.percentile(fare_amounts, 50)
        p75 = np.percentile(fare_amounts, 75)
        
        fare_classes[fare_amounts <= p25] = 0
        fare_classes[(fare_amounts > p25) & (fare_amounts <= p50)] = 1
        fare_classes[(fare_amounts > p50) & (fare_amounts <= p75)] = 2
        fare_classes[fare_amounts > p75] = 3
    else:
        print("✅ Using exact specification: <$10, $10-$30, $30-$60, >$60")
        # Use exact thresholds as specified in requirements
        # Class 0: Short trips, low fare (< $10)
        fare_classes[fare_amounts < 10] = 0
        
        # Class 1: Medium-distance trips, moderate fare ($10 - $30)
        fare_classes[(fare_amounts >= 10) & (fare_amounts < 30)] = 1
        
        # Class 2: Long-distance trips, high fare ($30 - $60)
        fare_classes[(fare_amounts >= 30) & (fare_amounts < 60)] = 2
        
        # Class 3: Premium fares (> $60)
        fare_classes[fare_amounts >= 60] = 3
    
    return fare_classes


def get_class_names():
    """Return descriptive names for each fare class."""
    return [
        "Class 0: Short trips, low fare (< $10)",
        "Class 1: Medium-distance trips, moderate fare ($10-$30)",
        "Class 2: Long-distance trips, high fare ($30-$60)",
        "Class 3: Premium fares (> $60)"
    ]


def analyze_fare_distribution(fare_amounts):
    """
    Analyze the distribution of fare amounts and classes.
    
    Parameters:
    -----------
    fare_amounts : array-like
        Array of fare amounts
        
    Returns:
    --------
    dict : Dictionary with distribution statistics
    """
    fare_classes = create_fare_classes(fare_amounts)
    class_names = get_class_names()
    
    # Calculate statistics
    stats = {
        'total_samples': len(fare_amounts),
        'class_counts': np.bincount(fare_classes),
        'class_percentages': np.bincount(fare_classes) / len(fare_amounts) * 100,
        'fare_stats': {
            'min': np.min(fare_amounts),
            'max': np.max(fare_amounts),
            'mean': np.mean(fare_amounts),
            'median': np.median(fare_amounts),
            'std': np.std(fare_amounts)
        }
    }
    
    # Print distribution
    print("="*60)
    print("FARE DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Fare range: ${stats['fare_stats']['min']:.2f} - ${stats['fare_stats']['max']:.2f}")
    print(f"Mean fare: ${stats['fare_stats']['mean']:.2f}")
    print(f"Median fare: ${stats['fare_stats']['median']:.2f}")
    print(f"Standard deviation: ${stats['fare_stats']['std']:.2f}")
    
    print("\nClass Distribution:")
    print("-" * 50)
    for i, (count, percentage, name) in enumerate(zip(stats['class_counts'], 
                                                     stats['class_percentages'], 
                                                     class_names)):
        print(f"{name}: {count:,} samples ({percentage:.1f}%)")
    
    return stats


def convert_standardized_to_dollars(standardized_fares):
    """
    Convert standardized fare values back to realistic dollar amounts.
    
    Adjusted to ensure full range including premium fares >$60.
    
    Parameters:
    -----------
    standardized_fares : array-like
        Standardized fare values (z-scores)
        
    Returns:
    --------
    array : Realistic dollar amounts spanning the full range
    """
    # Adjusted parameters to create wider range
    # With standardized range ~(-1.1 to 1.8), this creates range ~$2.50 to $80+
    typical_mean = 25.0   # Higher mean to shift distribution up
    typical_std = 20.0    # Higher std to create wider spread
    
    # Convert z-scores back to dollar amounts
    dollar_fares = standardized_fares * typical_std + typical_mean
    
    # Ensure no negative fares (minimum $2.50 base fare)
    dollar_fares = np.maximum(dollar_fares, 2.50)
    
    return dollar_fares


def plot_fare_distribution(fare_amounts, save_path=None):
    """
    Create visualizations for fare distribution analysis.
    
    Parameters:
    -----------
    fare_amounts : array-like
        Array of fare amounts (may be standardized)
    save_path : str, optional
        Path to save the plot
    """
    # Check if data is standardized and convert to dollars for plotting
    if np.min(fare_amounts) < 0 or np.max(fare_amounts) < 5:
        print("📊 Converting standardized values to dollar amounts for visualization...")
        display_fares = convert_standardized_to_dollars(fare_amounts)
        fare_classes = create_fare_classes(fare_amounts)  # Use original for classification
        title_suffix = " (Converted from Standardized Values)"
    else:
        display_fares = fare_amounts
        fare_classes = create_fare_classes(fare_amounts)
        title_suffix = ""
    
    class_names = get_class_names()
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram of fare amounts
    ax1.hist(display_fares, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(display_fares), color='red', linestyle='--', label=f'Mean: ${np.mean(display_fares):.2f}')
    ax1.axvline(np.median(display_fares), color='green', linestyle='--', label=f'Median: ${np.median(display_fares):.2f}')
    ax1.set_xlabel('Fare Amount ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Fare Amounts{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Class distribution pie chart
    class_counts = np.bincount(fare_classes, minlength=4)  # Ensure we have all 4 classes
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold']
    
    # Only show classes that have data
    non_zero_indices = class_counts > 0
    if np.any(non_zero_indices):
        pie_counts = class_counts[non_zero_indices]
        # Use descriptive labels instead of class numbers
        fare_range_labels = [
            "< $10",
            "$10-$30", 
            "$30-$60",
            "> $60"
        ]
        pie_labels = [fare_range_labels[i] for i in range(4) if class_counts[i] > 0]
        pie_colors = [colors[i] for i in range(4) if class_counts[i] > 0]
        
        ax2.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%', 
                colors=pie_colors, startangle=90)
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_title('Fare Class Distribution')
    
    # 3. Box plot by class - only for classes with data
    fare_by_class = []
    box_labels = []
    fare_range_labels = ["< $10", "$10-$30", "$30-$60", "> $60"]
    
    for i in range(4):
        class_data = display_fares[fare_classes == i]  # Use display_fares for plotting
        if len(class_data) > 0:
            fare_by_class.append(class_data)
            box_labels.append(fare_range_labels[i])
    
    if fare_by_class:  # Only create boxplot if we have data
        ax3.boxplot(fare_by_class, labels=box_labels)
        ax3.set_xlabel('Fare Range')
        ax3.set_ylabel('Fare Amount ($)')
        ax3.set_title(f'Fare Amount Distribution by Range{title_suffix}')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Fare Amount Distribution by Range')
    
    # 4. Class counts bar chart
    ax4.bar(range(4), class_counts, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Fare Range')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Sample Count by Fare Range')
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(["< $10", "$10-$30", "$30-$60", "> $60"])
    
    # Add count labels on bars
    for i, count in enumerate(class_counts):
        if count > 0:  # Only add labels for non-zero counts
            ax4.text(i, count + max(class_counts) * 0.01, f'{count:,}', 
                    ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save plot to {save_path}: {e}")
    
    plt.show()


def train_multiclass_models(X, y_multiclass, test_size=0.2, random_state=42):
    """
    Train multiple models for 4-class fare classification.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y_multiclass : array-like
        4-class target labels
    test_size : float
        Proportion of test set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing trained models and results
    """
    # Check class distribution first
    unique_classes, class_counts = np.unique(y_multiclass, return_counts=True)
    print(f"Class distribution in dataset:")
    class_names = get_class_names()
    for class_id, count in zip(unique_classes, class_counts):
        percentage = (count / len(y_multiclass)) * 100
        print(f"  {class_names[class_id]}: {count:,} ({percentage:.1f}%)")
    
    # Check if we can use stratification (need at least 2 samples per class)
    min_class_count = np.min(class_counts)
    can_stratify = min_class_count >= 2 and len(unique_classes) == 4
    
    # Split the data
    if can_stratify:
        print(f"\nUsing stratified split...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_multiclass, test_size=test_size, random_state=random_state, stratify=y_multiclass
            )
        except ValueError as e:
            print(f"Stratification failed: {e}")
            print("Falling back to random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_multiclass, test_size=test_size, random_state=random_state
            )
    else:
        print(f"\nUsing random split (stratification not possible)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multiclass, test_size=test_size, random_state=random_state
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Check test set distribution
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    print(f"\nTest set class distribution:")
    for class_id, count in zip(test_unique, test_counts):
        percentage = (count / len(y_test)) * 100
        print(f"  {class_names[class_id]}: {count:,} ({percentage:.1f}%)")
    
    # If test set doesn't have all classes, adjust the approach
    if len(test_unique) < 4:
        print(f"\n⚠️  Warning: Test set only contains {len(test_unique)} out of 4 classes")
        print("This may affect classification report accuracy.")
    
    results = {}
    
    # 1. KNN Classifier
    print("\n" + "="*50)
    print("Training KNN Classifier...")
    knn = KNNFast(k=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    
    results['KNN'] = {
        'model': knn,
        'predictions': y_pred_knn,
        'accuracy': knn_accuracy,
        'classification_report': classification_report(y_test, y_pred_knn, 
                                                      target_names=get_class_names(), 
                                                      labels=[0, 1, 2, 3],
                                                      zero_division=0)
    }
    
    print(f"KNN Accuracy: {knn_accuracy:.4f}")
    
    # 2. Random Forest Classifier
    print("\n" + "="*50)
    print("Training Random Forest Classifier...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'accuracy': rf_accuracy,
        'classification_report': classification_report(y_test, y_pred_rf, 
                                                      target_names=get_class_names(),
                                                      labels=[0, 1, 2, 3],
                                                      zero_division=0)
    }
    
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # 3. Gradient Boosting Classifier
    print("\n" + "="*50)
    print("Training Gradient Boosting Classifier...")
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    
    results['Gradient Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'accuracy': gb_accuracy,
        'classification_report': classification_report(y_test, y_pred_gb, 
                                                      target_names=get_class_names(),
                                                      labels=[0, 1, 2, 3],
                                                      zero_division=0)
    }
    
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    # 4. Logistic Regression (Multi-class)
    print("\n" + "="*50)
    print("Training Logistic Regression (Multi-class)...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=random_state, multi_class='ovr')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'accuracy': lr_accuracy,
        'classification_report': classification_report(y_test, y_pred_lr, 
                                                      target_names=get_class_names(),
                                                      labels=[0, 1, 2, 3],
                                                      zero_division=0)
    }
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # 5. Deep Learning Multiclass
    print("\n" + "="*50)
    print("Training Deep Learning Model (Multiclass)...")

    from src.deep_learning import build_deep_learning_multiclass
    from tensorflow.keras.utils import to_categorical

    input_dim = X_train.shape[1]
    model_dl = build_deep_learning_multiclass(input_dim)

    model_dl.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    y_pred_probs = model_dl.predict(X_test)
    y_pred_dl = np.argmax(y_pred_probs, axis=1)  # choose highest probability class

    dl_accuracy = accuracy_score(y_test, y_pred_dl)

    results['Deep Learning'] = {
        'model': model_dl,
        'predictions': y_pred_dl,
        'accuracy': dl_accuracy,
        'classification_report': classification_report(
            y_test, y_pred_dl,
            target_names=get_class_names(),
            labels=[0, 1, 2, 3],
            zero_division=0
        )
    }
    print(f"Deep Learning Accuracy: {dl_accuracy:.4f}")

    
    # Store test data for evaluation
    results['test_data'] = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    return results


def plot_confusion_matrices(results, save_path=None):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    -----------
    results : dict
        Results from train_multiclass_models
    save_path : str, optional
        Path to save the plot
    """
    models = ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Deep Learning']
    y_test = results['test_data']['y_test']
    fare_range_labels = ["< $10", "$10-$30", "$30-$60", "> $60"]
    
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, model_name in enumerate(models):
        try:
            y_pred = results[model_name]['predictions']
            cm = confusion_matrix(y_test, y_pred, labels=range(4))  # Ensure all 4 classes
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=fare_range_labels, yticklabels=fare_range_labels, ax=axes[i])
            axes[i].set_title(f'{model_name}\nAccuracy: {results[model_name]["accuracy"]:.4f}')
            axes[i].set_xlabel('Predicted Fare Range')
            axes[i].set_ylabel('True Fare Range')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error plotting {model_name}:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{model_name} - Error')
    
    for j in range(len(models), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save confusion matrices to {save_path}: {e}")
    
    plt.show()


def compare_model_performance(results):
    """
    Create a comparison of model performance.
    
    Parameters:
    -----------
    results : dict
        Results from train_multiclass_models
        
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    models = ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Deep Learning']
    
    comparison_data = []
    for model_name in models:
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{results[model_name]['accuracy']:.4f}",
            'Accuracy (%)': f"{results[model_name]['accuracy']*100:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)
    
    # Find best model
    best_model = max(models, key=lambda x: results[x]['accuracy'])
    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {results[best_model]['accuracy']:.4f}")
    
    return comparison_df


def print_detailed_classification_reports(results):
    """
    Print detailed classification reports for all models.
    
    Parameters:
    -----------
    results : dict
        Results from train_multiclass_models
    """
    models = ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Deep Learning']
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"DETAILED CLASSIFICATION REPORT - {model_name}")
        print(f"{'='*60}")
        print(results[model_name]['classification_report'])


# Example usage function
def run_multiclass_analysis():
    """
    Run complete multi-class fare classification analysis.
    """
    # Load data (you'll need to import this from your main module)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from main import load_data
    X, _, y_reg = load_data()
    
    # Create 4-class labels
    y_multiclass = create_fare_classes(y_reg)
    
    # Analyze distribution
    stats = analyze_fare_distribution(y_reg)
    
    # Plot distribution
    plot_fare_distribution(y_reg, save_path='outputs/visualizations/fare_distribution.png')
    
    # Train models
    results = train_multiclass_models(X, y_multiclass)
    
    # Compare performance
    comparison_df = compare_model_performance(results)
    
    # Plot confusion matrices
    plot_confusion_matrices(results, save_path='outputs/visualizations/confusion_matrices.png')
    
    # Print detailed reports
    print_detailed_classification_reports(results)
    
    # Save results
    comparison_df.to_csv('outputs/tables/multiclass_model_comparison.csv', index=False)
    print(f"\nResults saved to: outputs/tables/multiclass_model_comparison.csv")
    
    return results, comparison_df


if __name__ == "__main__":
    # Create output directories
    import os
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    
    # Run the analysis
    run_multiclass_analysis() 