"""
NYC Taxi 4-Class Fare Classification Analysis

This script implements the complete 4-class fare classification system as specified:
- Short trips, low fare (< $10)
- Medium-distance trips, moderate fare ($10 - $30)
- Long-distance trips, high fare ($30 - $60)
- Premium fares (> $60)

Run this script to perform comprehensive analysis including:
- Data distribution analysis
- Model training and evaluation
- Performance comparison
- Visualization generation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Import your existing modules
from src.multiclass_classification import (
    create_fare_classes, 
    get_class_names,
    analyze_fare_distribution,
    plot_fare_distribution,
    train_multiclass_models,
    plot_confusion_matrices,
    compare_model_performance,
    print_detailed_classification_reports
)
from main import load_data


def main():
    """
    Main function to run the complete 4-class fare classification analysis.
    """
    import os  # Move this to the top
    
    print("="*80)
    print("NYC TAXI 4-CLASS FARE CLASSIFICATION ANALYSIS")
    print("="*80)
    print("Implementing classification into 4 fare ranges:")
    print("• Short trips, low fare (< $10)")
    print("• Medium-distance trips, moderate fare ($10 - $30)")
    print("• Long-distance trips, high fare ($30 - $60)")
    print("• Premium fares (> $60)")
    print("="*80)
    
    # Create output directories
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    
    try:
        # Step 1: Load data
        print("\n📊 STEP 1: Loading NYC Taxi Data...")
        X, _, y_reg = load_data()
        print(f"✅ Data loaded successfully!")
        print(f"   • Features shape: {X.shape}")
        print(f"   • Samples: {X.shape[0]:,}")
        print(f"   • Features: {X.shape[1]}")
        
        # Step 2: Create 4-class labels
        print("\n🏷️  STEP 2: Creating 4-Class Fare Labels...")
        y_multiclass = create_fare_classes(y_reg)
        class_names = get_class_names()
        
        print("✅ 4-class labels created successfully!")
        print("   Fare range distribution:")
        unique, counts = np.unique(y_multiclass, return_counts=True)
        for class_id, count in zip(unique, counts):
            percentage = (count / len(y_multiclass)) * 100
            print(f"   • {class_names[class_id]}: {count:,} samples ({percentage:.1f}%)")
        
        # Step 3: Analyze fare distribution
        print("\n📈 STEP 3: Analyzing Fare Distribution...")
        stats = analyze_fare_distribution(y_reg)
        
        # Step 4: Create visualizations (with error handling)
        print("\n📊 STEP 4: Creating Distribution Visualizations...")
        try:
            # Set matplotlib backend to avoid display issues
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            plot_fare_distribution(y_reg, save_path='outputs/visualizations/fare_distribution.png')
            print("✅ Distribution plots saved!")
        except Exception as e:
            print(f"⚠️  Warning: Could not create distribution plots: {str(e)[:100]}...")
            print("   Continuing with analysis...")
        
        # Step 5: Train models
        print("\n🤖 STEP 5: Training Classification Models...")
        print("Training multiple models for 4-class classification...")
        results = train_multiclass_models(X, y_multiclass)
        print("✅ All models trained successfully!")
        
        # Step 6: Compare performance
        print("\n📊 STEP 6: Comparing Model Performance...")
        comparison_df = compare_model_performance(results)
        
        # Step 7: Create confusion matrices (with error handling)
        print("\n🔍 STEP 7: Generating Confusion Matrices...")
        try:
            plot_confusion_matrices(results, save_path='outputs/visualizations/confusion_matrices.png')
            print("✅ Confusion matrices saved!")
        except Exception as e:
            print(f"⚠️  Warning: Could not create confusion matrices: {str(e)[:100]}...")
            print("   Continuing with analysis...")
        
        # Step 8: Print detailed reports
        print("\n📋 STEP 8: Generating Detailed Classification Reports...")
        print_detailed_classification_reports(results)
        
        # Step 9: Save results
        print("\n💾 STEP 9: Saving Results...")
        comparison_df.to_csv('outputs/tables/multiclass_model_comparison.csv', index=False)
        
        # Create a comprehensive summary report
        create_summary_report(stats, comparison_df, results)
        
        print("\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*80)
        print("📁 Output files generated:")
        print("   • outputs/tables/multiclass_model_comparison.csv")
        print("   • outputs/analysis_summary_report.txt")
        
        # Check if plots were created
        if os.path.exists('outputs/visualizations/fare_distribution.png'):
            print("   • outputs/visualizations/fare_distribution.png")
        if os.path.exists('outputs/visualizations/confusion_matrices.png'):
            print("   • outputs/visualizations/confusion_matrices.png")
        
        print("="*80)
        
        # Display final summary
        best_model = max(['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'], 
                        key=lambda x: results[x]['accuracy'])
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
        print(f"   Accuracy: {results[best_model]['accuracy']:.4f} ({results[best_model]['accuracy']*100:.2f}%)")
        
        # Show key results
        print(f"\n📊 KEY RESULTS:")
        print(f"   • Total samples analyzed: {len(y_multiclass):,}")
        print(f"   • Number of fare ranges: 4")
        print(f"   • Best model accuracy: {results[best_model]['accuracy']*100:.2f}%")
        
        # Show class distribution summary
        dominant_class = np.argmax(counts)
        dominant_percentage = (counts[dominant_class] / len(y_multiclass)) * 100
        print(f"   • Most common fare range: {class_names[dominant_class]} ({dominant_percentage:.1f}%)")
        
        return results, comparison_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please ensure all required files and dependencies are available.")
        return None, None


def create_summary_report(stats, comparison_df, results):
    """
    Create a comprehensive summary report of the analysis.
    
    Parameters:
    -----------
    stats : dict
        Distribution statistics
    comparison_df : pd.DataFrame
        Model comparison results
    results : dict
        Detailed model results
    """
    report_path = 'outputs/analysis_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NYC TAXI 4-CLASS FARE CLASSIFICATION - ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data Overview
        f.write("DATA OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {stats['total_samples']:,}\n")
        f.write(f"Fare range: ${stats['fare_stats']['min']:.2f} - ${stats['fare_stats']['max']:.2f}\n")
        f.write(f"Mean fare: ${stats['fare_stats']['mean']:.2f}\n")
        f.write(f"Median fare: ${stats['fare_stats']['median']:.2f}\n")
        f.write(f"Standard deviation: ${stats['fare_stats']['std']:.2f}\n\n")
        
        # Class Distribution
        f.write("FARE RANGE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        class_names = get_class_names()
        for i, (count, percentage, name) in enumerate(zip(stats['class_counts'], 
                                                         stats['class_percentages'], 
                                                         class_names)):
            f.write(f"{name}: {count:,} samples ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Model Performance Summary
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best Model Details
        best_model = max(['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'], 
                        key=lambda x: results[x]['accuracy'])
        f.write(f"BEST PERFORMING MODEL: {best_model}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results[best_model]['accuracy']:.4f} ({results[best_model]['accuracy']*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(results[best_model]['classification_report'])
        f.write("\n\n")
        
        # Analysis Insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("1. FARE DISTRIBUTION:\n")
        
        # Find dominant class
        dominant_class_idx = np.argmax(stats['class_counts'])
        dominant_class_name = class_names[dominant_class_idx]
        dominant_percentage = stats['class_percentages'][dominant_class_idx]
        
        f.write(f"   • Most common fare range: {dominant_class_name} ({dominant_percentage:.1f}%)\n")
        f.write(f"   • Average fare amount: ${stats['fare_stats']['mean']:.2f}\n")
        f.write(f"   • Fare variability (std): ${stats['fare_stats']['std']:.2f}\n\n")
        
        f.write("2. MODEL PERFORMANCE:\n")
        accuracies = [results[model]['accuracy'] for model in ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression']]
        f.write(f"   • Best accuracy: {max(accuracies):.4f} ({best_model})\n")
        f.write(f"   • Worst accuracy: {min(accuracies):.4f}\n")
        f.write(f"   • Average accuracy: {np.mean(accuracies):.4f}\n")
        f.write(f"   • Performance range: {max(accuracies) - min(accuracies):.4f}\n\n")
        
        f.write("3. CLASSIFICATION CHALLENGES:\n")
        f.write("   • Multi-range classification is more challenging than binary\n")
        f.write("   • Fare range imbalance may affect model performance\n")
        f.write("   • Feature engineering could improve results\n")
        f.write("   • Ensemble methods generally perform better\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Comprehensive summary report saved to: {report_path}")


if __name__ == "__main__":
    # Set up matplotlib for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run the analysis
    results, comparison_df = main()
    
    if results is not None:
        print("\n🚀 Analysis completed successfully!")
        print("You can now review the generated outputs and reports.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.") 