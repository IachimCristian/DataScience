import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from src.knn_custom import KNNFast
from src.supervised_models import train_logistic_regression
from src.ensemble_models import train_random_forest, train_gradient_boosting
from src.deep_learning import build_deep_learning_model
from src.clustering import run_kmeans, run_dbscan
from src.regression import train_random_forest_regressor, evaluate_regression
from src.multiclass_classification import create_fare_classes, get_class_names, train_multiclass_models
from src.model_comparison import run_phase5_evaluation

def load_data():
    df = pd.read_csv("data/nyc_taxi_final.csv")

    # Features (excluding fare_amount to avoid data leakage in classification)
    feature_cols = ['trip_distance', 'total_amount', 'tolls_amount',
        'pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month',
        'trip_duration', 'speed_mph', 'is_weekend', 'is_rush_hour', 'is_night',
        'pulocationid', 'passenger_count', 'payment_type', 'improvement_surcharge',
        'tip_amount', 'mta_tax', 'extra']

    X = df[feature_cols].values
    y_class = df['high_fare'].values
    y_reg = df['fare_amount'].values  # Keep raw fare amounts for regression and classification

    return X, y_class, y_reg

def run_models():
    X, y_class, y_reg = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    print("="*80)
    print("NYC TAXI DATA SCIENCE PROJECT - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print("This analysis includes:")
    print("1. Binary Classification (high_fare vs regular)")
    print("2. 4-Class Fare Classification (< $10, $10-$30, $30-$60, > $60)")
    print("3. Regression (fare amount prediction)")
    print("4. Clustering Analysis")
    print("="*80)

    # BINARY CLASSIFICATION
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION ANALYSIS")
    print("="*60)

    # kNN
    knn = KNNFast(k=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    print(f"kNN Accuracy: {knn_acc:.4f}")

    # Logistic Regression
    _, y_true, y_pred, lr_acc = train_logistic_regression(X, y_class)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Random Forest Classifier
    rf = train_random_forest(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Gradient Boosting Classifier
    gb = train_gradient_boosting(X_train, y_train)
    gb_acc = gb.score(X_test, y_test)
    print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

    # Deep Learning (with error handling)
    try:
        model_dl = build_deep_learning_model(X.shape[1])
        model_dl.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        dl_score = model_dl.evaluate(X_test, y_test, verbose=0)
        dl_acc = dl_score[1]
        print(f"Deep Learning Accuracy: {dl_acc:.4f}")
    except Exception as e:
        print(f"Deep Learning: Not available ({str(e)[:50]}...)")
        dl_acc = None

    # 4-CLASS CLASSIFICATION
    print("\n" + "="*60)
    print("4-CLASS FARE CLASSIFICATION ANALYSIS")
    print("="*60)
    print("Classes:")
    class_names = get_class_names()
    for i, name in enumerate(class_names):
        print(f"  {name}")
    print()

    # Create 4-class labels
    y_multiclass = create_fare_classes(y_reg)
    
    # Show class distribution
    unique, counts = np.unique(y_multiclass, return_counts=True)
    print("Class Distribution:")
    for class_id, count in zip(unique, counts):
        percentage = (count / len(y_multiclass)) * 100
        print(f"  {class_names[class_id]}: {count:,} samples ({percentage:.1f}%)")
    
    # Train 4-class models
    print("\nTraining 4-class classification models...")
    multiclass_results = train_multiclass_models(X, y_multiclass, test_size=0.2, random_state=42)
    
    print("\n4-Class Classification Results:")
    for model_name in ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression']:
        acc = multiclass_results[model_name]['accuracy']
        print(f"  {model_name}: {acc:.4f}")

    # CLUSTERING ANALYSIS
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)

    # Scale before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    _, kmeans_labels, kmeans_score = run_kmeans(X_scaled)
    print(f"KMeans Silhouette Score: {kmeans_score:.4f}")
    _, dbscan_labels, dbscan_score = run_dbscan(X_scaled, eps=2.0, min_samples=20)
    print(f"DBSCAN Silhouette Score: {dbscan_score if dbscan_score is not None else 'N/A'}")
    
    # Reduce features to 2D for visualization
    X_vis = PCA(n_components=2).fit_transform(X_scaled)

    # Plot KMeans clusters with legend
    plt.figure(figsize=(8, 6))
    unique_kmeans_labels = np.unique(kmeans_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_kmeans_labels)))
    
    for i, label in enumerate(unique_kmeans_labels):
        mask = kmeans_labels == label
        plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                   c=[colors[i]], s=10, label=f'Cluster {label}', alpha=0.7)
    
    plt.title("KMeans Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot DBSCAN clusters with legend
    plt.figure(figsize=(8, 6))
    unique_dbscan_labels = np.unique(dbscan_labels)
    # Use tab10 colormap for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_dbscan_labels)))
    
    for i, label in enumerate(unique_dbscan_labels):
        mask = dbscan_labels == label
        if label == -1:
            # Noise points
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                       c='black', s=10, label='Noise', alpha=0.5, marker='x')
        else:
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                       c=[colors[i]], s=10, label=f'Cluster {label}', alpha=0.7)
    
    plt.title("DBSCAN Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # REGRESSION ANALYSIS
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)

    # Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    rf_reg = train_random_forest_regressor(X_train_r, y_train_r)
    mse, mae, r2 = evaluate_regression(rf_reg, X_test_r, y_test_r)
    print(f"Random Forest Regression -> MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # FINAL SUMMARY
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print("\nBINARY CLASSIFICATION (high_fare vs regular):")
    print(f"  kNN:                    {knn_acc:.4f}")
    print(f"  Logistic Regression:    {lr_acc:.4f}")
    print(f"  Random Forest:          {rf_acc:.4f}")
    print(f"  Gradient Boosting:      {gb_acc:.4f}")
    if dl_acc is not None:
        print(f"  Deep Learning:          {dl_acc:.4f}")
    
    print("\n4-CLASS FARE CLASSIFICATION:")
    for model_name in ['KNN', 'Random Forest', 'Gradient Boosting', 'Logistic Regression']:
        acc = multiclass_results[model_name]['accuracy']
        print(f"  {model_name}:           {acc:.4f}")
    
    print(f"\nCLUSTERING:")
    print(f"  KMeans Silhouette:      {kmeans_score:.4f}")
    print(f"  DBSCAN Silhouette:      {dbscan_score if dbscan_score is not None else 'N/A'}")
    
    print(f"\nREGRESSION:")
    print(f"  Random Forest R²:       {r2:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("✅ Binary classification implemented")
    print("✅ 4-class fare classification implemented")
    print("✅ Regression analysis completed")
    print("✅ Clustering analysis completed")
    print("\nFor detailed 4-class analysis, run: python run_multiclass_classification.py")
    print("For comprehensive Phase 5 evaluation, run: python run_phase5_evaluation.py")
    print("="*80)
    
if __name__ == "__main__":
    run_models()