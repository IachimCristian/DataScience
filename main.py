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

def load_data():
    df = pd.read_csv("data/nyc_taxi_final.csv")

    feature_cols = ['trip_distance', 'fare_amount', 'total_amount', 'tolls_amount',
        'pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month',
        'trip_duration', 'speed_mph', 'is_weekend', 'is_rush_hour', 'is_night',
        'pulocationid', 'passenger_count', 'payment_type', 'improvement_surcharge',
        'tip_amount', 'mta_tax', 'extra']

    X = df[feature_cols].values
    y_class = df['high_fare'].values
    y_reg = df['fare_amount'].values

    return X, y_class, y_reg

def run_models():
    X, y_class, y_reg = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # kNN
    knn = KNNFast(k=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("kNN Accuracy:", accuracy_score(y_test, y_pred_knn))

    # Logistic Regression
    _, y_true, y_pred, acc = train_logistic_regression(X, y_class)
    print("Logistic Regression Accuracy:", acc)

    # Random Forest Classifier
    rf = train_random_forest(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Gradient Boosting Classifier
    gb = train_gradient_boosting(X_train, y_train)
    gb_acc = gb.score(X_test, y_test)
    print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")


    # Deep Learning
    model_dl = build_deep_learning_model(X.shape[1])
    model_dl.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    dl_score = model_dl.evaluate(X_test, y_test, verbose=0)
    print(f"Deep Learning Accuracy: {dl_score[1]:.4f}")

    # Scale before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    _, kmeans_labels, kmeans_score = run_kmeans(X_scaled)
    print(f"KMeans Silhouette Score: {kmeans_score:.4f}")
    _, dbscan_labels, dbscan_score = run_dbscan(X_scaled, eps=2.0, min_samples=20)
    print(f"DBSCAN Silhouette Score: {dbscan_score if dbscan_score is not None else 'N/A'}")
    unique_labels = np.unique(dbscan_labels)
    # print("DBSCAN cluster labels found:", unique_labels)
    # print("Label counts:", np.bincount(dbscan_labels + 1))  # shift -1 to 0
    
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

    # Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    rf_reg = train_random_forest_regressor(X_train_r, y_train_r)
    mse, mae, r2 = evaluate_regression(rf_reg, X_test_r, y_test_r)
    print(f"Regression -> MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

    # Final summary
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(f"kNN Accuracy:               {accuracy_score(y_test, y_pred_knn):.4f}")
    print(f"Logistic Regression:        {acc:.4f}")
    print(f"Random Forest:              {rf_acc:.4f}")
    print(f"Gradient Boosting:          {gb_acc:.4f}")
    print(f"Deep Learning:              {dl_score[1]:.4f}")
    print(f"KMeans Silhouette Score:    {kmeans_score:.4f}")
    print(f"DBSCAN Silhouette Score:    {dbscan_score if dbscan_score is not None else 'N/A'}")
    print(f"Regression R² Score:        {r2:.4f}")
    
if __name__ == "__main__":
    run_models()