from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def run_kmeans(X, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return model, labels, score

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    return model, labels, score
