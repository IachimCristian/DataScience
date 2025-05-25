import numpy as np
from sklearn.neighbors import KDTree

class KNNFast:
    def __init__(self, k=3):
        self.k = k
        self.tree = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.tree = KDTree(X_train)
        self.y_train = y_train

    def predict(self, X_test):
        distances, indices = self.tree.query(X_test, k=self.k)
        predictions = []
        for i in range(len(X_test)):
            k_labels = self.y_train[indices[i]]
            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)
