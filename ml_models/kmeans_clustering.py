from sklearn.cluster import KMeans

class KMeansClusteringModel:
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    
    def fit(self, X):
        self.model.fit(X)
    
    def predict(self, X):
        return self.model.predict(X)
