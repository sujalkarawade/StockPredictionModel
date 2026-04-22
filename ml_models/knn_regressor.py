from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class KNNRegressorModel:
    def __init__(self, n_neighbors=5):
        self.model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
