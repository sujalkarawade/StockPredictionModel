from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class PolynomialRegressionModel:
    def __init__(self, degree=2):
        self.model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
