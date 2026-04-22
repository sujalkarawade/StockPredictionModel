from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()
    
    def fit(self, X, y_binary):
        self.model.fit(X, y_binary)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
