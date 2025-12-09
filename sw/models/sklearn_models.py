from sklearn.linear_model import LogisticRegression
from models.base import BaseModel

class LogisticBaseline(BaseModel):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)