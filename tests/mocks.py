import numpy as np 

# Mock model for api testing
class MockModel:

    def __init__(self, model_path: str = None):

        self.model_path = None
        self.model = None
        self.modelname = 'mock.h5'

        self.load()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape)

    def train(self, X: np.ndarray, y: np.ndarray):
        return self

    def load(self):
        return self
