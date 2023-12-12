import os
import mlflow
import torch
import pandas as pd
import numpy as np

class Model():
    def __init__(self, model_uri):
        print(model_uri)
        print(os.listdir(model_uri))

        self.model_uri = model_uri
        self.model = None
        self.load()

    def load(self):
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        
    def predict(self, X, feature_names = None, meta = None):
        mnist_example = torch.tensor(X)
        mnist_example_flat = mnist_example.view(-1, 28*28)
        prediction = self.model.predict(pd.DataFrame(mnist_example_flat))

        return np.asarray(prediction)
