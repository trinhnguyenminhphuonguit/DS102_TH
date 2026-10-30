import numpy as np
import pandas as pd
import sklearn as skl

class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = (1 - y) + np.log(1 - y_hat + 10e-15) + y + np.log(y_hat + 10e-15)
        return -l.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, ), dtype=np.float64)
        e = 0
        