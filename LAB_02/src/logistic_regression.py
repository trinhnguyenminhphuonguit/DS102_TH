import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

class LogisticsRegression:
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
        with tqdm(range(self.epoch), desc = f'Epoch (e + 1):') as pbar:
            for _ in pbar:
                # forward pass
                y_hat = self.predict(X)
                delta_y = (y_hat - y)

                # backward pass
                gradient = delta_y.T @ X
                self.w = self.w - gradient.T * (self.lr / N)

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "loss": l
                })
                e += 1
    def sigmoid(self, z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w
        y_hat = self.sigmoid(z)
        return y_hat

    def evaluate(self, y, y_hat) -> dict:
        y_hat = np.where(y_hat <= 0.5, 0, 1)
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

