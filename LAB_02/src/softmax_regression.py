import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -(y * np.log(y_hat + 10e-15)).sum(axis=1).mean()

    def fit(self, x: np.ndarray, y: np.ndarray):
        N, d = x.shape
        _, k = y.shape
        self.w = np.zeros((d, k))
        e = 0
        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar:
                # forward pass
                y_hat = self.predict(x)

                # backward pass
                delta_y = (y_hat - y)  # (n, k)
                gradient = delta_y.T @ x  # (k, d)
                self.w = self.w - (1 / N) * self.lr * gradient.T  # (d, k)

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "epoch": e + 1,
                    "loss": l
                })
                e += 1

    def softmax(self, z: np.ndarray):
        z = z - np.max(z, axis=1, keepdims=True)
        denom = np.exp(z).sum(axis=1, keepdims=True)
        return np.exp(z) / denom

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w
        y_hat = self.softmax(z)
        return y_hat

    def evaluate(self, y, y_hat) -> dict:
        y_hat = np.argmax(y_hat, axis=1)
        precision = precision_score(y, y_hat, average="macro")
        recall = recall_score(y, y_hat, average="macro")
        f1 = f1_score(y, y_hat, average="macro")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }