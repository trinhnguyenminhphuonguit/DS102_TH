import numpy as np
import idx2numpy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns

from softmax_regression import SoftmaxRegression

def convert_to_onehot_vector(labels: np.ndarray):
    """
    labels: (n,) -> (n, 10)
    """
    N = labels.shape[0]
    total_classes = labels.max() + 1
    oh_labels = np.zeros((N, total_classes))
    oh_labels[np.arange(N), labels] = 1

    return oh_labels

# loading data
train_images = idx2numpy.convert_from_file("../../data/lab-02/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("../../data/lab-02/train-labels.idx1-ubyte")
encoded_train_labels = convert_to_onehot_vector(train_labels)
train_data = (train_images, encoded_train_labels)

test_images = idx2numpy.convert_from_file("../../data/lab-02/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("../../data/lab-02/t10k-labels.idx1-ubyte")
encoded_test_labels = convert_to_onehot_vector(test_labels)
test_data = (test_images, encoded_test_labels)

# flatten images
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1)
train_images = (train_images - train_images.mean()) / train_images.std()

N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1)
test_images = (test_images - test_images.mean()) / test_images.std()


model = SoftmaxRegression(
    epoch=100,
    lr=0.01
)
model.fit(train_images, encoded_train_labels)

# du doan xac suat roi lay lop co xac suat lon nhat
predicted_probs = model.predict(test_images)
predicted_labels = np.argmax(predicted_probs, axis=1)

# evaluation
precision = precision_score(test_labels, predicted_labels, average="macro")
recall = recall_score(test_labels, predicted_labels, average="macro")
f1 = f1_score(test_labels, predicted_labels, average="macro")

print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

# confusion matrix
cfm = confusion_matrix(test_labels, predicted_labels)
print(cfm)
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("softmax regression - mnist 0 to 9")
plt.show()

# loss
plt.plot(model.losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("softmax regression loss")
plt.show()