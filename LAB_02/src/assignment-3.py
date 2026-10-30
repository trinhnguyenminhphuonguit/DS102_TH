import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


def filter_data(data, condition):
    images, labels = data
    new_images = images[labels == condition]
    new_labels = labels[labels == condition]
    return new_images, new_labels

# loading data
train_images = idx2numpy.convert_from_file("../../data/lab-02/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("../../data/lab-02/train-labels.idx1-ubyte")
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file("../../data/lab-02/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("../../data/lab-02/t10k-labels.idx1-ubyte")
test_data = (test_images, test_labels)

# flatten images
n, _, _ = train_images.shape
train_images = train_images.reshape(n, -1)

n, _, _ = test_images.shape
test_images = test_images.reshape(n, -1)

# chuan hoa du lieu
train_mean = train_images.mean()
train_std = train_images.std()

train_images = (train_images - train_mean) / train_std
test_images = (test_images - train_mean) / train_std

# logistic regression - 0 va 1
train_data = (train_images, train_labels)
test_data = (test_images, test_labels)

train_images_0, train_labels_0 = filter_data(train_data, 0)
train_images_1, train_labels_1 = filter_data(train_data, 1)

train_images_binary = np.concatenate(
    (train_images_0, train_images_1), axis=0
)
train_labels_binary = np.concatenate(
    (train_labels_0, train_labels_1), axis=0
)

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)

test_images_binary = np.concatenate(
    (test_images_0, test_images_1), axis=0
)
test_labels_binary = np.concatenate(
    (test_labels_0, test_labels_1), axis=0
)

# training logistic regression
model_logistic = LogisticRegression(
    max_iter=200,
)

model_logistic.fit(train_images_binary, train_labels_binary)
predicted_logistic = model_logistic.predict(test_images_binary)


# evaluation logistic regression
print("logistic regression 0-1")

print("precision:", precision_score(
    test_labels_binary, predicted_logistic
))
print("recall:", recall_score(
    test_labels_binary, predicted_logistic
))
print("f1:", f1_score(
    test_labels_binary, predicted_logistic
))

cfm = confusion_matrix(
    test_labels_binary,
    predicted_logistic,
    labels=[0, 1]
)

print(cfm)

sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("logistic regression 0-1")
plt.show()


# softmax regression
model_softmax = LogisticRegression(
    max_iter=200,
)

model_softmax.fit(train_images, train_labels)
predicted_softmax = model_softmax.predict(test_images)

# evaluation softmax regression
print("\nsoftmax regression 0-9")

print("precision:", precision_score(
    test_labels, predicted_softmax, average="macro"
))
print("recall:", recall_score(
    test_labels, predicted_softmax, average="macro"
))
print("f1:", f1_score(
    test_labels, predicted_softmax, average="macro",
))


