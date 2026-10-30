import numpy as np
import idx2numpy
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

from logistic_regression import LogisticsRegression

train_images = idx2numpy.convert_from_file("../../data/lab-02/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("../../data/lab-02/train-labels.idx1-ubyte")
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file("../../data/lab-02/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("../../data/lab-02/t10k-labels.idx1-ubyte")
test_data = (test_images, test_labels)

def filter_data(data, condition):
    images, labels = data

    new_images = images[labels == condition]
    new_labels = labels[labels == condition]

    return new_images, new_labels

train_images_0, train_labels_0 = filter_data(train_data,0)
train_images_1, train_labels_1 = filter_data(train_data,1)
train_images = np.concatenate(
    [train_images_0, train_images_1],
    axis=0
)
train_labels = np.concatenate(
    [train_labels_0, train_labels_1],
    axis=0
)

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)

test_images = np.concatenate(
    [test_images_0, test_images_1],
    axis=0
)
test_labels = np.concatenate(
    [test_labels_0, test_labels_1],
    axis=0
)

# chuyen anh 28x28 thanh vector 784 phan tu
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1)
train_images = train_images / 255

N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1)
test_images = test_images / 255

model = LogisticsRegression(
    epoch=100,
    lr=0.1
)
model.fit(train_images, train_labels)
predicted_labels = model.predict(test_images)
metrics = model.evaluate(test_labels, predicted_labels)

for metric in metrics:
    print(f"{metric}:\t {metrics[metric]}")

# chuyen xac suat thanh nhan 0 va 1
predicted_classes = np.where(predicted_labels <= 0.5, 0, 1)

# ma tran nham lan
cfm = confusion_matrix(test_labels, predicted_classes)
print(cfm)

sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("logistic regression - mnist 0 va 1")
plt.show()

# do thi loss
plt.plot(model.losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("logistic regression loss")
plt.show()