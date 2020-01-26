# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import itertools

dataset = pd.read_csv('TrainData_Voice.txt', delim_whitespace=True, header=None)
testset = pd.read_csv('TestData_Voice.txt', delim_whitespace=True, header=None)

dataset.head()
testset.head()

X_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values
X_test = testset.iloc[:, 1:].values
y_test = testset.iloc[:, 0].values
print(X_test)

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# in choosing the number of neigbours for KNN, make sure that it is an odd number so that we break ties
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    # this function prints the confusion matrix

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    # true labels on y axis
    plt.ylabel("True Label")

    # predicted labels on x axis
    plt.xlabel("Predicted Label")


cm_plot_labels = ["Alfredov", "Andreiv", "Guerzo", "Pacheco", "Pineda", "Tabios"]
plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix")

print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 50):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred_i = lr.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
