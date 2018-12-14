import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame


def check_accuracy(model, X, y):
    num_samples, _ = X.shape
    accurate = 0

    prediction = model.predict(X)
    for i in range(num_samples):
        label = 0
        if prediction[i] >= .5:
            label = 1

        if label == y[i]:
            accurate += 1
    return accurate / float(num_samples)


def check_accuracy_svm(model, X: np.ndarray, y: np.ndarray):
    predicted = model.predict(X)
    return (predicted == y).mean()


def text_vectorize(x: DataFrame):
    vect = CountVectorizer()
    return vect.fit_transform(x)


def zero_one_labels_to_signed(y: np.ndarray):
    y_new = 2 * y - 1
    return np.sign(y_new)

