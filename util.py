import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame

def check_accuracy(model, X, y):
    num_samples, num_features = X.shape
    accurate = 0

    prediction = model.predict(X)
    for i in range(num_samples):
        label = 0
        if prediction[i] >= .5:
            label = 1

        if label == y[i]:
            accurate += 1
    return accurate / float(num_samples)


def text_vectorize(x: DataFrame):
    vect = CountVectorizer()
    return vect.fit_transform(x)

def plot_fixed_losses(gd, sgd_1, sgd_10, sgd_100, agd, svrg, md):
    plt.figure(figsize=(14,10))
    plt.xlabel('Iteration Number',fontsize='xx-large')
    plt.xlim(0,1000)
    plt.ylabel('Loss',fontsize='xx-large')
    plt.plot(gd, '+' 'b')
    plt.plot(sgd_1, 'g')
    plt.plot(sgd_10, 'y')
    plt.plot(sgd_100, 'k')
    plt.plot(agd, 'c')
    plt.plot(svrg, 'r')
    plt.plot(md, 'm')
    plt.legend(['GD','SGD_1','SGD_10','SGD_100','AGD','SVRG','MD'], fontsize='x-large')
    plt.show()

def plot_dynamic_losses(gd, sgd_1, sgd_10, sgd_100, md):
    plt.figure(figsize=(14,10))
    plt.xlabel('Iteration Number',fontsize='xx-large')
    plt.xlim(0,500)
    plt.ylabel('Loss',fontsize='xx-large')
    plt.plot(gd, '+' 'b')
    plt.plot(sgd_1, 'g')
    plt.plot(sgd_10, 'y')
    plt.plot(sgd_100, 'k')
    plt.plot(md, 'm')
    plt.legend(['GD','SGD_1','SGD_10','SGD_100','MD'], fontsize='x-large')
    plt.show()
