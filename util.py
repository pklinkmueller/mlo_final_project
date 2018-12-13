import numpy as np
import matplotlib.pyplot as plt

def check_accuracy(model,X,y):
    num_samples, num_features = X.shape
    accurate = 0

    prediction = model.predict(X)
    for i in range(num_samples):
        label = 0
        if prediction[i]>= .5:
            label = 1

        if label == y[i]:
            accurate += 1
    return (accurate / float(num_samples))