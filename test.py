import numpy as np

import datasets.data as data
import models
import descent_algorithms


def test_load_sms_spam():
    text, labels = data.load_sms_spam()
    print(text.shape)
    print(labels.shape)


def test_logistic_regression():
    pass


test_load_sms_spam()
