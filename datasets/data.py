import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.datasets import load_svmlight_file


def load_mammogram() -> Tuple[np.ndarray, np.ndarray]:
    # Everything has been split roughly into 80:20 training:testing from the datasets
    ####################################################################################################
    # mammogram vectorization
    # 961 samples
    # contains '?' values for undefined/missing
    # label 0 for benign, 1 for malignant
    df = pd.read_csv('datasets/mammograms/mammographic_masses.data')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    mammograms = df.values

    mammogram_features = np.array(mammograms[:, :-1])
    mammogram_labels = np.array(mammograms[:, -1])

    return mammogram_features, mammogram_labels


def load_wisconsin_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:
    ####################################################################################################
    # wisconsin breast cancer vectorization
    # 669 samples
    # contains '?' values for undefined/missing
    # label 0 for benign, 1 for malignant

    df1 = pd.read_csv('datasets/wisconsin_breast_cancer/breast-cancer-wisconsin.data')
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df1 = df1.fillna(0)
    wbc = df1.values

    wbc_features = wbc[:, 1:-1]
    wbc_labels = wbc[:, -1]
    wbc_labels /= 2
    wbc_labels -= 1

    return np.array(wbc_features), np.array(wbc_labels)


def load_banknote_auth_set() -> Tuple[np.ndarray, np.ndarray]:
    ####################################################################################################
    # banknote authentication set
    # 1372 samples
    # label 0 for real, 1 for forgery

    df2 = pd.read_csv('datasets/banknote/data_banknote_authentication.txt')
    df2 = df2.apply(pd.to_numeric, errors='coerce')
    bn = df2.values

    bn_features = np.array(bn[:, :-1])
    bn_labels = np.array(bn[:, -1])

    return bn_features, bn_labels


def load_covtype_binary() -> Tuple[np.ndarray, np.ndarray]:
    ####################################################################################################
    # covtype.binary
    # 581,012 samples

    cov_features, cov_labels = load_svmlight_file('datasets/covtype_binary/covtype.libsvm.binary')
    cov_labels = cov_labels - 1
    return np.array(cov_features), np.array(cov_labels)


