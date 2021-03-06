import numpy as np
import pandas as pd
import os
from typing import Tuple
from sklearn.datasets import load_svmlight_file
from keras.datasets import mnist


dir_path = os.path.dirname(__file__)


#####################################################################################################
# mammogram vectorization
# 961 samples
# contains '?' values for undefined/missing
# label 0 for benign, 1 for malignant
def load_mammogram() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(dir_path, './mammograms/mammographic_masses.data'))
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    mammograms = df.values

    mammogram_features = np.array(mammograms[:, :-1])
    mammogram_labels = np.array(mammograms[:, -1])

    print(mammogram_features[0])
    return mammogram_features, mammogram_labels


####################################################################################################
# wisconsin breast cancer vectorization
# 669 samples
# contains '?' values for undefined/missing
# label 0 for benign, 1 for malignant
def load_wisconsin_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:

    df1 = pd.read_csv(os.path.join(dir_path, './wisconsin_breast_cancer/breast-cancer-wisconsin.data'))
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df1 = df1.fillna(0)
    wbc = df1.values

    wbc_features = wbc[:, 1:-1]
    wbc_labels = wbc[:, -1]
    wbc_labels /= 2
    wbc_labels -= 1
    wbc_labels = np.array(wbc_labels)
    wbc_labels = np.reshape(wbc_labels, (-1, 1))

    return np.array(wbc_features), wbc_labels


####################################################################################################
# banknote authentication set
# 1372 samples
# label 0 for real, 1 for forgery
def load_banknote_auth_set() -> Tuple[np.ndarray, np.ndarray]:
    df2 = pd.read_csv(os.path.join(dir_path, './banknote/data_banknote_authentication.txt'))
    df2 = df2.apply(pd.to_numeric, errors='coerce')
    bn = df2.values

    bn_features = np.array(bn[:, :-1])
    bn_labels = np.array(bn[:, -1])

    return bn_features, bn_labels


####################################################################################################
# cod-rna
# 59,535 samples
def load_cod_rna() -> Tuple[np.ndarray, np.ndarray]:
    cod_features, cod_labels = load_svmlight_file(os.path.join(dir_path,'./cod_rna/cod-rna'))
    cod_labels = cod_labels + 1
    cod_labels /= 2
    
    cod_features = cod_features.toarray()
    
    cod_labels = np.array(cod_labels)
    cod_labels = np.reshape(cod_labels, (-1,1))
    
    return cod_features, cod_labels


####################################################################################################
# MNIST 1 and 3
# sample num: 12873
def load_MNIST_13() -> Tuple[np.ndarray, np.ndarray]:
    # the function by default returns a tuple with train, and tuple with test
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    indices = []
    trimmed_labels = []
    
    for index in range(y_train.size):
        if y_train[index] == 1:
            indices.append(index)
            trimmed_labels.append(0)
        elif y_train[index] == 3:
            indices.append(index)
            trimmed_labels.append(1) 
    trimmed_features = (x_train[i] for i in indices)
    trimmed_features = list(trimmed_features)
    trimmed_features = np.array(trimmed_features)
    
    trimmed_features = trimmed_features.reshape(trimmed_features.shape[0], 
        trimmed_features.shape[1] * trimmed_features.shape[2])
    
    
    trimmed_labels = np.array(trimmed_labels)
    trimmed_labels = np.reshape(trimmed_labels, (-1, 1))
        
    return trimmed_features.astype(np.float32), trimmed_labels.astype(np.float32)


####################################################################################################
# covtype.binary
# 581,012 samples
def load_covtype_binary() -> Tuple[np.ndarray, np.ndarray]:
    cov_features, cov_labels = load_svmlight_file(os.path.join(dir_path,'./covtype_binary/covtype.libsvm.binary'))
    cov_labels = cov_labels - 1
    return np.array(cov_features), np.array(cov_labels)


####################################################################################################
# spam_sms vectorization
# label: 0 for ham, 1 for spam
def load_sms_spam() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(dir_path, '/spam_sms/spam.csv')
    data = pd.read_csv(os.path.join(dir_path, './spam_sms/spam.csv'), encoding='latin-1')
    data = data.rename(columns={"v1": "label", "v2": "text"})
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
    return data['text'], data['label_num']