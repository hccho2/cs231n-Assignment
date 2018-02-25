#  coding: utf-8
#import tensorflow as tf

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers import LinearSVM
import time

import matplotlib.pyplot as plt
import numpy as np
import sys

from cs231n.features import color_histogram_hsv, hog_feature
from cs231n.features import *



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

#####

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)#X_train(49000, 32, 32, 3)  ==> (49000, 154)
X_val_feats = extract_features(X_val, feature_fns)  # ==> (1000, 154)
X_test_feats = extract_features(X_test, feature_fns)  # ==> (1000, 154)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)  # (49000, 154) ==> (1, 154)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))]) # (49000, 154) ==> (49000, 155)
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])  # (1000, 154) ==> (1000, 155)
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))]) # (1000, 154) ==> (1000, 155)



