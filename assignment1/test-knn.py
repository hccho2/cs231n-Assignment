#  coding: utf-8
#import tensorflow as tf

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor
import matplotlib.pyplot as plt

import numpy as np
import sys

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)



# data 전체르 하면 다운됨
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]


X_train = np.reshape(X_train, (X_train.shape[0], -1))  # 2dim
X_test = np.reshape(X_test, (X_test.shape[0], -1))  # 2dim



num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []


X_train_folds = np.array_split(X_train, num_folds)  # (50000,3072) ==> (10000,3072)로 5(num_folds)개
y_train_folds = np.array_split(y_train, num_folds)


k_to_accuracies = {}


for k_val in k_choices:
    k_to_accuracies[k_val]=[]
    for i in range(num_folds):
        # print 'Cross-validation :'+ str(i)
        X_train_cycle = np.concatenate([f for j,f in enumerate(X_train_folds) if j!=i ])
        y_train_cycle = np.concatenate([f for j,f in enumerate(y_train_folds) if j!=i ])
        X_val_cycle =  X_train_folds[i]
        y_val_cycle = y_train_folds[i]
        knn = KNearestNeighbor()
        knn.train(X_train_cycle,y_train_cycle)
        y_val_pred = knn.predict(X_val_cycle,k_val)
        num_correct = np.sum(y_val_cycle == y_val_pred)
        k_to_accuracies[k_val].append(float(num_correct) / float(len(y_val_cycle)))
        
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print ('k = %d, accuracy = %f' % (k, accuracy))        
        
        