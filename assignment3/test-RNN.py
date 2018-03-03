#  coding: utf-8


import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = load_coco_data(pca_features=True)

img_num =  13945
A = np.where( np.isin(data['train_image_idxs'],img_num))[0]
print(A.tolist())


plt.imshow(image_from_url(data['train_urls'][img_num]))
plt.axis('off')
plt.show()
for i in A:
    caption_str = decode_captions(data['train_captions'][i], data['idx_to_word'])
    print(caption_str)
exit()








# for k, v in data.items():
#     if type(v) == np.ndarray:
#         print(k, type(v), v.shape, v.dtype)
#     else:
#         print(k, type(v), len(v))


# Sanity check for temporal softmax loss


np.random.seed(231)

max_train = 10000
batch_size=128
num_epochs = 1


small_data = load_coco_data(max_train=max_train)

small_rnn_model = CaptioningRNN(cell_type='rnn',word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],hidden_dim=512,wordvec_dim=256,)

small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,update_rule='adam',num_epochs=50,
           batch_size=batch_size,optim_config={'learning_rate': 5e-3,},lr_decay=0.95,verbose=True, print_every=10,)

small_rnn_solver.train()

# Plot the training losses
plt.plot(small_rnn_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()


for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = small_rnn_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()




