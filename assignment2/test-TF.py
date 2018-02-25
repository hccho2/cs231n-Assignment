#  coding: utf-8
#import tensorflow as tf

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'D://hccho//ML//cs231n//assignment1//cs231n//datasets//cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# clear old variables
tf.reset_default_graph()


def simple_model(X,y):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])  # 'VALID': (32 - 7 +1) / 2 = 13
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10]) # 13 x 13 x 32 = 5408
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_flat,W1) + b1
    return y_out


# define model
def complex_model(X,y,is_training):
    weight_scale = 0.01
    filter_size = 3
    Wconv1 = tf.get_variable("Wconv1", shape=[filter_size, filter_size, 3, 32], initializer=tf.constant_initializer(weight_scale*np.random.randn(filter_size, filter_size, 3, 32)))
    bconv1 = tf.get_variable("bconv1", shape=[32],initializer=tf.constant_initializer(np.zeros(32)))
    W4 = tf.get_variable("W4", shape=[8192, 1024],initializer=tf.constant_initializer(weight_scale*np.random.randn(8192, 1024))) # 16 x 16 x 32 = 8192
    b4 = tf.get_variable("b4", shape=[1024],initializer=tf.constant_initializer(np.zeros(1024)))        

    W6 = tf.get_variable("W6", shape=[1024, 10],initializer=tf.constant_initializer(weight_scale*np.random.randn(1024, 10))) # 13 x 13 x 32 = 5408
    b6 = tf.get_variable("b6", shape=[10],initializer=tf.constant_initializer(np.zeros(10))) 

    h1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    h2 = tf.nn.relu(h1)
    h2_flat = tf.reshape(h2,[-1,32])
    h3_flat = tf.contrib.layers.batch_norm(h2_flat,decay=0.9,updates_collections=None,epsilon=1e-5,
                      center=True,scale=True,is_training=is_training,scope='BN')
    h3 = tf.reshape(h3_flat,[-1,32,32,32])
    h4 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Pooling') # 16 x 16 x 32 = 8192
    h4_flat = tf.reshape(h4,[-1,8192])
    h5 = tf.matmul(h4_flat,W4) + b4
    h6 = tf.nn.relu(h5)
    y_out = tf.matmul(h6,W6) + b6
    
    loss_reg = tf.nn.l2_loss(Wconv1) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W6)
    
    return y_out, loss_reg

def MyCNN(X,y,is_training):
    conv1_1 = tf.layers.conv2d(inputs=X, filters=32, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu, name = 'conv1_1')
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu,name = 'conv1_2')
    pool1_3 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=2, strides=2, padding='SAME', name= 'pool1_3')  # ==> (?, 16, 16, 32)
 
    conv2_1 = tf.layers.conv2d(inputs=pool1_3, filters=64, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu, name = 'conv2_1')
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=64, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu,name = 'conv2_2')
    pool2_3 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=2, strides=2, padding='SAME', name= 'pool2_3')  # ==> (?, 8, 8, 64)
 
    conv3_1 = tf.layers.conv2d(inputs=pool2_3, filters=128, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu, name = 'conv3_1')
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=128, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu,name = 'conv3_2')
    pool3_3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=2, strides=2, padding='SAME', name= 'pool3_3')  # ==> (?, 4, 4, 128)
     
    flatten1 = tf.reshape(pool3_3,[-1,2048])
    
    
    mode = 2
    if mode == 1:
        fc3_1 = tf.layers.dense(inputs=flatten1, units=1024 , activation=tf.nn.relu,name='fc3_1')
        drop3_2 = tf.layers.dropout(inputs=fc3_1, rate=0.5, training=is_training, name='drop3_2')
        fc4_1 = tf.layers.dense(inputs=drop3_2, units=1024 , activation=tf.nn.relu,name='fc4_1')
    else:
        fc3_1 = tf.layers.dense(inputs=flatten1, units=1024 , activation=None,name='fc3_1')
        bn3_2 = tf.layers.batch_normalization(inputs=fc3_1, training=is_training,name='bn3_2')
        relu3_3 = tf.nn.relu(bn3_2,name='relu3_3')
        fc4_1 = tf.layers.dense(inputs=relu3_3, units=1024 , activation=tf.nn.relu,name='fc4_1')
     
 
    y_out = tf.layers.dense(inputs=fc4_1, units=10 , activation=None,name='out')
 
    return y_out





def run_model(session, predict, loss_val, Xd, yd,epochs=1, batch_size=64, print_every=100,training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    num_batch = int(math.ceil(Xd.shape[0]/batch_size))
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(num_batch):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {}/{} (epoch {}): with minibatch training loss = {:.3g} and accuracy of {:.2g}"\
                      .format(iter_cnt,num_batch*epochs,e,loss,np.sum(corr)/actual_batch_size))

            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct




mode = 2
if mode ==1:
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])    
    is_training = tf.placeholder(tf.bool)
    y_out, loss_reg = complex_model(X,y,is_training)
    
    reg_weight = 0.01
    total_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_out,labels=tf.one_hot(y,10))
    mean_loss = tf.reduce_mean(total_loss) + reg_weight * tf.reduce_mean(loss_reg)
    optimizer = tf.train.AdamOptimizer(5e-4)
    
    # batch normalization in tensorflow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess,y_out,mean_loss,X_train,y_train,5,64,100,train_step)
    print('Validation')
    run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
else:
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = my_model(X,y,is_training)
    
    
    total_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_out,labels=tf.one_hot(y,10))
    mean_loss = tf.reduce_mean(total_loss)
    optimizer = tf.train.AdamOptimizer(5e-4)
    
    # batch normalization in tensorflow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)
    
    
    
    with tf.Session() as sess:
        with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()
    
            print('Training')
            run_model(sess,y_out,mean_loss,X_train,y_train,10,64,100,train_step)
            print('Validation')
            run_model(sess,y_out,mean_loss,X_val,y_val,1,64)    
            print('Test')
            run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
#############










