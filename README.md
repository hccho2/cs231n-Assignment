# cs231n-2017-Assignment
### Assignment1 -  Done

### Assignment2 -  Done
참고: Assignment2의 과제 함수 중 일부는 Assignment3에 정답(?)이 있다.

### Assignment3 -  Ongoing

-----------------------------------------------------------
-----------------------------------------------------------
# Tuning-Network(cs231n Assignment2에서)
 - 목표: Validation에서 acc 70% 이상

### 기본 Network 구조 
	* 7x7 Convolutional Layer with 32 filters and stride of 1
	* ReLU Activation Layer
	* Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
	* 2x2 Max Pooling layer with a stride of 2
	* Affine layer with 1024 output units
	* ReLU Activation Layer
	* Affine layer from 1024 input units to 10 outputs
``` js
def complex_model(X,y,is_training):
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W4 = tf.get_variable("W4", shape=[8192, 1024]) # 13 x 13 x 32 = 5408
    b4 = tf.get_variable("b4", shape=[1024])        

    W6 = tf.get_variable("W6", shape=[1024, 10]) # 13 x 13 x 32 = 5408
    b6 = tf.get_variable("b6", shape=[10]) 

    h1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    h2 = tf.nn.relu(h1)
    h2_flat = tf.reshape(h2,[-1,32])
    h3_flat = tf.contrib.layers.batch_norm(h2_flat,decay=0.9,updates_collections=None,epsilon=1e-5,
                      center=True,scale=True,is_training=is_training,scope='BN')
    h3 = tf.reshape(h3_flat,[-1,32,32,32])
    h4 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Pooling') # 6 x 6 x 32
    h4_flat = tf.reshape(h4,[-1,8192])
    h5 = tf.matmul(h4_flat,W4) + b4
    h6 = tf.nn.relu(h5)
    y_out = tf.matmul(h6,W6) + b6
    
    return y_out
``` 
	

### tf.train.AdamOptimizer(5e-4)
    * weight 초기화 없이: Epoch 1, Overall loss = 1.62 and accuracy of 0.471 
    * weight를 초기화: Epoch 1, Overall loss = 1.45 and accuracy of 0.488 --> BN이 있었기 때문에, 초기화 효과는 많지 않음.
    * filter size(7 --> 5로 축소): Epoch 1, Overall loss = 1.43 and accuracy of 0.496(Val: accuracy of 0.571)
    * filter size(5 --> 3로 축소): Epoch 1, Overall loss = 1.43 and accuracy of 0.493(Val: accuracy of 0.607)
    * filter size(3): Epoch 5, Overall loss = 0.196 and accuracy of 0.934(Val: accuracy of 0.647)
    * regularization(lambda = 0.01): Epoch 5, Overall loss = 0.127 and accuracy of 0.661(Val: accuracy of 0.651)
    * regularization(lambda = 0.1): Epoch 5, Overall loss = 0.157 and accuracy of 0.576(Val: accuracy of 0.571)


### 좀더 deep하게 Network2 다시 구성
``` js
def Network2(X,y,is_training):
    conv1_1 = tf.layers.conv2d(inputs=X, filters=32, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu, name = 'conv1_1')
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu,name = 'conv1_2')
    pool1_3 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=2, strides=2, padding='SAME', name= 'pool1_3')  # ==> (?, 16, 16, 32)

    conv2_1 = tf.layers.conv2d(inputs=pool1_3, filters=64, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu, name = 'conv2_1')
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=64, padding='SAME', kernel_size=3, strides=1, activation=tf.nn.relu,name = 'conv2_2')
    pool2_3 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=2, strides=2, padding='SAME', name= 'pool2_3')  # ==> (?, 8, 8, 64)
    
    flatten1 = tf.reshape(pool2_3,[-1,4096])
    fc3_1 = tf.layers.dense(inputs=flatten1, units=1024 , activation=tf.nn.relu,name='fc3_1')
    drop3_2 = tf.layers.dropout(inputs=fc3_1, rate=0.5, training=is_training, name='drop3_2')

    fc4_1 = tf.layers.dense(inputs=drop3_2, units=1024 , activation=tf.nn.relu,name='fc4_1')

    y_out = tf.layers.dense(inputs=fc4_1, units=10 , activation=None,name='out')

    return y_out
```
### Network2에 대한 training 결과
	* Epoch 5, Overall loss = 0.656 and accuracy of 0.769 (Val: accuracy of 0.708, Test: acc of 0.674)
 
 
### 1층 더 추가하여 Network3 구성
``` js
def Network3(X,y,is_training):
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
``` 
 
### Network3(Drop Out)에 대한 training 결과
	* Epoch 5, Overall loss = 0.601 and accuracy of 0.792 (Val: accuracy of 0.748, Test: acc of 0.741)
	* Epoch 10, Overall loss = 0.601 and accuracy of 0.895 (Val: accuracy of 0.745, Test: acc of 0.741)
	* Network2, Network3 모두 5epoch에 Validation Accuracy 70% 이상 달성.
	
	
### Network3의 	drop3_2를 Batch Normalization으로 변경
	* Epoch 5, Overall loss = 0.332 and accuracy of 0.884 (Val: accuracy of 0.728, Test: acc of 0.71)
	* Epoch 10, Overall loss = 0.163 and accuracy of 0.962 (Val: accuracy of 0.716, Test: acc of 0.705)	
	* Batch Normalization보다 Drop Out의 성능이 좀 더 나아 보임.
 
 
 
-----------------------------------------------------------
-----------------------------------------------------------
# Captioning RNN(cs231n Assignment2에서)
 - 목표: Validation에서 BLEU score 0.3 이상
 
### 실행결과
	* hidden_dim=512, wordvec_dim=256, num_layers=1, train_size = 0.2M x epoch(5) 
		- Average BLEU score for train: 0.28
		- Average BLEU score for val:: 0.27
	* hidden_dim=512, wordvec_dim=256, num_layers=1, train_size = 0.2M x epoch(30)
		- Average BLEU score for train: 0.344124
		- Average BLEU score for val:: 0.279812 
	* epoch만 늘려서는 val BLEU가 좋아지지 않는다.

