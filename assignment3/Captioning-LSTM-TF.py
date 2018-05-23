#  coding: utf-8
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import time, datetime, os
import numpy as np
import nltk
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


s = time.time()
print(datetime.datetime.now())

def get_max_iteration_checkpoint(ckpt):
    import re
    counter = int(next(re.finditer("(\d+)(?!.*\d)",os.path.basename(ckpt.model_checkpoint_path))).group(0))
    max_index = 0
    for i in range(len(ckpt.all_model_checkpoint_paths)):
        tmp = int(next(re.finditer("(\d+)(?!.*\d)",os.path.basename(ckpt.all_model_checkpoint_paths[i]))).group(0))
        if tmp > counter:
            max_index = i
    
    print('loaded checkpoint: ', ckpt.all_model_checkpoint_paths[max_index])
    return ckpt.all_model_checkpoint_paths[max_index]

def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ') 
                 if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ') 
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore

def evaluate_model_tf(sess,model,data,batch_size=1000):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.
    """
    BLEUscores ={}
    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(data, split=split, batch_size=batch_size)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = model.sample(sess,features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        total_score = 0.0
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            total_score += BLEU_score(gt_caption, sample_caption)

        BLEUscores[split] = total_score / len(sample_captions)

    for split in BLEUscores:
        print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))


class CaptioningRNN:
    
    def __init__(self,word_to_idx, input_dim=512, wordvec_dim=128,hidden_dim=128,batch_size = 128,seq_length=1,num_layers=1):
    
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        
        self.vocab_dim = len(word_to_idx)
        self.input_dim = input_dim # Dimension D of input image feature vectors.
        self.hidden_dim = hidden_dim # Dimension H for the hidden state of the RNN.
        self.num_layers = num_layers # Multi RNN layer number
        self.wordvec_dim = wordvec_dim # word embedding dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)    

        self.params={}
        

        self.params['embedding'] = tf.get_variable("embedding", 
                    initializer=0.01*np.random.randn(self.vocab_dim, self.wordvec_dim).astype(np.float32),dtype = tf.float32)
        self.params['Wp'] = tf.get_variable("Wp", shape=[input_dim,hidden_dim],initializer=tf.constant_initializer(np.random.randn(input_dim,hidden_dim)/np.sqrt(input_dim)))
        self.params['bp'] = tf.get_variable("bp", shape=[hidden_dim],initializer=tf.constant_initializer(np.zeros(hidden_dim)))

        self.captions_in = tf.placeholder(tf.int32,[self.batch_size,self.seq_length],name='caption_in')
        self.captions_out = tf.placeholder(tf.int32,[self.batch_size,self.seq_length],name='caption_out')        
        self.features = tf.placeholder(tf.float32,[self.batch_size,self.input_dim],name='features')
        self.build()
    def build(self):
        # features, captions: tf.placeholder
        
        self.tuple_mode = True

        cells = []
        for _ in range(self.num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=self.tuple_mode)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)  

        inputs = tf.nn.embedding_lookup(self.params['embedding'], self.captions_in)  #batch_size x seq_length x wordvec_dim
        h0 = tf.matmul(self.features,self.params['Wp']) + self.params['bp'] # batch_size x hidden_dim
        

        #initial_state =  cell.zero_state(batch_size, tf.float32) # (batch_size x hidden_dim) x layer 개수 
        # LSTMStateTuple (c,h), h부분을 given 값인 h0로 초기화
        if self.tuple_mode:
            self.initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(self.num_layers-1)
        else:
            self.initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (self.num_layers-1)

        
        helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([self.seq_length]*self.batch_size,dtype=np.int32))
        output_layer = Dense(self.vocab_dim, activation = None, name='output_projection')
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=self.initial_state,output_layer=output_layer) 
        self.outputs, self.last_state, last_sequence_lengths = \
                    tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True)
        #weight = tf.not_equal(self.captions_out, tf.fill(self.captions_out.get_shape(),self._null))
        weights = tf.cast(tf.not_equal(self.captions_out, self._null),tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs.rnn_output, targets=self.captions_out, weights=weights)

    def sample(self, sess, features, max_length=30):
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)
        caption_in = np.zeros((N,1),dtype = np.int32)
        caption_in[:,0] = self._start        
         
        h0 = tf.matmul(features,self.params['Wp']) + self.params['bp']
        
        if self.tuple_mode:
            state=sess.run((tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(self.num_layers-1))
        else:
            state = sess.run((tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (self.num_layers-1))
        
        for i in range(max_length):
            feed =  {self.captions_in: caption_in,self.initial_state:state}
            out, state = sess.run([self.outputs.rnn_output,self.last_state], feed_dict = feed)
            captions[:,i] = np.argmax(np.squeeze(out,axis=1),axis=1)
            caption_in[:,0] = captions[:,i]
    
        return captions
    
max_train = 200000
batch_size = 512
num_epoch = 1




    
    
data = load_coco_data(pca_features=True)
small_data = load_coco_data(max_train=max_train)

word_to_idx=data['word_to_idx']
input_dim=data['train_features'].shape[1]
hidden_dim=512
wordvec_dim=512
num_layers=2
if num_layers == 1:
    ckpt_save_dir = ".\save-sigle-layer"    
elif num_layers ==2:
    ckpt_save_dir = ".\save-double-layer"    
else:
    print('check num_layers')
    exit()

Mode = 3 # 0: train 1: test.  2: BLEU socre

if Mode==0: 
    with tf.device('/cpu:0'):
        sess = tf.Session()
         
        minibatch = sample_coco_minibatch(small_data, split='train', batch_size=batch_size)
        captions, features, urls = minibatch
        _,T = captions.shape
         
        model = CaptioningRNN(word_to_idx=data['word_to_idx'],input_dim=input_dim, 
                              wordvec_dim=wordvec_dim,hidden_dim=hidden_dim,batch_size=batch_size,seq_length = T-1,num_layers=num_layers)
         
         
        train = tf.train.AdamOptimizer(0.001).minimize(model.loss)    
     
        num_batch = int(max_train/batch_size)
        loss_history=[]
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        for i in range(num_epoch):
             
            for j in range(num_batch):
                minibatch = sample_coco_minibatch(small_data, split='train', batch_size=batch_size)
                captions, features, urls = minibatch            
                feed = {model.features: features,model.captions_in: captions[:,:-1],model.captions_out: captions[:,1:]}
                _, loss = sess.run([train,model.loss], feed_dict=feed)
                loss_history.append(loss)
                if j % 10 == 0:
                    print('(Iteration %d / %d) loss: %f' % (i*num_batch+j, num_epoch*num_batch, loss_history[-1]))
     
     
        if not os.path.isdir(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
         
        checkpoint_path = os.path.join(ckpt_save_dir, 'model')
        saver.save(sess, checkpoint_path,global_step=num_epoch*max_train)
        print("model saved to {}".format(checkpoint_path))    
     
        sess.close()
    
elif Mode==1:
    tf.reset_default_graph()
    batch_size=2
    sample_model = CaptioningRNN(word_to_idx=data['word_to_idx'],input_dim=input_dim, 
                          wordvec_dim=wordvec_dim,hidden_dim=hidden_dim,batch_size=batch_size,seq_length = 1,num_layers=num_layers)
     
    with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(ckpt_save_dir)
        max_iteration_chekpoint = get_max_iteration_checkpoint(ckpt)
        if ckpt and max_iteration_chekpoint:
            saver.restore(sess, max_iteration_chekpoint)
             
            for split in ['train', 'val']:
                minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
                gt_captions, features, urls = minibatch
                gt_captions = decode_captions(gt_captions, data['idx_to_word'])
              
                sample_captions = sample_model.sample(sess,features)
                sample_captions = decode_captions(sample_captions, data['idx_to_word'])
              
                for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
                    plt.imshow(image_from_url(url))
                    plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
                    plt.axis('off')
                    plt.show()
                print(split, sample_caption,"\n--->", gt_caption)          
        sess.close()
    
elif Mode==2:  
    tf.reset_default_graph()
    with tf.Session() as sess:
        batch_size=1000
        sample_model = CaptioningRNN(word_to_idx=data['word_to_idx'],input_dim=input_dim, 
                              wordvec_dim=wordvec_dim,hidden_dim=hidden_dim,batch_size=batch_size,seq_length = 1,num_layers=num_layers)      
        
        #tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(ckpt_save_dir)
        max_iteration_chekpoint = get_max_iteration_checkpoint(ckpt)
        if ckpt and max_iteration_chekpoint:
            saver.restore(sess, max_iteration_chekpoint)
            evaluate_model_tf(sess,sample_model,data,batch_size)   
        sess.close()
else:
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        ckpt = tf.train.get_checkpoint_state(ckpt_save_dir)
        max_iteration_chekpoint = get_max_iteration_checkpoint(ckpt)
        if ckpt and max_iteration_chekpoint:
            saver = tf.train.import_meta_graph(max_iteration_chekpoint+'.meta')
            saver.restore(sess,max_iteration_chekpoint)
        
            graph = tf.get_default_graph().as_graph_def()
            
            for node in graph.node:
                print(node.name)

        


    
###########################################





e = time.time()
print(e-s,"sec")


